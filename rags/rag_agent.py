import asyncio
import threading

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.base import BaseStore
from psycopg.types import uuid
from langgraph.prebuilt import tools_condition, ToolNode
from html import escape
from typing import Literal,Optional
from langgraph.graph import StateGraph, START, END
from langgraph.store.postgres import PostgresStore
from psycopg import OperationalError
from psycopg_pool import ConnectionPool

from agentmanage.db_agent import db_mcp_agent
from configs.config import Config
from dbmanage.pgsql import ConnectionPoolError, test_connection
from llms.llm_models import MessageState, DocumentRelevanceScore, get_lastest_question, create_chain
from tools.ToolConfig import ToolConfig, ParallelToolNode
from tools.mcp_configs import get_mcp_tools
from utils.logger_util import LoggerUtil

logger = LoggerUtil.setup_logger(__name__)

_mcp_tools_cache = None
def store_memory(question: BaseMessage, config: RunnableConfig, store: BaseStore) ->str:
    """存储用户输入中的记忆信息。

    Args:
        question: 用户输入的消息。
        config: 运行时配置。
        store: 数据存储实例。

    Returns:
        str: 用户相关的记忆信息字符串。
    """
    namespace = ("memories", config["configurable"]["user_id"])
    try:
        # 在跨线程存储数据库中搜索相关记忆
        memories = store.search(namespace, query=str(question.content))
        user_info = "\n".join([d.value["data"] for d in memories])
        # 如果包含“记住”，存储新记忆
        if "记住" in question.content:
            memory = escape(question.content)
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
            logger.info(f"Stored memory: {memory}")
        return user_info
    except Exception as e:
        logger.error(f"Error in store_memory: {e}")
        return ""


# 定义线程内的持久化存储消息过滤函数
def filter_messages(messages: list) -> list:
    """过滤消息列表，仅保留 AIMessage 和 HumanMessage 类型消息"""
    # 过滤出 AIMessage 和 HumanMessage 类型的消息
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    # 如果过滤后的消息超过N条，返回最后N条，否则返回过滤后的完整列表
    return filtered[-5:] if len(filtered) > 5 else filtered



def agent(state: MessageState, config: RunnableConfig, *, store: BaseStore, llm_chat, tool_config: ToolConfig) -> dict:
    """代理函数，根据用户问题决定是否调用工具或结束。

    Args:
        state: 当前对话状态。
        config: 运行时配置。
        store: 数据存储实例。
        llm_chat: Chat模型。
        tool_config: 工具配置参数。

    Returns:
        dict: 更新后的对话状态。
    """
    # 记录代理开始处理查询
    logger.info("Agent processing user query")
    # 定义存储命名空间，使用用户ID
    namespace = ("memories", config["configurable"]["user_id"])

    try:
        # 获取最后一条消息即用户问题
        question = state["messages"][-1]
        logger.info(f"agent question:{question}")
        # 自定义跨线程持久化存储记忆并获取相关信息
        user_info = store_memory(question, config, store)
        # 自定义线程内存储逻辑 过滤消息
        messages = filter_messages(state["messages"])

        # 将工具绑到LLM
        llm_chat_with_tool = llm_chat.bind_tools(tool_config.get_tools())

        # 创建代理处理链
        agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_RAG_AGENT)
        # 调用代理链处理消息
        response = agent_chain.invoke({"question": question, "messages": messages, "userInfo": user_info})
        logger.info(f"Agent response: {response}")
        # 返回更新后的对话状态
        # response_dict = response.dict()
        # logger.info(f"Agent response: {response_dict}")
        return {"messages": [response]}
    except Exception as e:
        # 记录错误日志
        logger.error(f"Error in agent processing: {e}")
        # 返回错误消息
        return {"messages": [{"role": "system", "content": "处理请求时出错"}]}
def rewrite(state: MessageState, llm_chat) -> dict:
    """重写用户查询以改进问题。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """

    logger.info("-------------------------------Rewriting user query...")
    try:
        question = get_lastest_question(state)
        # 重写处理链
        rewrite_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        # 调用重写后的处理链生成新查询
        response = rewrite_chain.invoke({"question": question})
        rewrite_count = state.get("rewrite_count", 0) + 1
        logger.info(f"-------------------------------Rewrite count: {rewrite_count}")
        # 返回更新后的对话状态
        return {"messages": [response], "rewrite_count": rewrite_count}
    except Exception as e:
        # 记录错误日志
        logger.error(f"Message access error in rewrite: {e}")
        # 返回错误消息
        return {"messages": [{"role": "system", "content": "无法重写查询"}]}


def generate(state: MessageState, llm_chat) -> dict:
    """基于工具返回的内容生成最终回复。

    Args:
        state: 当前对话状态。

    Returns:
        dict: 更新后的消息状态。
    """
    # 记录开始生成回复
    logger.info("-------------------------------Generating response")
    try:
        question = get_lastest_question(state)
        context = state["messages"][-1].content
        all_messages = state.get("messages", [])
        logger.info(f"-------------------------------Generating response, question:{question}, all_messages:{all_messages}")

        generate_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_DB_GENERATE)
        response = generate_chain.invoke({"context": context, "question": question})
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Failed to get latest question: {e}")
        return {"messages": [{"role": "system", "content": "无法生成回复"}]}

def mcp_generate(state: MessageState, llm_chat) -> dict:
    """ 根据rag_agent的结果，判断是否需要调用mcp工具
    :param state:
    :param llm_chat:
    :return:
    """
    logger.info(f"-------------------------------mcp_generate")
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {
            "messages": [{"role": "system", "content": "状态为空，无法生成"}],
            "relevance_score": None
        }
    try:
        question = get_lastest_question(state)
        context = state["messages"][-1].content
        grade_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        # 调用评分链评估相关性
        scored_result = grade_chain.invoke({"question": question, "context": context})
        # logger.info(f"scored_result:{scored_result}")
        # 获取评分结果
        score = scored_result.binary_score
        logger.info(f"Document relevance score: {score}")
        # 返回更新后的状态，包括评分结果
        return {
            # 保持消息不变
            "messages": state["messages"],
            # 存储评分结果
            "relevance_score": score
        }
    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        return {
            "messages": [{"role": "system", "content": "创建图失败"}],
            "relevance_score": None
        }


def grade_documents(state: MessageState, llm_chat) -> dict:
    """评估检索到的文档内容与问题的相关性，并将评分结果存储在状态中。

    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        dict: 更新后的状态，包含评分结果。
    """
    logger.info("Grading documents for relevance")
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {
            "messages": [{"role": "system", "content": "状态为空，无法评分"}],
            "relevance_score": None
        }

    try:
        question = get_lastest_question(state)
        context = state["messages"][-1].content
        grade_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        # 调用评分链评估相关性
        scored_result = grade_chain.invoke({"question": question, "context": context})
        # logger.info(f"scored_result:{scored_result}")
        # 获取评分结果
        score = scored_result.binary_score
        logger.info(f"Document relevance score: {score}")
        # 返回更新后的状态，包括评分结果
        return {
            # 保持消息不变
            "messages": state["messages"],
            # 存储评分结果
            "relevance_score": score
        }
    except Exception as e:
        logger.error(f"Failed to create graph: {e}")
        return {
            "messages": [{"role": "system", "content": "创建图失败"}],
            "relevance_score": None
        }


def route_after_tools(state: MessageState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    """
    根据工具调用的结果动态决定下一步路由，使用配置字典支持多工具并包含容错处理。

    Args:
        state: 当前对话状态，包含消息历史和可能的工具调用结果。
        tool_config: 工具配置参数。

    Returns:
        Literal["generate", "grade_documents"]: 下一步的目标节点。
    """
    # 检查状态是否包含消息列表，若为空则记录错误并默认路由到 generate
    if not state.get("messages") or not isinstance(state["messages"], list):
        logger.error("Messages state is empty or invalid, defaulting to generate")
        return "generate"
    try:
        last_message = state["messages"][-1]
        logger.info(f"-------------------------------Last message: {last_message}")
        if not hasattr(last_message, "name") or last_message.name is None:
            logger.error("Last message does not have a name, defaulting to generate")
            return "generate"

        tool_name = last_message.name
        if tool_name not in tool_config.get_tool_names():
            logger.error(f"Invalid tool name: {tool_name}, defaulting to generate")
            return "generate"

        # 根据配置字典决定路由，若无配置则默认路由到 generate
        target = tool_config.get_tool_routing_config().get(tool_name, "generate")
        logger.info(f"-------------------------------Tool {tool_name} routed to {target} based on config")
        return target

    except IndexError:
        # 捕获消息列表为空或索引错误的异常，记录错误并默认路由到 generate
        logger.error("No messages available in state, defaulting to generate")
        return "generate"
    except AttributeError:
        # 捕获消息对象属性访问错误的异常，记录错误并默认路由到 generate
        logger.error("Invalid message object, defaulting to generate")
        return "generate"
    except Exception as e:
        # 捕获其他未预期的异常，记录详细错误信息并默认路由到 generate
        logger.error(f"Unexpected error in route_after_tools: {e}, defaulting to generate")
        return "generate"


def route_after_grade(state: MessageState) -> Literal["generate", "rewrite"]:
    """
    根据状态中的评分结果决定下一步路由，包含增强的状态校验和容错处理。

    Args:
        state: 当前对话状态，预期包含 messages 和 relevance_score 字段。

    Returns:
        Literal["generate", "rewrite"]: 下一步的目标节点。
    """
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "rewrite"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "rewrite"
    # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "rewrite"
    # 获取状态中的 relevance_score，若不存在则返回 None
    relevance_score = state.get("relevance_score")
    # 获取状态中的 rewrite_count
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"Routing based on relevance_score: {relevance_score}, rewrite_count: {rewrite_count}")
    # 如果重写次数超过 3 次，强制路由到 generate
    if rewrite_count >= 3:
        logger.info("Max rewrite limit reached, proceeding to generate")
        return "generate"
    try:
        # 检查 relevance_score 是否为有效字符串，若不是则视为无效评分
        if not isinstance(relevance_score, str):
            logger.warning(f"Invalid relevance_score type: {type(relevance_score)}, defaulting to rewrite")
            return "rewrite"

        # 如果评分结果为 "yes"，表示文档相关，路由到 generate 节点
        if relevance_score.lower() == "yes":
            logger.info("Documents are relevant, proceeding to generate")
            return "generate"

        # 如果评分结果为 "no" 或其他值（包括空字符串），路由到 rewrite 节点
        logger.info("Documents are not relevant or scoring failed, proceeding to rewrite")
        return "rewrite"
    except AttributeError:
        # 捕获 relevance_score 不支持 lower() 方法的异常（例如 None），默认路由到 rewrite
        logger.error("relevance_score is not a string or is None, defaulting to rewrite")
        return "rewrite"
    except Exception as e:
        # 捕获其他未预期的异常，记录详细错误信息并默认路由到 rewrite
        logger.error(f"Unexpected error in route_after_grade: {e}, defaulting to rewrite")
        return "rewrite"


# 在 create_graph 函数内部，添加以下异步函数定义
def db_mcp_agent_node(state, config, store, llm_chat, mcp_tools):
    return db_mcp_agent(state, config, store=store, llm_chat=llm_chat, mcp_tools=mcp_tools)

def create_graph(db_connection_pool: ConnectionPool, llm_chat, llm_embedding, tool_config: ToolConfig) -> StateGraph:
    """创建并配置状态图。

    Args:
        db_connection_pool: 数据库连接池。
        llm_chat: Chat模型。
        llm_embedding: Embedding模型。
        tool_config: 工具配置参数。

    Returns:
        StateGraph: 编译后的状态图。

    Raises:
        ConnectionPoolError: 如果连接池未正确初始化或状态异常。
    """

    # 检查连接池是否为None或未打开
    if db_connection_pool is None or db_connection_pool.closed:
        logger.error("Connection db_connection_pool is None or closed")
        raise ConnectionPoolError("数据库连接池未初始化或已关闭")
    try:
        # 获取当前活动连接数和最大连接数
        active_connections = db_connection_pool.get_stats().get("connections_in_use", 0)
        max_connections = db_connection_pool.max_size
        if active_connections >= max_connections:
            logger.error(
                f"Connection db_connection_pool exhausted: {active_connections}/{max_connections} connections in use")
            raise ConnectionPoolError("连接池已耗尽，无可用连接")
        if not test_connection(db_connection_pool):
            raise ConnectionPoolError("连接池测试失败")
        logger.info("Connection db_connection_pool status: OK, test connection successful")
    except OperationalError as e:
        logger.error(f"Database operational error during connection test: {e}")
        raise ConnectionPoolError(f"连接池测试失败，可能已关闭或超时: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to verify connection db_connection_pool status: {e}")
        raise ConnectionPoolError(f"无法验证连接池状态: {str(e)}")

    # 线程内持久化存储
    try:
        # 创建Postgres检查点保存实例
        checkpointer = PostgresSaver(db_connection_pool)
        # 初始化检查点
        checkpointer.setup()
    except Exception as e:
        logger.error(f"Failed to setup PostgresSaver: {e}")
        raise ConnectionPoolError(f"检查点初始化失败: {str(e)}")

    try:
        store = PostgresStore(db_connection_pool, index={"dims": 1536, "embed": llm_embedding})
        store.setup()
    except Exception as e:
        logger.error(f"Failed to setup PostgresStore: {e}")
        raise ConnectionPoolError(f"存储初始化失败: {str(e)}")
    # 创建状态图实例，使用MessagesState作为状态类型
    workflow = StateGraph(MessageState)

    # 修改 mcp_tools 节点，创建一个异步处理函数
    async def async_mcp_tools_node(state, config):
        global _mcp_tools_cache

        # 每次都重新获取工具，避免协程复用问题
        if _mcp_tools_cache is None:
            mcp_tools = get_mcp_tools()  # 每次都重新获取
            if asyncio.iscoroutine(mcp_tools):
                _mcp_tools_cache = await mcp_tools
            else:
                _mcp_tools_cache = mcp_tools

        tools = _mcp_tools_cache
        # 创建工具节点并执行
        tool_node = ToolNode(tools)
        return await tool_node.ainvoke(state, config)
    workflow.add_node("rag_agent", lambda state, config: agent(state, config, store=store, llm_chat=llm_chat, tool_config=tool_config))
    # 添加工具节点，使用并行工具节点
    workflow.add_node("rag_tools", ParallelToolNode(tool_config.get_tools(), max_workers=5))
    # 添加重写节点
    # workflow.add_node("rewrite", lambda state: rewrite(state,llm_chat=llm_chat))
    # db mcp 节点
    workflow.add_node("db_mcp_agent", lambda state, config: db_mcp_agent_node(state, config, store, llm_chat, get_mcp_tools()))
    workflow.add_node("mcp_tools", lambda state, config: asyncio.run(async_mcp_tools_node(state, config)))
    # 添加生成节点
    workflow.add_node("generate", lambda state: generate(state, llm_chat=llm_chat))


    # 添加从起始到rag代理的边
    workflow.add_edge(START, end_key="rag_agent")
    # 添加rag代理的条件边，根据工具调用的工具名称决定下一步路由
    workflow.add_conditional_edges(source="rag_agent", path=tools_condition, path_map={"tools": "rag_tools", END: END})
    # 添加检索的条件边，根据工具调用的结果动态决定下一步路由
    workflow.add_conditional_edges(source="rag_tools", path=lambda state: route_after_tools(state, tool_config),path_map={"generate": "generate", "db_mcp_agent": "db_mcp_agent"})
    workflow.add_conditional_edges(source="db_mcp_agent", path=tools_condition, path_map={"tools": "mcp_tools", END: END})
    workflow.add_edge(start_key="mcp_tools", end_key="generate")
    # 添加检索的条件边，根据状态中的评分结果决定下一步路由
    # workflow.add_conditional_edges(source="grade_documents", path=lambda state: route_after_grade(state), path_map={"generate": "generate", "rewrite": "rewrite"})
    # 添加从重写到代理的边
    # orkflow.add_edge(start_key="db_agent", end_key="rag_agent")

    # 编译状态图，绑定检查点和存储
    return workflow.compile(store=store)


def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """保存状态图的可视化表示。

    Args:
        graph: 状态图实例。
        filename: 保存文件路径。
    """
    # 尝试执行以下代码块
    try:
        # 以二进制写模式打开文件
        with open(filename, "wb") as f:
            # 将状态图转换为Mermaid格式的PNG并写入文件
            f.write(graph.get_graph().draw_mermaid_png())
        # 记录保存成功的日志
        logger.info(f"Graph visualization saved as {filename}")
    # 捕获IO错误
    except IOError as e:
        # 记录警告日志
        logger.warning(f"Failed to save graph visualization: {e}")


async def graph_response(graph: StateGraph, user_input: str, config: dict, tool_config: ToolConfig) -> str:
    """
    处理用户输入并输出响应，区分工具输出和大模型输出，支持多工具。

    Args:
        graph: 状态图实例。
        user_input: 用户输入。
        config: 运行时配置。
        tool_config: 工具配置

    Returns:
        str: 最终的响应内容
    """
    final_response = ""
    try:
        final_response = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # for event in events:
        #     for value in event.values():
        #         # 检查是否有有效消息
        #         if "messages" not in value or not isinstance(value["messages"], list):
        #             logger.warning("No valid messages in response")
        #             continue
        #         # 获取最后一条消息
        #         last_message = value["messages"][-1]
        #         if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        #             for tool_call in last_message.tool_calls:
        #                 if isinstance(tool_call, dict) and "name" in tool_call:
        #                     logger.info(f"Calling tool: {tool_call['name']}")
        #                 elif hasattr(tool_call, "name"):  # 处理对象形式的tool_call
        #                     logger.info(f"Calling tool: {tool_call.name}")
        #             continue
        #
        #         if hasattr(last_message, "content"):
        #             content = last_message.content
        #             # 情况1：工具输出（动态检查工具名称）
        #             if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
        #                 tool_name = last_message.name
        #                 print(f"Tool Output [{tool_name}]: {content}")
        #                 # 不更新final_response，因为这是工具输出
        #             # 情况2：大模型输出（非工具消息）
        #             else:
        #                 print(f"Assistant: {content}")
        #                 # 更新最终响应
        #                 final_response = content
        #         else:
        #             # 如果消息没有内容，可能是中间状态
        #             logger.info("Message has no content, skipping")
        #             print("Assistant: 未获取到相关回复")

    except Exception as e:
        logger.error(f"Error in graph_response: {e}", exc_info=True)
        final_response = f"错误: {str(e)}"

    return final_response if final_response else "未获取到相关回复"


