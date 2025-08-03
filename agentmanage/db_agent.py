import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from configs.config import Config
from llms.llm_models import MessageState, get_lastest_question, create_chain
from utils.logger_util import LoggerUtil

logger = LoggerUtil.setup_logger(__name__)


async def _db_mcp_agent_async(state: MessageState, config: RunnableConfig, *, store: BaseStore, llm_chat,
                              mcp_tools) -> dict:
    last_message = state["messages"][-1]
    question = get_lastest_question(state)
    logger.info(f"-------------------------------db_mcp_agent, last_message:{last_message}")
    logger.info(f"-------------------------------db_mcp_agent, question:{question}")

    try:
        # 获取实际的工具列表
        if asyncio.iscoroutine(mcp_tools):
            tools = await mcp_tools
        else:
            tools = mcp_tools
        logger.info(f"-------------------------------db_mcp_agent, tools:{tools}")
        # 将工具绑定到 LLM
        llm_chat_with_tool = llm_chat.bind_tools(tools)

        # 创建代理处理链
        agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_DB_GENERATE)

        # 使用异步调用
        response = await agent_chain.ainvoke({"question": question, "context": last_message})
        logger.info(f"-------------------------------mcp_agent response:{response}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in agent processing: {e}")
        raise


def db_mcp_agent(state: MessageState, config: RunnableConfig, *, store: BaseStore, llm_chat, mcp_tools) -> dict:
    """同步包装器，用于在同步环境中运行异步代理"""
    try:
        # 在新事件循环中运行异步函数
        return asyncio.run(_db_mcp_agent_async(state, config, store=store, llm_chat=llm_chat, mcp_tools=mcp_tools))
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        raise
