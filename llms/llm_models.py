import threading

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from utils.logger_util import LoggerUtil

logger = LoggerUtil.setup_logger("main")

class MessageState(TypedDict):
    # 定义messages字段，类型为消息序列，使用add_messages处理追加
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 定义relevance_score字段，用于存储文档相关性评分
    relevance_score: Annotated[Optional[str],  "Relevance score of retrieved documents, 'yes' or 'no'"]
    # 定义rewrite_count字段，用于跟踪问题重写的次数，达到次数退出graph的递归循环
    rewrite_count: Annotated[int, "Number of times query has been rewritten"]

# 文档相关性评分
class DocumentRelevanceScore(BaseModel):
    # 定义binary_score字段，表示相关性评分，取值为"yes"或"no"
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


def get_lastest_question(state: MessageState) -> Optional[str]:
    """从状态中安全地获取最新用户问题。

    Args:
        state: 当前对话状态，包含消息历史。

    Returns:
        Optional[str]: 最新问题的内容，如果无法获取则返回 None。
    """
    try:
        if not state.get("messages") or not isinstance(state["messages"], (list, tuple)) or len(state["messages"]) == 0:
            logger.warning("No valid messages found in state for getting latest question")
            return None

        for messsage in reversed(state["messages"]):
            if messsage.__class__.__name__ == "HumanMessage" and hasattr(messsage, "content"):
                return messsage.content
        # 如果没有找到 HumanMessage，返回 None
        logger.info
        return None
    except Exception as e:
        logger.error(f"Error getting latest question: {e}")
        return None


def create_chain(llm_chat, template_file: str, structured_output=None):
    """创建 LLM 处理链，加载提示模板并绑定模型，使用缓存避免重复读取文件。

    Args:
        llm_chat: 语言模型实例。
        template_file: 提示模板文件路径。
        structured_output: 可选的结构化输出模型。

    Returns:
        Runnable: 配置好的处理链。

    Raises:
        FileNotFoundError: 如果模板文件不存在。
    """

    if not hasattr(create_chain, "prompt_cache"):
        # 缓存字典
        create_chain.prompt_cache = {}
        # 线程锁 确保缓存的读写是线程安全的
        create_chain.lock = threading.Lock()
    try:
        # 先检查缓存，无锁访问
        if template_file in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[template_file]
            logger.info(f"Using cached prompt template for {template_file}")
        else:
            with create_chain.lock:
                # 检查缓存中是否已有该模板
                if template_file not in create_chain.prompt_cache:
                    logger.info(f"Loading and caching prompt template from {template_file}")
                    # 从文件加载提示模板并存入缓存
                    create_chain.prompt_cache[template_file] = PromptTemplate.from_file(template_file, encoding="utf-8")
                # 从缓存中获取提示模板
                prompt_template = create_chain.prompt_cache[template_file]
        # 创建聊天提示模板，使用模板内容
        prompt = ChatPromptTemplate.from_messages([("human", prompt_template.template)])
        # 返回提示模板与LLM的组合链，若有结构化输出则绑定
        return prompt | (llm_chat.with_structured_output(structured_output) if structured_output else llm_chat)
    except FileNotFoundError as e:
        logger.error(f"Template file {template_file} not found")
        raise

