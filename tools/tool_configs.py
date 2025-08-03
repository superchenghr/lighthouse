from langchain_chroma import Chroma
from langchain_core.tools import tool

from configs.config import Config
from utils.logger_util import LoggerUtil

logger = LoggerUtil.setup_logger(__name__)

def get_tools(llm_embedding):
    """
        创建并返回工具列表

        Args:
            llm_embedding: 嵌入模型实例，用于初始化向量存储

        Returns:
            list: 工具列表
        """

    verctorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )

    # retriever = verctorstore.as_retriever()
    #
    # retriever_tool = create_retriever_tool(
    #     retriever,
    #     name="retrieve",
    #     description="这是问题沉淀记录工具，搜索并返回有关问题的记录。"
    # )

    # 自定义检索工具，返回完整的元数据
    @tool
    def retrieve(query: str) -> str:
        """
        这是问题沉淀记录工具，搜索并返回有关问题的记录，包含详细元数据信息。

        Args:
            query: 搜索查询

        Returns:
            str: 包含问题记录和元数据的格式化字符串
        """
        # 获取检索结果，包含元数据
        results = verctorstore.similarity_search_with_score(query, k=2)

        if not results:
            return "未找到相关问题记录"

        formatted_results = []
        for doc, score in results:
            # 提取元数据
            metadata = doc.metadata

            # 格式化结果
            result_str = f"问题: {doc.page_content}\n"
            result_str += f"相似度得分: {score}\n"

            # 添加元数据中的详细信息
            if 'keywords' in metadata:
                result_str += f"关键词: {metadata['keywords']}\n"
            if 'reason' in metadata:
                result_str += f"原因: {metadata['reason']}\n"
            if 'steps' in metadata:
                result_str += f"解决步骤: {metadata['steps']}\n"
            if 'tools' in metadata:
                result_str += f"使用工具: {metadata['tools']}\n"

            formatted_results.append(result_str)
            logger.info(f"--------------------------Formatted result: {result_str}")
        return "\n---\n".join(formatted_results)

    @tool()
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具，返回最终的计算结果"""
        return a * b

    # 生成一个加法的工具，命名为sum
    # 返回工具列表
    return [retrieve, multiply]