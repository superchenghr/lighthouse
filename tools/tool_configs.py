from langchain_chroma import Chroma
from langchain_core.tools import tool

from configs.config import Config
from langchain.tools.retriever import create_retriever_tool


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

    retriever = verctorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是健康档案查询工具，搜索并返回有关用户的健康档案信息。"
    )

    @tool()
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具，返回最终的计算结果"""
        return a * b

    # 生成一个加法的工具，命名为sum
    # 返回工具列表
    return [retriever_tool, multiply]