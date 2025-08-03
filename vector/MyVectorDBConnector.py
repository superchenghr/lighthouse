import logging
import uuid

import configs.config as conf
import chromadb

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MyVectorDBConnector:

    def __init__(self, collection_name, embedding_fn):
        # 实例化一个chromadb对象
        # 设置一个文件夹进行向量数据库的持久化存储  路径为当前文件夹下chromaDB文件夹
        chroma_client = chromadb.PersistentClient(path=conf.Config.CHROMADB_DIRECTORY)
        # 创建一个collection数据集合
        # get_or_create_collection()获取一个现有的向量集合，如果该集合不存在，则创建一个新的集合
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        # embedding处理函数
        self.embedding_fn = embedding_fn

    def add_documents(self, documents, metadatas):
        texts = [doc.page_content for doc in documents]
        self.collection.add(
            embeddings=self.embedding_fn(texts),
            metadatas=metadatas,
            documents=texts,
            ids=[str(uuid.uuid4()) for i in range(len(documents))]  # 文档的唯一标识符 自动生成uuid,128位
        )

    def get_all_documents(self):
        """
        查询并返回所有文档
        :return: 包含所有文档信息的字典
        """
        try:
            results = self.collection.get()
            return results
        except Exception as e:
            logger.info(f"获取所有文档时出错: {e}")
            return []


    def search(self, query, top_n):
        try:
            results = self.collection.query(
                # 计算查询文本的向量，然后将查询文本生成的向量在向量数据库中进行相似度检索
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n
            )
            return results
        except Exception as e:
            logger.info(f"检索向量数据库时出错: {e}")
            return []

