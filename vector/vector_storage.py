from langchain_core.documents import Document
from langchain_openai import OpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter

from utils.logger_util import LoggerUtil
import os

from vector.MyVectorDBConnector import MyVectorDBConnector
import configs.config as conf
from vector.vectorStorage import generate_vectors

logger = LoggerUtil.setup_logger(__name__)
from dotenv import load_dotenv
load_dotenv()

def get_embeddings(texts):
    try:
        client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        data = client.embeddings.create(input=texts, model="text-embedding-v1").data
        return [x.embedding for x in data]
    except Exception as e:
        logger.info(f"生成向量时出错: {e}")
        return []

headers_to_split_on = [
    ("#", "Header")
]
file_path = "../input/问题2.md"
with open(file_path, 'r', encoding='utf-8') as file:
    markdown_document = file.read()
# 初始化分割器（默认移除标题文本）
splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
splits = splitter.split_text(markdown_document)


final_doc = Document(page_content="")
final_metadata = dict()
for doc in splits:
    if "关键词" in doc.metadata["Header"]:
        # 将doc.page_content按逗号切割
        keywords = doc.page_content.split(",")
        final_metadata["keywords"] = ", ".join(keywords)
    if "问题" in doc.metadata["Header"]:
        final_doc.page_content = doc.page_content
    if "根本原因" in doc.metadata["Header"]:
        final_metadata["reason"] = doc.page_content
    if "关联反馈单" in doc.metadata["Header"]:
        final_metadata["uri"] = doc.page_content
    if "工具" in doc.metadata["Header"]:
        final_metadata["tools"] = doc.page_content
    if "排查步骤" in doc.metadata["Header"]:
        final_metadata["steps"] = doc.page_content

vector_db = MyVectorDBConnector(conf.Config.CHROMADB_COLLECTION_NAME, generate_vectors)
# 向向量数据库中添加文档（文本数据、文本数据对应的向量数据）
vector_db.add_documents([final_doc], [final_metadata])
result = vector_db.search(query="客服发送消息时，提示：发送失败，消息发送者不能为空",top_n=2)
print(result)

