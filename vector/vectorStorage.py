import os
import logging
from openai import OpenAI
from utils import pdfSplitTest_Ch
from MyVectorDBConnector import MyVectorDBConnector
import configs.config as conf
from dotenv import load_dotenv
load_dotenv()

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置测试文本类型 Chinese 或 English
TEXT_LANGUAGE = 'Chinese'
# get_embeddings方法计算向量
def get_embeddings(llmType, texts):
    print(texts)
    if llmType == 'qwen':
        try:
            client = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY")
            )
            data = client.embeddings.create(input=texts,model="text-embedding-v1").data
            print(data)
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'zhipu':
        try:
            client = OpenAI(
                base_url="https://open.bigmodel.cn/api/paas/v4",
                api_key=os.getenv("ZHIPUAI_API_KEY")
            )
            data = client.embeddings.create(input=texts,model="embedding-3").data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
    else:
        logger.error("未定义的LLM类型")


# 对文本按批次进行向量计算
def generate_vectors(data, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        # 调用向量生成get_embeddings方法  根据调用的API不同进行选择
        response = get_embeddings("qwen", batch)
        results.extend(response)
    return results

def vectorStoreSave(path, page):
    # 1、获取处理后的文本数据
    # 演示测试对指定的全部页进行处理，其返回值为划分为段落的文本列表
    paragraphs = pdfSplitTest_Ch.getParagraphs(
        filename=path,
        page_numbers=page,
        min_line_length=1
    )
    # 2、将文本片段灌入向量数据库
    # 实例化一个向量数据库对象
    # 其中，传参collection_name为集合名称, embedding_fn为向量处理函数
    vector_db = MyVectorDBConnector(conf.Config.CHROMADB_COLLECTION_NAME, generate_vectors)
    # 向向量数据库中添加文档（文本数据、文本数据对应的向量数据）
    vector_db.add_documents(paragraphs)
    # 3、封装检索接口进行检索测试
    user_query = "张三九的基本信息是什么"
    # 将检索出的5个近似的结果
    search_results = vector_db.search(user_query, 5)
    logger.info(f"检索向量数据库的结果: {search_results}")

if __name__ == "__main__":
    # 测试文本预处理及灌库
    vectorStoreSave("/Users/chenghaoran/PycharmProjects/lighthouse/input/健康档案.pdf", None)