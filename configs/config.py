import os

class Config:
    """统一的配置类，集中管理所有常量"""
    # prompt文件路径
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "/Users/chenghaoran/PycharmProjects/lighthouse/chromaDB"
    CHROMADB_COLLECTION_NAME = "lighthouse"

    # 日志持久化存储
    LOG_FILE = "output/app.log"
    MAX_BYTES = 5 * 1024 * 1024,
    BACKUP_COUNT = 3

    # 数据库 URI，默认值
    DB_URI = os.getenv("DB_URI", "postgresql://localhost:5432/lighthouse?sslmode=disable")

    LLM_TYPE = "qwen"

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012

