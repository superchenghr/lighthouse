import sys
from typing import Optional, Dict, Any, Generator
from psycopg_pool import ConnectionPool
from configs.config import Config
from dbmanage.pgsql import ConnectionPoolError, monitor_connection_pool
from llms.llm import get_llm
from rags.ragAgent import create_graph, graph_response
from tools.ToolConfig import ToolConfig, logger
from tools.tool_configs import get_tools
from dotenv import load_dotenv

load_dotenv()


class ChatBotInterface:
    """
    聊天机器人通用接口类
    支持多用户、多线程、多模型的聊天机器人接口
    """

    def __init__(self):
        """初始化聊天机器人接口"""
        self.db_connection_pool = None
        self.graph = None
        self.monitor_thread = None
        self._initialized = False
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # thread_id: {user_id, model, config}

    def initialize(self) -> bool:
        """
        初始化聊天机器人系统

        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            return True

        try:
            # 初始化大模型
            llm_chat, llm_embedding = get_llm("qwen")
            # 初始化工具集
            tools = get_tools(llm_embedding)  # llm_embedding会在创建graph时传入
            # 创建 ToolConfig 实例
            self.tool_config = ToolConfig(tools)

            # 定义数据库连接参数
            connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
            self.db_connection_pool = ConnectionPool(
                conninfo=Config.DB_URI,
                max_size=20,
                min_size=2,
                kwargs=connection_kwargs,
                timeout=10
            )

            try:
                self.db_connection_pool.open()
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to open connection pool: {e}")
                raise ConnectionPoolError(f"无法打开数据库连接池: {str(e)}")

            # 启动连接池监控
            self.monitor_thread = monitor_connection_pool(self.db_connection_pool, interval=60)
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ChatBotInterface: {e}")
            return False

    def create_session(
            self,
            user_id: str,
            thread_id: str,
            model_name: str
    ) -> bool:
        """
        创建一个新的聊天会话

        Args:
            user_id (str): 用户ID
            thread_id (str): 线程ID
            model_name (str): 模型名称，默认为"zhipu"

        Returns:
            bool: 会话创建是否成功
        """
        try:
            if not self._initialized:
                if not self.initialize():
                    return False

            # 初始化指定的大模型
            llm_chat, llm_embedding = get_llm(model_name)

            # 创建graph（每个会话可以使用不同模型）
            graph = create_graph(self.db_connection_pool, llm_chat, llm_embedding, self.tool_config)

            # 保存会话配置
            session_config = {
                "user_id": user_id,
                "thread_id": thread_id,
                "model_name": model_name,
                "graph": graph,
                "llm_chat": llm_chat,
                "llm_embedding": llm_embedding,
                "config": {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            }

            self.active_sessions[thread_id] = session_config
            logger.info(f"Session created for user {user_id} with thread {thread_id} using model {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            return False

    def chat(
            self,
            user_input: str,
            user_id: str,
            thread_id: str,
            model_name: str,
            stream: bool = False
    ) -> Generator[str, None, None] | str:
        """
        与聊天机器人交互

        Args:
            user_input (str): 用户输入
            user_id (str): 用户ID
            thread_id (str): 线程ID
            model_name (str): 模型名称
            stream (bool): 是否流式输出

        Yields:
            str: 机器人回复内容

        Returns:
            str: 机器人回复内容（非流式）
        """
        try:
            import langchain
            langchain.debug = True
            # 检查会话是否存在，不存在则创建
            if thread_id not in self.active_sessions:
                if not self.create_session(user_id, thread_id, model_name):
                    raise Exception("Failed to create session")

            session = self.active_sessions[thread_id]
            graph = session["graph"]
            config = session["config"]

            # 处理用户输入
            if stream:
                # 流式输出实现需要在 graph_response 中支持
                # 这里假设 graph_response 可以返回生成器
                def response_generator():
                    full_response = ""
                    try:
                        # 这里需要修改 graph_response 来支持流式输出
                        # 暂时以普通方式处理
                        response = graph_response(graph, user_input, config, self.tool_config)
                        yield response
                    except Exception as e:
                        logger.error(f"Error in chat stream: {e}")
                        yield f"错误: {str(e)}"

                return response_generator()
            else:
                # 普通响应
                response = graph_response(graph, user_input, config, self.tool_config)
                return response

        except Exception as e:
            logger.error(f"Chat error for user {user_id}: {e}")
            error_msg = f"错误: {str(e)}"
            if stream:
                def error_generator():
                    yield error_msg

                return error_generator()
            else:
                return error_msg

    def end_session(self, thread_id: str) -> bool:
        """
        结束指定会话

        Args:
            thread_id (str): 线程ID

        Returns:
            bool: 是否成功结束会话
        """
        try:
            if thread_id in self.active_sessions:
                del self.active_sessions[thread_id]
                logger.info(f"Session ended for thread {thread_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error ending session {thread_id}: {e}")
            return False

    def is_session_active(self, thread_id: str) -> bool:
        """
        检查会话是否活跃

        Args:
            thread_id (str): 线程ID

        Returns:
            bool: 会话是否活跃
        """
        return thread_id in self.active_sessions

    def cleanup(self):
        """清理资源"""
        try:
            # 结束所有会话
            self.active_sessions.clear()

            # 关闭数据库连接池
            if self.db_connection_pool and not self.db_connection_pool.closed:
                self.db_connection_pool.close()
                logger.info("Database connection pool closed")

            self._initialized = False
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# 兼容旧的main函数调用方式
def run_chatbot_interface():
    """运行聊天机器人接口（兼容旧版本）"""
    chatbot = ChatBotInterface()

    try:
        if not chatbot.initialize():
            print("Failed to initialize chatbot")
            sys.exit(1)

        # 创建默认会话
        if not chatbot.create_session("1", "1", "zhipu"):
            print("Failed to create default session")
            sys.exit(1)

        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")

        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            if not user_input:
                print("请输入聊天内容！")
                continue

            response = chatbot.chat(user_input, "1", "1", "zhipu")
            print(f"机器人: {response}")

    except KeyboardInterrupt:
        print("\n被用户打断。再见！")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"错误: 发生未知错误 - {e}")
    finally:
        chatbot.cleanup()
