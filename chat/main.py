import asyncio
import sys

from psycopg_pool import ConnectionPool

from configs.config import Config
from dbmanage.pgsql import test_connection, ConnectionPoolError, monitor_connection_pool
from llms.llm import get_llm
from rags.rag_agent import create_graph, save_graph_visualization, graph_response
from tools.ToolConfig import ToolConfig, logger
from tools.tool_configs import get_tools
from dotenv import load_dotenv
load_dotenv()

def main():
    import langchain
    langchain.debug = True
    """主函数，初始化并运行聊天机器人。"""
    db_connection_pool = None
    try:
        #初始化大模型
        llm_chat, llm_embedding = get_llm("qwen")
        # 初始化工具集
        tools = get_tools(llm_embedding)
        # 创建 ToolConfig 实例
        tool_config = ToolConfig(tools)

        # 定义数据库连接参数，自动提交且无预准备阈值，5秒超时
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        db_connection_pool = ConnectionPool(conninfo=Config.DB_URI, max_size=20, min_size=2, kwargs=connection_kwargs, timeout=10)
        try:
            db_connection_pool.open()
            logger.info("Database connection pool initialized")
            logger.debug("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to open connection pool: {e}")
            raise ConnectionPoolError(f"无法打开数据库连接池: {str(e)}")

        # 启动连接池监控 监控线程为守护线程，随主程序退出而停止
        monitor_thread = monitor_connection_pool(db_connection_pool, interval=60)
        try:
            graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        except ConnectionPoolError as e:
            logger.error(f"Graph creation failed: {e}")
            print(f"错误: {e}")
            sys.exit(1)

        # 保存状态图可视化
        save_graph_visualization(graph)
        # 打印机器人就绪提示
        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        # 定义运行时配置，包含线程ID和用户ID
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            if not user_input:
                print("请输入聊天内容！")
                continue
            # 处理用户输入并选择是否流式输出响应
            asyncio.run(graph_response(graph, user_input, config, tool_config))
    except ConnectionPoolError as e:
        # 捕获连接池相关的异常
        logger.error(f"Connection pool error: {e}")
        print(f"错误: 数据库连接池问题 - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # 捕获键盘中断
        print("\n被用户打断。再见！")
    except Exception as e:
        # 捕获未预期的其他异常
        logger.error(f"Unexpected error: {e}")
        print(f"错误: 发生未知错误 - {e}")
        sys.exit(1)
    finally:
        # 清理资源
        if db_connection_pool and not db_connection_pool.closed:
            db_connection_pool.close()
            logger.info("Database connection pool closed")

if __name__ == '__main__':
    main()