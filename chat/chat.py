import json
import asyncio
import time

import websockets
from session.session_manage import run_chatbot_interface, ChatBotInterface
from utils.logger_util import LoggerUtil

logger = LoggerUtil.setup_logger(__name__)

class WebSocketChatServer:
    """
    WebSocket聊天服务器类
    管理ChatBotInterface实例并在多个客户端间共享
    """

    def __init__(self):
        self.chatbot = ChatBotInterface()
        self.initialized = False

    async def initialize_chatbot(self):
        """
        初始化聊天机器人
        """
        if not self.initialized:
            # 在executor中运行同步的初始化代码
            loop = asyncio.get_event_loop()
            self.initialized = await loop.run_in_executor(None, self.chatbot.initialize)
            if not self.initialized:
                raise Exception("Failed to initialize chatbot")
            logger.info("ChatBotInterface initialized successfully")

    async def handle_client(self, websocket, path=None):
        """
        处理客户端连接
        """
        logger.info(f"新客户端连接: {websocket.remote_address}")
        try:
            # 确保聊天机器人已初始化
            await self.initialize_chatbot()

            async for message in websocket:
                logger.info(f"收到消息: {message}")
                try:
                    # 解析消息为dict
                    data = json.loads(message)

                    # 提取必要参数
                    user_input = data.get("content", "")
                    user_id = data.get("user_id", "default_user")
                    thread_id = data.get("thread_id", "default_thread")
                    model_name = data.get("model_name", "zhipu")
                    end_session = data.get("end_session", False)

                    if end_session:
                        # 结束会话
                        success = self.chatbot.end_session(thread_id)
                        response = {"type": "session_end", "success": success, "message": "会话已结束"}
                        await websocket.send(json.dumps(response))
                        continue
                    # thread_id 为当前时间
                    thread_id = str(int(time.time() * 1000))
                    # 处理聊天消息
                    chat_response = self.chatbot.chat(
                        user_input=user_input,
                        user_id=user_id,
                        thread_id=thread_id,
                        model_name="qwen"
                    )
                    logger.info(f"回复: {chat_response}")
                    # 构造响应
                    row_data = {
                        "type": "chat_response",
                        "content": chat_response,
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "model_name": model_name
                    }
                    await websocket.send('服务端回复:' + json.dumps(row_data, ensure_ascii=False))
                    logger.info("服务端回复:发送回复")

                except json.JSONDecodeError:
                    error_msg = {"type": "error", "message": "无效的JSON格式"}
                    await websocket.send(json.dumps(error_msg))
                except Exception as e:
                    logger.error(f"处理消息时出错: {e}")
                    error_msg = {"type": "error", "message": str(e)}
                    await websocket.send(json.dumps(error_msg))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开连接: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"处理客户端消息时出错: {e}")
        finally:
            # 可选：客户端断开时清理资源
            pass

    def cleanup(self):
        """
        清理资源
        """
        if self.initialized:
            self.chatbot.cleanup()
            self.initialized = False


# 全局服务器实例
server_instance = WebSocketChatServer()


async def handle_client(websocket, path=None):
    """
    兼容原有接口的客户端处理函数
    """
    await server_instance.handle_client(websocket, path)


async def chat_main():
    """
    启动WebSocket服务器
    """
    # 关键配置：绑定自定义 IP 或域名
    server = await websockets.serve(
        handle_client,
        host="0.0.0.0",  # 绑定所有网络接口，支持外部访问
        port=8765,  # 自定义端口
        reuse_port=True  # 允许多进程复用端口
    )
    print("WebSocket 服务器已启动，监听端口 8765...")
    print("服务器地址: ws://0.0.0.0:8765")

    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        print("\n服务器被用户中断")
    finally:
        # 清理资源
        server_instance.cleanup()
        print("服务器已关闭")

if __name__ == '__main__':
    asyncio.run(chat_main())
