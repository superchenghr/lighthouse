
# # 设置日志基本配置，级别为DEBUG或INFO
import logging

from concurrent_log_handler import ConcurrentRotatingFileHandler
from langgraph.prebuilt import tools_condition, ToolNode
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
from configs.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


class ToolConfig:
    def __init__(self, tools):
        self.tools = tools
        self.tool_names = {tool.name for tool in tools}
        self.tool_routing_config = self._build_routing_config(tools)
        logger.info(f"Initialized ToolConfig with tools: {self.tool_names}, routing: {self.tool_routing_config}")

    def _build_routing_config(self, tools):
        # 创建一个空字典，用于存储工具名称到目标节点的映射
        routing_config = {}
        for tool in tools:
            tool_name = tool.name
            if "retrieve" in tool_name:
                routing_config[tool_name] = "grade_documents"
                logger.info(f"Tool '{tool_name}' routed to 'grade_documents' (retrieval tool)")
            elif "multiply" in tool_name:
                # 将其路由目标设置为 "generate"（直接生成结果）
                routing_config[tool_name] = "generate"
                # 记录调试日志，说明该工具被路由到 "generate"，并标注为非检索工具
                logger.debug(f"Tool '{tool_name}' routed to 'generate' (non-retrieval tool)")
            else:
                routing_config[tool_name] = "generate"
                logger.info(f"Tool '{tool_name}' routed to 'generate' (default tool)")
        if not routing_config:
            # 如果为空，记录警告日志，提示工具列表可能为空或未正确处理
            logger.warning("No tools provided or routing config is empty")
            # 返回生成的路由配置字典
        return routing_config

    def get_tools(self):
        return self.tools

    def get_tool_names(self):
        return self.tool_names

        # 获取工具路由配置的方法，返回动态生成的路由配置

    def get_tool_routing_config(self):
        # 直接返回 self.tool_routing_config，提供外部访问路由配置的接口
        return self.tool_routing_config


class ParallelToolNode(ToolNode):
    def __init__(self, tools, max_workers: int = 5):
        super().__init__(tools)
        self.max_workers = max_workers

    def _run_single_tool(self, tool_call: dict, tool_map: dict):
        """执行单个工具调用"""
        try:
            tool_name = tool_call["name"]
            tool = tool_map.get(tool_name)
            # 检查工具是否存在，若不存在则抛出ValueError异常
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            # 调用工具的invoke方法，传入工具参数，执行工具逻辑
            result = tool.invoke(tool_call["args"])
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        except Exception as e:
            # 记录工具执行失败的错误日志，包含工具名称和异常信息
            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
            # 返回包含错误内容的ToolMessage对象，用于状态更新
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"],
                name=tool_call.get("name", "unknown")
            )

    def __call__(self, state: dict) -> dict:
        """并行执行所有工具调用"""
        # 记录日志，表示开始处理工具调用
        logger.info("ParallelToolNode processing tool calls")

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            logger.warning("No tool calls found in last message")
            return {"message": []}

        tool_map = {tool.name: tool for tool in self.tools}
        # 初始化结果列表，用于存储所有工具调用的返回消息
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {
                executor.submit(self._run_single_tool, tool_call, tool_map): tool_call
                for tool_call in tool_calls
            }
            for future in as_completed(future_to_tool):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # 记录工具执行失败的错误日志，包含异常信息
                    logger.error(f"Tool execution failed: {e}")
                    # 获取失败任务对应的tool_call
                    tool_call = future_to_tool[future]
                    # 创建包含错误信息的ToolMessage并添加到结果列表
                    results.append(ToolMessage(
                        content=f"Unexpected error: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))
        # 执行结束
        logger.info(f"Completed {len(results)} tool calls")
        return {"messages": results}
