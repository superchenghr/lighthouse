import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastmcp import FastMCP
from utils.logger_util import LoggerUtil

mcp = FastMCP(name="qiyu_mcp")
logger = LoggerUtil.setup_logger("mcp_server")
@mcp.tool()
async def get_corp(code: str) -> int:
    """根据企业code查询企业id"""
    logger.info(f"调用查询企业id的工具，参数为：{code}")
    return 6007

@mcp.tool()
async def get_session(sessonId: int) -> dict:
    """通过表ysf_session根据会话id查询会话的基础信息"""
    logger.info(f"++++++++++++++++++++++++++++++++调用查询会话id的工具，参数为：{sessonId}")

    return {"id": sessonId,
            "endTime": 123154155}

@mcp.tool()
async def get_session_extend(sessonId: int) -> dict:
    """通过表ysf_session_extend根据会话id查询会话的扩展信息"""
    logger.info(f"++++++++++++++++++++++++++++++++调用查询会话extend的工具，参数为：{sessonId}")

    return {"sessionId": sessonId,
            "property": "sldfkalf"}

if __name__ == '__main__':
    mcp.run(transport="stdio")