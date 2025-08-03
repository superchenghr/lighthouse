from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_mcp_tools():
    client = MultiServerMCPClient(
        {
            "db_query": {
                "command": "python",
                "args": ["/Users/chenghaoran/PycharmProjects/lighthouse/mcpmanage/mcp_server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    return tools