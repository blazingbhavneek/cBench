import asyncio
import logging
import json
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MCPAgent(ABC):
    """Base class for MCP (Model Context Protocol) client functionality"""

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from MCP servers.

        Returns:
            List of tool definitions
        """

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """

    @abstractmethod
    async def get_system_prompt(self) -> str:
        """
        Get system prompt describing available tools.

        Returns:
            System prompt text
        """


class NoOpMCPAgent(MCPAgent):
    """Empty implementation - does nothing"""

    async def list_tools(self) -> List[Dict[str, Any]]:
        return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": False, "error": "MCP not enabled"}

    async def get_system_prompt(self) -> str:
        return ""


class MCPClientAgent(MCPAgent):
    """
    MCP client that connects to MCP servers and exposes their tools.
    
    Uses the mcp library to connect to MCP servers via stdio or SSE.
    """

    def __init__(
        self,
        servers: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 30,
    ):
        """
        Initialize MCP client.

        Args:
            servers: List of server configurations. Each config should have:
                - name: Server name
                - command: Command to run (for stdio transport)
                - args: Command arguments (for stdio transport)
                - url: Server URL (for SSE transport)
                - transport: "stdio" or "sse" (default: "stdio")
            timeout: Timeout for tool calls in seconds
        """
        self.servers = servers or []
        self.timeout = timeout
        self._clients = {}
        self._tools = {}
        self._initialized = False

    async def _ensure_initialized(self):
        """Initialize MCP connections to all servers."""
        if self._initialized:
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.sse import sse_client
        except ImportError:
            logger.error(
                "MCP library not installed. Install with: pip install mcp"
            )
            self._initialized = True
            return

        for server_config in self.servers:
            name = server_config.get("name", "unknown")
            transport = server_config.get("transport", "stdio")

            try:
                if transport == "stdio":
                    # Stdio transport
                    command = server_config.get("command")
                    args = server_config.get("args", [])
                    
                    if not command:
                        logger.warning(f"MCP server {name}: missing 'command' for stdio transport")
                        continue

                    server_params = StdioServerParameters(
                        command=command,
                        args=args,
                    )
                    
                    stdio_transport = await stdio_client(server_params).__aenter__()
                    read, write = stdio_transport
                    
                    session = ClientSession(read, write)
                    await session.initialize()
                    
                elif transport == "sse":
                    # SSE transport
                    url = server_config.get("url")
                    
                    if not url:
                        logger.warning(f"MCP server {name}: missing 'url' for SSE transport")
                        continue
                    
                    sse_transport = await sse_client(url).__aenter__()
                    read, write = sse_transport
                    
                    session = ClientSession(read, write)
                    await session.initialize()
                    
                else:
                    logger.warning(f"MCP server {name}: unknown transport '{transport}'")
                    continue

                self._clients[name] = session
                
                # List available tools
                tools_response = await session.list_tools()
                tools = tools_response.tools if hasattr(tools_response, 'tools') else []
                
                for tool in tools:
                    tool_name = f"{name}_{tool.name}"
                    self._tools[tool_name] = {
                        "server": name,
                        "tool": tool,
                        "session": session,
                    }
                    logger.info(f"Registered tool: {tool_name}")

            except Exception as e:
                logger.error(f"Failed to initialize MCP server {name}: {e}")

        self._initialized = True

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from connected MCP servers."""
        await self._ensure_initialized()
        
        tool_list = []
        for tool_name, tool_info in self._tools.items():
            tool = tool_info["tool"]
            tool_list.append({
                "name": tool_name,
                "description": tool.description if hasattr(tool, 'description') else "",
                "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                "server": tool_info["server"],
            })
        
        return tool_list

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool by name.

        Args:
            name: Tool name (optionally prefixed with server name)
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        await self._ensure_initialized()
        
        if name not in self._tools:
            # Try to find by suffix match
            for tool_name in self._tools:
                if tool_name.endswith(f"_{name}"):
                    name = tool_name
                    break
        
        if name not in self._tools:
            return {
                "success": False,
                "error": f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}",
            }

        tool_info = self._tools[name]
        session = tool_info["session"]
        tool = tool_info["tool"]
        actual_tool_name = tool.name if hasattr(tool, 'name') else name

        try:
            result = await asyncio.wait_for(
                session.call_tool(actual_tool_name, arguments),
                timeout=self.timeout,
            )
            
            # Format result
            content = []
            if hasattr(result, 'content'):
                for item in result.content:
                    if hasattr(item, 'text'):
                        content.append(item.text)
                    elif hasattr(item, 'data'):
                        content.append(str(item.data))
            
            return {
                "success": True,
                "content": "\n".join(content) if content else str(result),
                "raw": result,
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Tool call timed out: {name}")
            return {
                "success": False,
                "error": f"Tool call timed out after {self.timeout} seconds",
            }
        except Exception as e:
            logger.error(f"Tool call failed: {name}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def get_system_prompt(self) -> str:
        """Get system prompt describing available MCP tools."""
        tools = await self.list_tools()
        
        if not tools:
            return ""

        prompt_parts = [
            "You have access to the following tools via MCP (Model Context Protocol):",
            "",
        ]

        for tool in tools:
            prompt_parts.append(f"### {tool['name']}")
            prompt_parts.append(f"Description: {tool['description']}")
            
            if tool.get('inputSchema'):
                schema = tool['inputSchema']
                if 'properties' in schema:
                    prompt_parts.append("Parameters:")
                    for param_name, param_info in schema.get('properties', {}).items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        prompt_parts.append(f"  - {param_name} ({param_type}): {param_desc}")
            
            prompt_parts.append("")

        prompt_parts.append(
            "To use a tool, respond with a JSON object containing 'use_tool': true, "
            "'tool_name': the tool name, and 'arguments': the tool arguments."
        )

        return "\n".join(prompt_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about MCP connections."""
        return {
            "initialized": self._initialized,
            "servers_connected": list(self._clients.keys()),
            "tools_available": list(self._tools.keys()),
        }


def create_mcp_agent_from_config(config_path: str) -> MCPAgent:
    """
    Create an MCP agent from a JSON configuration file.

    Config file format:
    {
        "servers": [
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"]
            },
            {
                "name": "database",
                "transport": "sse",
                "url": "http://localhost:8080/sse"
            }
        ],
        "timeout": 30
    }

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configured MCPAgent instance
    """
    import json
    from pathlib import Path

    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"MCP config file not found: {config_path}")
        return NoOpMCPAgent()

    try:
        with open(config_file) as f:
            config = json.load(f)
        
        return MCPClientAgent(
            servers=config.get("servers", []),
            timeout=config.get("timeout", 30),
        )
    except Exception as e:
        logger.error(f"Failed to load MCP config: {e}")
        return NoOpMCPAgent()
