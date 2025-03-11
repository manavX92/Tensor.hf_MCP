#!/usr/bin/env python3
"""
Test script for the HuggingFace MCP server.
This script tests basic functionality of the server.
"""

import asyncio
import sys
from mcp.client import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """Test the HuggingFace MCP server by connecting to it and calling some methods."""
    print("Testing HuggingFace MCP server...")
    
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=["hf_mcp_server.py"],
        env=None
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                print("✅ Connection initialized")
                
                # List available tools
                tools = await session.list_tools()
                print(f"✅ Found {len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                # Test search_models tool
                print("\nTesting search_models tool...")
                result = await session.call_tool("search_models", {"query": "gpt2", "limit": 3})
                print(f"✅ Search results: {result[:100]}...")
                
                # Test get_recommended_models tool
                print("\nTesting get_recommended_models tool...")
                result = await session.call_tool("get_recommended_models", {"task": "text-generation"})
                print(f"✅ Recommended models: {result[:100]}...")
                
                # List available resources
                resources = await session.list_resources()
                print(f"\n✅ Found {len(resources)} resource patterns:")
                for resource in resources:
                    print(f"  - {resource.pattern}: {resource.description}")
                
                # Test reading a resource
                print("\nTesting resource reading...")
                content, mime_type = await session.read_resource("hf://models/text-generation")
                print(f"✅ Resource content: {content[:100]}...")
                
                print("\nAll tests passed!")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)