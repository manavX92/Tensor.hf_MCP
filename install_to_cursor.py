#!/usr/bin/env python3
"""
Script to install the HuggingFace MCP server in Cursor.
"""

import subprocess
import os
import sys

def install_to_cursor():
    """Install the HuggingFace MCP server in Cursor."""
    print("Installing HuggingFace MCP server to Cursor...")
    
    # Check if mcp CLI is available
    try:
        subprocess.run(["mcp", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: MCP CLI not found. Please install it with 'pip install mcp[cli]'")
        return False
    
    # Check if the server file exists
    if not os.path.exists("hf_mcp_server.py"):
        print("Error: hf_mcp_server.py not found")
        return False
    
    # Check if the token file exists
    if not os.path.exists("docs/Hf_token"):
        print("Error: HuggingFace token file not found at docs/Hf_token")
        return False
    
    # Install the server
    try:
        subprocess.run(
            ["mcp", "install", "hf_mcp_server.py", "--name", "HuggingFace Models"],
            check=True
        )
        print("✅ HuggingFace MCP server installed successfully!")
        print("You can now use it in Cursor.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing MCP server: {e}")
        return False

if __name__ == "__main__":
    success = install_to_cursor()
    sys.exit(0 if success else 1)