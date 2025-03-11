#!/usr/bin/env python3
"""
Script to run the HuggingFace MCP server in development mode.
"""

import subprocess
import os
import sys

def run_dev_server():
    """Run the HuggingFace MCP server in development mode."""
    print("Starting HuggingFace MCP server in development mode...")
    
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
    
    # Run the server in development mode
    try:
        # This will block until the user exits
        subprocess.run(
            ["mcp", "dev", "hf_mcp_server.py"],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running MCP server: {e}")
        return False
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return True

if __name__ == "__main__":
    success = run_dev_server()
    sys.exit(0 if success else 1)