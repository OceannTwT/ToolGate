"""
Tool Executor

Responsible for actually calling tool APIs
Supports Toolbench interface
"""

from typing import Dict, Any, Optional, Callable
import json
import os
import sys


class ToolExecutor:
    """
    Tool Executor
    
    Responsible for executing tool calls and returning results
    Supports Toolbench's get_rapidapi_response interface
    """
    
    def __init__(self, 
                 tool_executor_func: Optional[Callable] = None,
                 use_toolbench: bool = False,
                 toolbench_config: Optional[Dict[str, Any]] = None):
        """
        Initialize executor
        
        Args:
            tool_executor_func: Optional tool execution function
                If provided, will use this function to execute tool calls
            use_toolbench: Whether to use Toolbench interface
            toolbench_config: Toolbench configuration
                - tools_root: Tool root directory (default: "data.toolenv.tools")
                - schema_root: Schema root directory (default: "data/toolenv/response_examples")
                - rapidapi_key: RapidAPI key (obtained from env var TOOLBENCH_API_KEY or RAPIDAPI_KEY)
                - toolbench_key: Toolbench key (obtained from env var TOOLBENCH_API_KEY)
                - api_customization: Whether to use custom API (default: False)
                - strip_method: Result truncation method (default: "truncate")
        """
        self.tool_executor_func = tool_executor_func
        self.use_toolbench = use_toolbench
        self.toolbench_config = toolbench_config or {}
        
        # Initialize Toolbench interface (if needed)
        if use_toolbench:
            self._init_toolbench()
    
    def _init_toolbench(self):
        """Initialize Toolbench interface"""
        # Don't import here, defer until actually needed
        # This avoids triggering dependency checks during initialization
        self.toolbench_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..", "StableToolBench"
        )
        self.toolbench_path = os.path.abspath(self.toolbench_path)
        self.toolbench_available = None  # None means not checked, True/False means checked
        self.toolbench_execute = None
    
    def _ensure_toolbench_loaded(self):
        """Ensure Toolbench interface is loaded (lazy loading)"""
        if self.toolbench_available is not None:
            # Already checked
            return self.toolbench_available
        
        try:
            # Try importing Toolbench's get_rapidapi_response
            if self.toolbench_path not in sys.path:
                sys.path.insert(0, self.toolbench_path)
            
            from toolbench.inference.server import get_rapidapi_response
            self.toolbench_execute = get_rapidapi_response
            self.toolbench_available = True
            return True
        except ImportError as e:
            error_msg = str(e)
            # Check if it's a tokenizers version issue
            if "tokenizers" in error_msg.lower():
                print(f"Warning: Unable to import Toolbench interface - tokenizers version conflict")
                print(f"   Error details: {error_msg}")
                print("   Solutions:")
                print("   1. Install compatible tokenizers version: pip install 'tokenizers>=0.11.1,!=0.11.3,<0.14'")
                print("   2. Or upgrade transformers: pip install transformers -U")
                print("   3. Or install StableToolBench complete dependencies")
            else:
                print(f"Warning: Unable to import Toolbench interface: {error_msg}")
                print("   Please ensure StableToolBench is in the correct path")
            self.toolbench_available = False
            self.toolbench_execute = None
            return False
    
    def execute(self, tool_name: str, params: Dict[str, Any],
               tool_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute tool call
        
        Args:
            tool_name: Tool name
            params: Call parameters
            tool_config: Tool configuration
                If using Toolbench, should contain:
                - category: Tool category
                - api_name: API name
                - tool_input: Tool input (dict or JSON string)
        
        Returns:
            Tool return result in format: {"error": "", "response": ...}
        """
        # If custom execution function provided, use it
        if self.tool_executor_func:
            try:
                return self.tool_executor_func(tool_name, params, tool_config)
            except Exception as e:
                return {"error": str(e), "response": ""}
        
        # If using Toolbench interface
        if self.use_toolbench and self.toolbench_available:
            return self._execute_toolbench(tool_name, params, tool_config)
        
        # Otherwise use default HTTP request method
        return self._execute_http(tool_name, params, tool_config)
    
    def _execute_toolbench(self, tool_name: str, params: Dict[str, Any],
                          tool_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute tool using Toolbench interface
        
        Args:
            tool_name: Tool name
            params: Call parameters
            tool_config: Tool configuration, should contain category and api_name
        
        Returns:
            Tool return result
        """
        # Lazy load Toolbench interface
        if not self._ensure_toolbench_loaded():
            return {"error": "Toolbench interface not available", "response": ""}
        
        if not self.toolbench_execute:
            return {"error": "Toolbench interface not available", "response": ""}
        
        if not tool_config:
            return {"error": "Tool config required for Toolbench execution", "response": ""}
        
        # Get configuration
        category = tool_config.get("category")
        api_name = tool_config.get("api_name")
        
        if not category or not api_name:
            return {
                "error": "Tool config must contain 'category' and 'api_name'",
                "response": ""
            }
        
        # Prepare input
        # Prefer tool_input in tool_config, otherwise use params
        tool_input = tool_config.get("tool_input")
        if tool_input is None:
            # If tool_input not in tool_config, use params
            tool_input = params
        
        # Convert to JSON string format
        if isinstance(tool_input, dict):
            tool_input = json.dumps(tool_input, ensure_ascii=False)
        elif not isinstance(tool_input, str):
            tool_input = json.dumps(tool_input, ensure_ascii=False)
        
        # Get API keys
        rapidapi_key = (
            self.toolbench_config.get("rapidapi_key") or
            os.getenv("RAPIDAPI_KEY") or
            os.getenv("TOOLBENCH_API_KEY") or
            ""
        )
        
        toolbench_key = (
            self.toolbench_config.get("toolbench_key") or
            os.getenv("TOOLBENCH_API_KEY") or
            ""
        )
        
        # Build input dictionary
        input_dict = {
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "tool_input": tool_input,
            "strip": self.toolbench_config.get("strip_method", "truncate"),
            "rapidapi_key": rapidapi_key or toolbench_key
        }
        
        # Call Toolbench interface
        try:
            result = self.toolbench_execute(
                input_dict,
                api_customization=self.toolbench_config.get("api_customization", False),
                tools_root=self.toolbench_config.get("tools_root", "data.toolenv.tools"),
                schema_root=self.toolbench_config.get("schema_root", "data/toolenv/response_examples")
            )
            
            # Ensure return format is consistent
            if isinstance(result, dict):
                return result
            else:
                return {"error": "", "response": str(result)}
        except Exception as e:
            return {"error": f"Toolbench execution error: {str(e)}", "response": ""}
    
    def _execute_http(self, tool_name: str, params: Dict[str, Any],
                     tool_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute tool using HTTP request (default implementation)
        
        This is a simplified implementation, should call corresponding API based on tool config in practice
        """
        if not tool_config:
            return {"error": "No tool configuration provided", "response": ""}
        
        # Should build actual API request based on tool_config here
        # Simplified: return an example result
        return {
            "error": "",
            "response": {
                "status": "success",
                "data": params,
                "message": f"Tool {tool_name} executed with params: {params}"
            }
        }

