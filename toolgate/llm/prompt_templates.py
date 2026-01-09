"""
Prompt Templates

Defines prompt templates for ToolGate
"""

from typing import Dict, Any, List, Optional
from ..core.symbolic_state import SymbolicState


class PromptTemplate:
    """Base Prompt Template"""
    
    def __init__(self, system_prompt: str = ""):
        """
        Initialize template
        
        Args:
            system_prompt: System prompt
        """
        self.system_prompt = system_prompt
    
    def format(self, query: str, state: SymbolicState, 
              history: Optional[List[Dict[str, Any]]] = None,
              tool_results: Optional[List[str]] = None) -> str:
        """
        Format prompt
        
        Args:
            query: User query
            state: Current state
            history: Conversation history
            tool_results: Tool call result list
        
        Returns:
            Formatted prompt
        """
        raise NotImplementedError


class ReActPromptTemplate(PromptTemplate):
    """
    ReAct-style Prompt Template
    
    Supports special markers:
    - <start_call_tool>: Start tool calling phase
    - <end_call_tool>: End tool calling phase
    - <start_tool_result>: Tool result start
    - <end_tool_result>: Tool result end
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer user questions.

You have access to a set of tools and a symbolic state that tracks verified facts.

**Important Control Tokens:**
- When you need to use a tool, output: `<start_call_tool>`
- After tool results are provided, they will be wrapped in: `<start_tool_result>...</end_tool_result>`
- When you finish using tools, output: `<end_call_tool>`

**State Information:**
The current symbolic state contains verified facts. Use this information to:
1. Check if you have enough information to answer directly
2. Determine what information is missing and needs to be retrieved via tools
3. Understand what tools can be called based on the current state

**Tool Calling Process:**
1. Think about what information you need
2. Output `<start_call_tool>` followed by a brief description of what you need
3. Wait for tool results
4. Continue reasoning with the new information
5. Repeat if needed, or provide the final answer

**CRITICAL: When Tools Fail - Keep Trying!**
- If a tool call fails, DO NOT give up immediately. Consider:
  * Try a different tool that might provide similar information
  * Try the same tool with different parameters
  * Try alternative approaches or search strategies
  * Think about what other information sources might help
- Only provide a final answer when you are CONFIDENT you have:
  * Successfully retrieved the necessary information, OR
  * Exhausted all reasonable tool options and can provide a helpful answer based on available information
- Do NOT end reasoning prematurely just because one tool failed
- Be persistent and creative in finding alternative solutions

**Output Format:**
- If you can answer directly: Provide the answer without `<start_call_tool>`
- If you need tools: Output `<start_call_tool>` followed by your reasoning about what tool to use
- If tools fail: Think about alternatives and try again with `<start_call_tool>`
"""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize ReAct template
        
        Args:
            system_prompt: Custom system prompt (optional)
        """
        super().__init__(
            system_prompt or self.DEFAULT_SYSTEM_PROMPT
        )
    
    def format(self, query: str, state: SymbolicState,
              history: Optional[List[Dict[str, Any]]] = None,
              tool_results: Optional[List[str]] = None) -> str:
        """
        Format ReAct prompt
        
        Args:
            query: User query
            state: Current state
            history: Conversation history
            tool_results: Tool call result list (wrapped in markers)
        
        Returns:
            Formatted prompt
        """
        parts = []
        
        # Add state information
        state_summary = state.get_summary()
        if state_summary and state_summary != "Current State: Empty":
            parts.append(f"**Current State:**\n{state_summary}\n")
        
        # Add tool results (if exist)
        if tool_results:
            parts.append("**Tool Results:**\n")
            for result in tool_results:
                parts.append(result)
            parts.append("")
        
        # Add user query
        parts.append(f"**User Query:** {query}\n")
        
        # Add instructions
        parts.append("**Your Task:**")
        parts.append("Think step by step. If you need to use tools, output `<start_call_tool>` followed by a description of what you need.")
        parts.append("If a tool fails, think about alternative approaches and try other tools before giving up.")
        parts.append("Only provide the final answer when you are CONFIDENT you have enough information or have exhausted all reasonable options.\n")
        
        return "\n".join(parts)
    
    def format_messages(self, query: str, state: SymbolicState,
                       history: Optional[List[Dict[str, Any]]] = None,
                       tool_results: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Format as message list (for OpenAI API)
        
        Args:
            query: User query
            state: Current state
            history: Conversation history
            tool_results: Tool call result list
        
        Returns:
            Message list
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        if history:
            for msg in history:
                messages.append(msg)
        
        # Add current query and context
        user_content = self.format(query, state, history, tool_results)
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    @staticmethod
    def extract_action(text: str) -> Dict[str, Any]:
        """
        Extract action from LLM output
        
        Args:
            text: LLM generated text
        
        Returns:
            Action dictionary containing:
            - type: "answer" or "start_call_tool"
            - content: Content
            - reasoning: Reasoning process (if any)
        """
        text = text.strip()
        
        # Check if contains <start_call_tool>
        if "<start_call_tool>" in text:
            # Extract tool calling part
            start_idx = text.find("<start_call_tool>")
            end_idx = text.find("<end_call_tool>")
            
            if end_idx == -1:
                # No end marker, extract content after
                content = text[start_idx + len("<start_call_tool>"):].strip()
            else:
                content = text[start_idx + len("<start_call_tool>"):end_idx].strip()
            
            # Extract reasoning process (part before marker)
            reasoning = text[:start_idx].strip() if start_idx > 0 else ""
            
            return {
                "type": "start_call_tool",
                "content": content,
                "reasoning": reasoning,
                "full_text": text
            }
        else:
            # Direct answer
            return {
                "type": "answer",
                "content": text,
                "reasoning": "",
                "full_text": text
            }
    
    @staticmethod
    def wrap_tool_result(tool_name: str, result: Dict[str, Any], 
                        summary: Optional[str] = None) -> str:
        """
        Wrap tool result with special markers
        
        Args:
            tool_name: Tool name
            result: Tool return result
            summary: Result summary (optional, use summary if provided)
        
        Returns:
            Wrapped result string
        """
        if summary:
            content = summary
        else:
            # Simplify result representation
            import json
            try:
                content = json.dumps(result, ensure_ascii=False, indent=2)
                # If too long, truncate
                if len(content) > 500:
                    content = content[:500] + "\n... (truncated)"
            except:
                content = str(result)
        
        return f"<start_tool_result>\nTool: {tool_name}\nResult:\n{content}\n<end_tool_result>"

