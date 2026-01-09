"""
GPT Client

GPT client implementation based on OpenAI API
"""

import json
import time
from typing import Dict, Any, List, Optional

try:
    from tenacity import retry, wait_random_exponential, stop_after_attempt
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

try:
    import openai
except ImportError:
    openai = None

DEFAULT_SYS_PROMPT = "You are a helpful agent."


def _chat_completion_request_impl(key, messages, model="gpt-4o-mini", functions=None, 
                                 function_call=None, stop=None, **args):
    """
    Call OpenAI ChatCompletion API (internal implementation)
    
    Args:
        key: OpenAI API key
        messages: Message list
        model: Model name
        functions: Function definition list
        function_call: Function call mode
        stop: Stop token list
        **args: Other parameters
    
    Returns:
        API response
    """
    if openai is None:
        raise ImportError("openai package is required. Install it with: pip install openai")
    
    json_data = {
        "model": model,
        "messages": messages,
        "max_tokens": 16384,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    
    if stop is not None:
        json_data.update({"stop": stop})
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    
    try:
        if model == "gpt-3.5-turbo" or "gpt-4o-mini" in model:
            openai.api_key = key
        else:
            raise NotImplementedError(f"Model {model} not supported")
        
        openai_response = openai.ChatCompletion.create(**json_data)
        json_data = json.loads(str(openai_response))
        return json_data
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"OpenAI calling Exception: {e}")
        return e


# If tenacity is available, use retry decorator; otherwise use function directly
if HAS_TENACITY:
    chat_completion_request = retry(
        wait=wait_random_exponential(min=1, max=40), 
        stop=stop_after_attempt(3)
    )(_chat_completion_request_impl)
else:
    chat_completion_request = _chat_completion_request_impl


class GPTFactory:
    """
    GPT Factory Class
    
    Manages conversation history and API calls
    """
    
    def __init__(self, model="gpt-4o-mini", openai_key=""):
        """
        Initialize GPT factory
        
        Args:
            model: Model name
            openai_key: OpenAI API key
        """
        self.model = model
        self.conversation_history = [
            {"role": "system", "content": DEFAULT_SYS_PROMPT}
        ]
        self.openai_key = openai_key
        self.TRY_TIME = 6
        self.time = time.time()
    
    def add_conv(self, conv):
        """Add conversation"""
        self.conversation_history.append(conv)
    
    def set_key(self, openai_key):
        """Set API key"""
        self.openai_key = openai_key
    
    def set_sys_conv(self, sys_prompt):
        """Set system prompt"""
        self.conversation_history = [
            {"role": "system", "content": sys_prompt}
        ]
    
    def add_user_conv(self, user_prompt):
        """Add user message"""
        message = {"role": "user", "content": user_prompt}
        self.conversation_history.append(message)
    
    def set_default_conv(self):
        """Reset to default conversation"""
        self.conversation_history = [
            {"role": "system", "content": ""}
        ]
    
    def change_conv(self, conv):
        """Change entire conversation history"""
        self.conversation_history = conv
    
    def clear_conv(self):
        """Clear conversation history"""
        self.conversation_history = [
            {"role": "system", "content": DEFAULT_SYS_PROMPT}
        ]
    
    def predict(self, **gpt_args):
        """
        Predict (generate text)
        
        Args:
            **gpt_args: GPT parameters
        
        Returns:
            Generated text
        """
        self.time = time.time()
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            json_data = chat_completion_request(
                self.openai_key, 
                self.conversation_history, 
                self.model, 
                **gpt_args
            )
            try:
                content = json_data["choices"][0]["message"]["content"]
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": content
                })
                return content
            except BaseException as e:
                print(f"Parsing Exception: {repr(e)}. Try again.")
                if json_data is not None:
                    print(f"OpenAI return: {json_data}")
        return None
    
    def predict_fun(self, **gpt_args):
        """
        Predict (function call)
        
        Args:
            **gpt_args: GPT parameters
        
        Returns:
            Function call information
        """
        self.time = time.time()
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            json_data = chat_completion_request(
                self.openai_key, 
                self.conversation_history, 
                **gpt_args
            )
            try:
                return json_data["choices"][0]["message"]["function_call"]
            except BaseException as e:
                print(f"Parsing Exception: {repr(e)}. Try again.")
                if json_data is not None:
                    print(f"OpenAI return: {json_data}")
        return None


class GPTClient:
    """
    GPT Client Wrapper
    
    Provides higher-level interface for ToolGate
    """
    
    def __init__(self, model="gpt-4o-mini", openai_key="", system_prompt=None):
        """
        Initialize client
        
        Args:
            model: Model name
            openai_key: OpenAI API key
            system_prompt: System prompt (optional)
        """
        self.factory = GPTFactory(model=model, openai_key=openai_key)
        if system_prompt:
            self.factory.set_sys_conv(system_prompt)
    
    def generate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Generate text
        
        Args:
            prompt: Input prompt
            stop: Stop token list
            **kwargs: Other parameters
        
        Returns:
            Generated text
        """
        self.factory.add_user_conv(prompt)
        return self.factory.predict(stop=stop, **kwargs)
    
    def generate_with_history(self, messages: List[Dict[str, str]], 
                            stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Generate based on message history
        
        Args:
            messages: Message list
            stop: Stop token list
            **kwargs: Other parameters
        
        Returns:
            Generated text
        """
        self.factory.change_conv(messages)
        return self.factory.predict(stop=stop, **kwargs)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.factory.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.factory.clear_conv()

