import time
import litellm
from litellm import completion

import logging
logging.basicConfig(level=logging.WARN)

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

log = logging.getLogger("roboai")
log.setLevel(logging.INFO)

# litellm.success_callback = ["langfuse"]


class Agent:
    def __init__(self, name, model, system_message="", base_url=None):
        self.name = name
        self.model = model
        self._last_response = None
        self._last_response_content = None
        self.messages = []
        self.system_message = system_message
        self.set_system_message(system_message)
        self.base_url = base_url

    def chat(self, message):
        self.messages.append({"content": message, "role": "user"})
        completion_args = {
            "model": self.model,
            "messages": self.messages,
        }
        if self.base_url:
            completion_args["base_url"] = self.base_url
        response = completion(**completion_args)
        self._last_response = response
        self._last_response_content = response["choices"][0]["message"]["content"]
        self.messages.append({"content": self._last_response_content, "role": "assistant"})
        
        return self._last_response_content
    
    def task_chat(self, messages):
        completion_args = {
            "model": self.model,
            "messages": messages,
        }
        if self.base_url:
            completion_args["base_url"] = self.base_url
        
        start = time.time()
        response = completion(**completion_args)
        log.debug(f"Completion time: {time.time() - start}")
        self._last_response = response
        self._last_response_content = response["choices"][0]["message"]["content"]        
        return self._last_response_content
    
    def get_last_response(self):
        return self._last_response_content
    
    def get_last_response_obj(self):
        return self._last_response
    
    def clear_messages(self):
        self.messages = []
    
    def set_system_message(self, message):
        if not message:
            log.warn(f"System message for agent '{self.name}' is empty.")
            return
        
        system_message = None
        for m in self.messages:
            if m["role"] == "system":
                system_message = m
                break
        if system_message:
            system_message["content"] = message
        else:
            self.messages.append({"content": message, "role": "system"})
        
        self.system_message = message