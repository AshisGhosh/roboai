from litellm import completion

import logging
logging.basicConfig(level=logging.WARN)

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

log = logging.getLogger("roboai")
log.setLevel(logging.INFO)

class Agent:
    def __init__(self, name, model, system_message=""):
        self.name = name
        self.model = model
        self._last_response = None
        self._last_response_content = None
        self.messages = []
        self.system_message = system_message
        self.set_system_message(system_message)

    def chat(self, message):
        self.messages.append({"content": message, "role": "user"})
        response = completion(model=self.model, messages=self.messages)
        self._last_response = response
        self._last_response_content = response["choices"][0]["message"]["content"]
        self.messages.append({"content": self._last_response_content, "role": "assistant"})
        
        return self._last_response_content
    
    def task_chat(self, messages):
        response = completion(model=self.model, messages=messages)
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