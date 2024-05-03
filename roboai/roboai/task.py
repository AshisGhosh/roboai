import io
import base64
import logging 

from pydantic import BaseModel
from typing import Callable

logging.basicConfig(level=logging.WARN)

log = logging.getLogger("roboai")
log.setLevel(logging.INFO)

class Tool(BaseModel):
    name: str
    func: Callable
    description: str
    example: str


def extract_code(raw_input, language="python"):
    start_delimiter = f"```{language}\n" 
    end_delimiter = "\n```"
    code_start_index = raw_input.find(start_delimiter) + len(start_delimiter)
    code_end_index = raw_input.find(end_delimiter, code_start_index)
    code = raw_input[code_start_index:code_end_index].strip()
    return code


def str_from_messages(messages):
    # Extract the text from the messages, ignore images
    text = ""
    for m in messages:
        if isinstance(m["content"], str):
            text += m["role"] + ": " + m["content"] + "\n"
        else:
            text += m["role"] + ": " + m["content"]["text"] + "\n"
    return text



class Task():
    def __init__(self, task_description, solving_agents=None, expected_output_format = None, finish_when=None):
        self.task_description = task_description
        self.solving_agents = solving_agents if solving_agents else []
        self.expected_output_format = expected_output_format
        self.finish_when = finish_when
        self.chat_messages = []
        self.tools = []
    
    def add_solving_agent(self, agent):
        self.solving_agents.append(agent)
    
    @property
    def task_description_str(self):
        if isinstance(self.task_description, list):
            return self.task_description[0]["text"]
        return self.task_description

    def add_task_image(self, image):
        try:
            # Create a bytes buffer to hold the image data
            buffer = io.BytesIO()
            # Save the Pillow image object to the buffer in a specific format (e.g., JPEG)
            image.save(buffer, format="JPEG")
            # Seek to the start of the buffer
            buffer.seek(0)
            # Read the buffer content and encode it to Base64
            image_str = base64.b64encode(buffer.read()).decode("utf-8")
            
            # Format the Base64 string as a data URL, specifying the MIME type
            # data_url = f"data:image/jpeg;base64,{image_str}"
            data_url = image_str

            # Update the task description with the text and the image data URL
            self.task_description = [
                {
                    "type": "text",
                    "text": self.task_description
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
            log.info("Task image added.")
        except Exception as e:
            log.error(f"Failed to add task image: {e}")

    
    def register_tool(self, name, func, description, example):
        self.tools.append(
            Tool(
                name=name,
                func=func,
                description=description,
                example=example

            )
        )
        log.debug(f"Tool {name} added.")
    
    def generate_tool_prompt(self):
        tool_prompt = """
            You can use the following python functions:
            """
        for tool in self.tools:
            tool_prompt += f"""'{tool.name}()'
                Description: {tool.description}
                Usage: {tool.example}
                """
        return tool_prompt
    
    def get_exec_vars(self):
        exec_vars = {}
        for tool in self.tools:
            exec_vars[tool.name] = tool.func
        
        return exec_vars
    
    def run(self):
        task_description = self.task_description
        if self.expected_output_format:
            task_description += f"""
                        Ensure your output follows the following format strictly: \n{self.expected_output_format}"""
        self.chat_messages.append(
            {
                "task":
                    {
                        "content": task_description,
                        "role": "user"
                    }
            }
        )
        log.info(f"Task:    '{self.task_description_str}'")
        for agent in self.solving_agents:
            response = self.task_chat(agent, self.chat_messages)
            log.info(f"> AGENT '{agent.name}':     {response}")
            self.chat_messages.append(
                {
                    agent.name:
                        {
                            "content": response,
                            "role": "assistant"
                        }
                }
            )
        
        return next(iter(self.chat_messages[-1].values()))["content"]
    
    def task_chat(self, agent, messages):
        agent_messages = []
        if agent.system_message:
            agent_messages.append({'role': 'system', 'content': agent.system_message})
        for m in messages:
            if next(iter(m)) == "task":
                agent_messages.append(m["task"])
            elif next(iter(m)) in [a.name for a in self.solving_agents if a != agent]:
                message = m[next(iter(m))] 
                message["role"] = "user"
                agent_messages.append(message)
            elif next(iter(m)) == agent.name:
                message = m[next(iter(m))] 
                agent_messages.append(message)
        
        log.debug(f"{str_from_messages(agent_messages)}")
        response = agent.task_chat(agent_messages)
        return response
    
    def __str__(self):
        task_info=f"Task: {self.task_description_str}"
        if self.expected_output_format:
            task_info += f"\n     Expected Output Format: {self.expected_output_format}"
        if self.solving_agents:
            task_info +="\n     Solving Agents:"
            for a in self.solving_agents:
                task_info+=f"\n      - {a.name}"
        if self.tools:
            task_info +="\n     Registered Tools:"
            for t in self.tools:
                task_info+=f"\n      - {t.name}"
        
        return task_info
