from pydantic import BaseModel
from typing import Callable
import base64  # noqa: F401
from PIL import Image  # noqa: F401
from abc import ABC, abstractmethod

from roboai.agent import Agent
from roboai.task import Task

# from shared.utils.robosim_client import (  # noqa: F401
#     get_objects_on_table,
#     pick,
#     place,
#     get_image,
#     get_grasp_image,
# ) 

from shared.utils.isaacsim_client import (
    get_image, 
    pick,
    place
)

from shared.utils.model_server_client import answer_question_from_image  # noqa: F401
import shared.utils.gradio_client as gradio  # noqa: F401
import shared.utils.replicate_client as replicate  # noqa: F401
from shared.utils.llm_utils import get_closest_text_sync as get_closest_text 

import gradio as gr

import logging

log = logging.getLogger("roboai")
log.setLevel(logging.DEBUG)


class Tool(BaseModel):
    name: str
    func: Callable
    description: str
    example: str


def extract_code(raw_input, language="python"):
    start_delimiter = f"```{language}"
    if start_delimiter not in raw_input:
        start_delimiter = "```"

    code_start_index = raw_input.find(start_delimiter)
    if code_start_index == -1:
        code_start_index = 0
    else:
        code_start_index += len(start_delimiter)

    end_delimiter = "```"
    code_end_index = raw_input.find(end_delimiter, code_start_index)
    if code_end_index == -1:
        code_end_index = len(raw_input)

    code = raw_input[code_start_index:code_end_index].strip()
    log.debug(f"Extracted code: \n{code}")
    return code

class RobotJob(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass
class ClearTableJob(RobotJob):
    def __init__(self):
        pass

    def run(self, chat_history=None):
        """
        Job to:
            1. Understand the scene
            2. Create a plan to clear the table
        """
        if chat_history:
            if not chat_history[-1][1]:
                chat_history[-1][1] = ""
            else:
                chat_history[-1][1] += "\n"
            chat_history[-1][1] += "Getting image...\n"
            yield chat_history

        im = get_image()
        prompt = "What objects are on the table?"

        if chat_history:
            chat_history[-1][1] += "Asking VLA model...\n"
            yield chat_history
        
        output = gradio.moondream_answer_question_from_image(im, prompt)["result"]
        if chat_history:
            chat_history[-1][1] += f"Response:\n{output}\n"
            yield chat_history

        
        if chat_history:
            chat_history[-1][1] += "Creating plan...\n"
            yield chat_history

        task = Task(
            f"Given the following summary, return just a list in python of the objects on the table. The table is not an object. Summary: \n{output}",
            expected_output_format="""
                objects_on_table = ["Object 1", "Object 2", "Object 3"]
            """,
        )
        analyzer_agent = Agent(
            name="Analyzer",
            model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
            system_message="""
            You are a helpful agent that concisely responds with only code.
            Use only the provided functions, do not add any extra code.
            """,
        )
        task.add_solving_agent(analyzer_agent)
        output = task.run()
        # output = '```objects_on_table = ["Box of Cereal", "Carton of Milk", "Can of Soup"]```'
        code = extract_code(output)
        try:
            exec_vars = {}
            exec(code, exec_vars)
            log.info(exec_vars.get("objects_on_table", None))
            list_of_objects = exec_vars.get("objects_on_table", None)
        except Exception as e:
            log.error(f"Error executing code: {e}")
            list_of_objects = None
            if chat_history:
                chat_history[-1][1] += f"Error executing code: {e}"
                yield chat_history
            return

        plan_task = Task(
            f"""Create a plan for a robot to remove the following objects from the table:
                {list_of_objects}
                Do not add any extra steps.
            """,
            # expected_output_format="""
            #     1. pick object1
            #     2. place object1
            #     3. pick object2
            #     4. place object2
            #     5. pick object3
            #     6. place object3
            # """
            expected_output_format="A numbered list of steps constrained to the provided functions.",
        )
        plan_task.register_tool(
            name="pick",
            func=pick,
            description="Robot picks up the provided arg 'object_name'",
            example='"pick_success = pick(object_name="Object 1")" --> Returns: True ',
        )

        plan_task.register_tool(
            name="place",
            func=place,
            description="Robot places the provided arg 'object_name'",
            example='"place_success = place(object_name="Object 1")" --> Returns: True ',
        )

        planner_agent = Agent(
            name="Planner",
            model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
            system_message="""
            You are a planner that breaks down tasks into steps for robots.
            Create a conscise set of steps that a robot can do.
            Do not add any extra steps.
            """
            + plan_task.generate_tool_prompt(),
        )

        plan_task.add_solving_agent(planner_agent)
        # log.info(plan_task)
        output = plan_task.run()
        log.info(output)

        if chat_history:
            chat_history[-1][1] += f"Response:\n{output}"
            yield chat_history

        if chat_history:
            chat_history[-1][1] += "Converting plan to code...\n"
            yield chat_history

        plan_generated = True
        code = extract_code(output)
        exec_vars = plan_task.get_exec_vars()
        try:
            exec(code, exec_vars)
        except Exception as e:
            log.error(f"Error executing plan: {e}")
            plan_generated = False

        # Validate the plan?

        # Execute the plan
        if not plan_generated:
            coder_task = Task(
                f"""Return python code to execute the plan using only the provided functions.
                    {output}
                """
            )
            coder_task.register_tool(
                name="pick",
                func=pick,
                description="Robot picks up the provided arg 'object_name'",
                example='"pick_success = pick(object_name="Object 1")" --> Returns: True ',
            )
            coder_task.register_tool(
                name="place",
                func=place,
                description="Robot places the provided arg 'object_name'",
                example='"place_success = place(object_name="Object 1")" --> Returns: True ',
            )
            coder_agent = Agent(
                name="Coder",
                # model="ollama/gemma:7b",
                model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
                system_message="""
                You are a coder that writes concise and exact code to execute the plan.
                Use only the provided functions.
                """
                + coder_task.generate_tool_prompt(),
            )
            coder_task.add_solving_agent(coder_agent)
            log.info(coder_task)
            output = coder_task.run()

            if chat_history:
                chat_history[-1][1] += f"Response:\n{output}\n"
                yield chat_history

            if chat_history:
                chat_history[-1][1] += "Extracting and running code...\n"
                yield chat_history

            code = extract_code(output)

            if chat_history:
                chat_history[-1][1] += f"Response:\n```{code}```"
                yield chat_history

            try:
                exec_vars = coder_task.get_exec_vars()
                exec(code, exec_vars)
                result = "Successful execution of plan."
            except Exception as e:
                log.error(f"Error executing code: {e}")
                result = "Error executing plan."
            finally:
                if chat_history:
                    chat_history[-1][1] += f"\nResponse:\n**{result}**"
                    yield chat_history

class WhatIsOnTableJob(RobotJob):
    image = None
    def __init__(self):
        self.image = get_image()

    def get_image(self):
        if not self.image:
            self.image = get_image()
        return self.image

    def run(self, chat_history=None):
        if chat_history:
            if not chat_history[-1][1]:
                chat_history[-1][1] = ""
            else:
                chat_history[-1][1] += "\n"
                yield chat_history
            chat_history[-1][1] += "Getting image...\n"
            yield chat_history

        im = get_image()
        prompt = "What objects are on the table?"
        
        if chat_history:
            chat_history[-1][1] += "Asking VLA model...\n"
            yield chat_history

        output = gradio.moondream_answer_question_from_image(im, prompt)
        if chat_history:
            chat_history[-1][1] += f"Response:\n{output['result']}"
            yield chat_history
        return output["result"]

class TestJob(RobotJob):
    def __init__(self):
        pass

    def run(self, chat_history=None):
        responses = [
            "I am a robot.",
            "I can help you with tasks.",
            "Ask me to do something"
        ]
        if chat_history:
            if not chat_history[-1][1]:
                chat_history[-1][1] = ""
            else:
                chat_history[-1][1] += "\n"
                yield chat_history
            for response in responses:
                chat_history[-1][1] += response
                yield chat_history


def chat():
    with gr.Blocks() as demo:
        gr.Markdown("## RoboAI Chatbot")
        chatbot = gr.Chatbot(height=700)
        msg = gr.Textbox(placeholder="Ask me to do a task.", container=False, scale=7)
        image_output = gr.Image(label="Response Image")
        clear = gr.ClearButton([msg, chatbot])
        current_task = [None]

        def respond(message, chat_history):
            nonlocal current_task
            closest_text = get_closest_text(message, ["Clear the table", "What is on the table?"])
            response = None
            image = None
            if closest_text:
                print(f"Closest text: {closest_text}")
                current_task[0] = closest_text
                response = f"Doing task: {closest_text}"
                # image = WhatIsOnTableJob().get_image()
            chat_history.append((message, None))
            return "", chat_history, image
        
        def do_function(chat_history):
            nonlocal current_task
            if not current_task:
                return "", chat_history, None

            chat_history[-1][1] = f"**{current_task[0]}**"
            yield chat_history
            
            if current_task[0] == "What is on the table?":
                job = WhatIsOnTableJob()
                image = job.get_image()
                yield from job.run(chat_history)
            elif current_task[0] == "Clear the table":
                job = ClearTableJob()
                yield from job.run(chat_history)
                image = WhatIsOnTableJob().get_image()
            elif current_task[0] == "Test Job":
                job = TestJob()
                yield from job.run(chat_history)
            else:
                chat_history[-1][1] = "Sorry, I don't understand that command."
                image = None

            return None, chat_history, image            
        
        def get_image_output():
            image_output = WhatIsOnTableJob().get_image()
            return image_output

        msg.submit(respond, [msg, chatbot], [msg, chatbot, image_output], queue=False).then(
            get_image_output, [], [image_output]
        ).then(
            do_function, chatbot, chatbot
        )

    
    demo.queue()
    demo.launch()

if __name__ == "__main__":
    chat()
