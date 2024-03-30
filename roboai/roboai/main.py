from pydantic import BaseModel
from typing import Callable

from roboai.agent import Agent
from roboai.task import Task

import logging

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


class RobotJob:
    def __init__(self):
        pass

    def run(self):
        '''
        Job to:
            1. Understand the scene
            2. Create a plan to clear the table
        '''        
        
        task = Task("Return python code to identify the objects on the table using only the provided functions.")
        task.register_tool(
            name='get_objects_on_table',
            func=get_objects_on_table,
            description='Returns a list of the objects on the table.',
            example='"objects_on_table = get_objects_on_table()" --> Returns: ["Object 1", "Object 2", "Object 3"]'
        )
        analyzer_agent = Agent(
            name="Analyzer",
            model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
            system_message="""
            You are a helpful agent that concisely responds with just code.
            Use only the provided functions.
            """ + task.generate_tool_prompt()
        )
        task.add_solving_agent(analyzer_agent)
        
        log.info(task)
        output = task.run()

        code = extract_code(output)
        exec_vars = task.get_exec_vars()
        exec(code, exec_vars)
        log.info(exec_vars.get("objects_on_table", None))

        plan_task = Task(
            f"""Create a plan for a robot to remove the following objects from the table:
                {exec_vars.get("objects_on_table", None)}
                Do not add any extra steps.
            """,
            expected_output_format="""
                1. pick object1
                2. place object1
                3. pick object2
                4. place object2
                5. pick object3
                6. place object3
                ...
            """
        )
        plan_task.register_tool(
            name="pick",
            func=pick,
            description="Robot picks up the provided arg 'object'",
            example='"pick_success = pick(object="Cheese")" --> Returns: True '
        )

        plan_task.register_tool(
            name="place",
            func=place,
            description="Robot places the provided arg 'object'",
            example='"place_success = place(object="Cheese")" --> Returns: True '
        )

        planner_agent = Agent(
            name="Planner",
            model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
            system_message=f"""
            You are a planner that breaks down tasks into steps for robots.
            Create a conscise set of steps that a robot can do.
            Do not add any extra steps.
            """ + plan_task.generate_tool_prompt()
        )

        plan_task.add_solving_agent(planner_agent)
        log.info(plan_task)
        plan_task.run()

def get_objects_on_table():
    return ["Cheese", "Beer", "Toy"]

def pick(object):
    log.debug(f"Picked: {object} ")
    return True

def place(object):
    log.debug(f"Placed: {object}")
    return True


if __name__ == "__main__":
    job = RobotJob()
    job.run()
