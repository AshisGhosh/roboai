from typing import List, Optional, Tuple
from PIL import Image  # noqa: F401

from burr.core import Application, ApplicationBuilder, State, default, when
from burr.core.action import action
from burr.lifecycle import LifecycleAdapter
from burr.tracking import LocalTrackingClient

from shared.utils.llm_utils import get_closest_text_sync as get_closest_text
# from shared.utils.isaacsim_client import get_image as get_image_from_sim, pick, place  # noqa: F401
from shared.utils.omnigibson_client import get_image as get_image_from_sim, pick, place, navigate_to # noqa: F401
from shared.utils.image_utils import pil_to_b64, b64_to_pil
from shared.utils.gradio_client import moondream_answer_question_from_image as moondream

from task import Task
from agent import Agent
from plans import PLANS
from skills import SKILLS
from semantic_locations import SEMANTIC_LOCATIONS

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


MODES = {
    "answer_question": "text",
    "generate_image": "image",
    "generate_code": "code",
    "unknown": "text",
}

DEFAULT_MODEL = "openrouter/meta-llama/llama-3-8b-instruct:free"
# DEFAULT_MODEL = "openrouter/huggingfaceh4/zephyr-7b-beta:free"
# DEFAULT_MODEL = "ollama/llama3:latest"
# DEFAULT_MODEL = "ollama/phi3"
# CODING_MODEL = "ollama/codegemma:instruct"
CODING_MODEL = DEFAULT_MODEL


def pick_mock(object_name: str):
    name = pick_mock.__name__
    print(f"Called {name}  TEST MODE ENABLED")
    return True


def place_mock(location: str):
    name = place_mock.__name__
    print(f"Called {name}  TEST MODE ENABLED")
    return True


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


@action(reads=[], writes=["chat_history", "prompt"])
def process_prompt(state: State, prompt: str) -> Tuple[dict, State]:
    result = {"chat_item": {"role": "user", "content": prompt, "type": "text"}}
    return result, state.append(chat_history=result["chat_item"]).update(prompt=prompt)


@action(reads=["prompt"], writes=["task"])
def determine_task(state: State) -> Tuple[dict, State]:
    closest_text = get_closest_text(
        state["prompt"], ["What is on the table?", "Clear the table"],
        threshold=0.75
    )
    if closest_text:
        result = {"task": closest_text}
    else:
        result = {"task": "unknown"}

    return result, state.update(**result)


@action(reads=["task"], writes=["response", "current_state"])
def response_for_unknown_task(state: State) -> Tuple[dict, State]:
    result = {
        "response": {
            "content": f"Unknown task: **{state['prompt']}**\nParsing task...",
            "type": "text",
            "role": "assistant",
        },
        "current_state": "PARSING",
    }
    return result, state.update(**result)

@action(reads=["task"], writes=["task", "feasible"])
def parse_unknown_task(state: State) -> Tuple[dict, State]:
    task = Task(
        f"Given the following prompt, return a simplified high level plan for a robot to perform. Prompt: \n{state['prompt']}" +
            "\n\Do not include steps related to confirming successful execution or getting feedback. Do not include steps related to repeating steps."
            f"\n\nExamples: {PLANS.keys()} ",
        expected_output_format="A numbered list of steps.",
    )
    parser_agent = Agent(
        name="Parser",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds.
        """,
    )
    task.add_solving_agent(parser_agent)
    plan = task.run()

    robot_grounded_plan = Task(
        f"Map the following steps to the Available Robot Skills: \n{plan}" +
            f"\n\nAvailable Robot Skills: {SKILLS} " +
            "\n\nIf there is no match for that step, return 'False'. Be conservative in the matching. There shall only be one skill per step. Summarize if the plan if feasible at the end.",
        expected_output_format="A numbered list of steps mapped to single skill each or 'False' followed by a summary if the task is feaible.",
    )
    robot_grounded_agent = Agent(
        name="Robot Grounded",
        model=DEFAULT_MODEL,
        system_message="""
        You are an agent that grounds a set of actions to robot skills.
        """,
    )
    robot_grounded_plan.add_solving_agent(robot_grounded_agent)
    robot_grounded_plan_output = robot_grounded_plan.run()

    extract_feasibility = Task(
        f"Given the following summary, return if the task is feasible. Summary: \n{robot_grounded_plan_output}",
        expected_output_format="True or False. Do not add any other information.",
    )
    feasibility_agent = Agent(
        name="Feasibility",
        # model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
        model=DEFAULT_MODEL,
        system_message="""
        You are a conservative agent that determines if a plan is feasible.
        """,
    )
    extract_feasibility.add_solving_agent(feasibility_agent)
    feasibility_output = extract_feasibility.run()

    feasible = get_closest_text(feasibility_output, ["True", "False"])
    feasible = True if feasible == "True" else False if feasible is not None else None

    result = {
        "task": robot_grounded_plan_output,
        "feasible": feasible,
    }
   
    return result, state.update(**result)

@action(reads=["task", "feasible"], writes=["response", "current_state", "task"])
def convert_plan_to_steps(state: State) -> Tuple[dict, State]:
    plan = state["task"]

    plan_to_list_of_steps = Task(
        f"Given the following output, take the numbered list and return is as a python list assigned to `list_of_steps`: \n{plan}",
        expected_output_format="""
        ```python
            list_of_steps = ["Go to the required location", "Identify the objects to manipulate", "Form a plan to pick and place the objects", ...]
        ```
        """,
    )
    plan_to_list_of_steps_agent = Agent(
        name="Plan to List of Steps",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds with only code.
        """,
    )
    plan_to_list_of_steps.add_solving_agent(plan_to_list_of_steps_agent)
    output = plan_to_list_of_steps.run()
    code = extract_code(output)
    try:
        exec_vars = {}
        exec(code, exec_vars)
        log.info(exec_vars.get("list_of_steps", None))
        steps = exec_vars.get("list_of_steps", None)
    except Exception as e:
        log.error(f"Error executing code: {e}")
        steps = None    
    
    # formatted_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    feasible = state["feasible"]
    current_state = "STARTING" if feasible else "DONE"

    result = {
        "response": {
            "content": f"Performing task: \n\n{plan}\n\nRobot Feasibility: **{feasible}**",
            "type": "text",
            "role": "assistant",
        },
        "current_state": current_state,
        "task": steps
    }
    
    return result, state.update(**result)


@action(reads=["task"], writes=["response", "current_state"])
def confirm_task(state: State) -> Tuple[dict, State]:
    result = {
        "response": {
            "content": f"Performing task: **{state['task']}**",
            "type": "text",
            "role": "assistant",
        },
        "current_state": "STARTING",
    }
    return result, state.update(**result)


@action(reads=["task"], writes=["state_machine", "task_state", "task_state_idx", "current_state"])
def create_state_machine(state: State) -> Tuple[dict, State]:
    '''
    Create a viable state machine for the task.
    Every task requires:
    * the robot and environment state
    * ensuring the robot has the skills to perform the required steps
    '''
    if state["task"] == "What is on the table?":
        result = {
            "state_machine": [
                "get_image",
                "ask_vla",
                "get_list_of_objects",
            ],
            "task_state": "not_started",
            "current_state": "RUNNING",
        }
    elif state["task"] == "Clear the table":
        result = {
            "state_machine": [
                "get_image",
                "ask_vla",
                "get_list_of_objects",
                "create_plan",
                "code",
                "execute_code",
            ],
            "task_state": "not_started",
            "current_state": "RUNNING",
        }
    elif state["task"] == "unknown":
        result = {
            "state_machine": "unknown",
            "task_state": "unknown",
            "current_state": "DONE",
        }
    else:
        plan = state["task"]
        # create_sm_task = Task(
        #     f"Given the following plan, extract the robot actions and return them as a list assigned to `state_machine` in python: \n{plan}" +
        #         f"\n\nHere is the list of available robot skills: {SKILLS}",
        #     expected_output_format="""
        #     ```python
        #         state_machine = ["get image", "get information about the scene", "pick object"]
        #     ```
        #     """,
        # )
        # create_sm_agent = Agent(
        #     name="State Machine Creator",
        #     model=DEFAULT_MODEL,
        #     system_message="""
        #     You are a helpful agent that concisely responds with only code.
        #     Use only the provided functions, do not add any extra code.
        #     """,
        # )
        # create_sm_task.add_solving_agent(create_sm_agent)
        # output = create_sm_task.run()
        # code = extract_code(output)
        # try:
        #     exec_vars = {}
        #     exec(code, exec_vars)
        #     log.info(exec_vars.get("state_machine", None))
        #     state_machine = exec_vars.get("state_machine", None)
        #     state_machine = [get_closest_text(s, SKILLS) for s in state_machine]
        #     # assert len(plan) == len(state_machine), "Number of steps do not match the plan" 
        #     result = {
        #         "state_machine": state_machine,
        #         "task_state": "output_state_machine",
        #         "current_state": "RUNNING",
        #     }
            
        # except Exception as e:
        #     log.error(f"Error executing code: {e}")
        #     result = {
        #         "state_machine": "unknown",
        #         "task_state": "unknown",
        #         "current_state": "DONE",
        #     }
        state_machine = [get_list_of_objects(step) for step in plan]
        result = {
            "state_machine": state_machine,
            "task_state": "output_state_machine",
            "current_state": "RUNNING",
        }
        result["task_state_idx"] = 0
    return result, state.update(**result)


@action(reads=["state_machine", "task_state_idx"], writes=["task_state", "task_state_idx", "current_state", "state_machine"])
def execute_state_machine(state: State) -> Tuple[dict, State]:
    current_state = "RUNNING"
    state_machine = state["state_machine"]
    if state["task_state"] == "not_started":
        task_state = state["state_machine"][0]
        task_state_idx = state["task_state_idx"]
    elif state["task_state"] == "output_state_machine":
        task_state = "not_started"
        task_state_idx = state["task_state_idx"]
    else:
        task_state_idx = state["task_state_idx"] + 1
        if task_state_idx < len(state["state_machine"]):
            task_state = state["state_machine"][task_state_idx]
        else:
            task_state = "done"
            current_state = "DONE"
    
    if task_state == "get information about the scene":
        task_state = "get_image"
        state_machine[task_state_idx] = task_state
        state_machine.insert(task_state_idx + 1, "ask_vla")
        state_machine.insert(task_state_idx + 2, "get_list_of_objects")
    result = {"task_state": task_state, "task_state_idx": task_state_idx, "current_state": current_state, "state_machine": state_machine}
    return result, state.update(**result)

@action(reads=["state_machine", "task_state"], writes=["response"])
def output_state_machine(state: State) -> Tuple[dict, State]:
    result = {
        "response": {
            "content": f"State Machine: \n\n{state['state_machine']}",
            "type": "text",
            "role": "assistant",
        }
    }
    return result, state.update(**result)

@action(reads=["state_machine", "task", "task_state", "task_state_idx"], writes=["location"])
def navigate_to_location(state: State) -> Tuple[dict, State]:
    # location = "kitchen"
    # result = {"location": location}
    step = state["task"][state["task_state_idx"]]
    extract_location = Task(
        f"Given the following step, extract the location (e.g. kitchen), item (e.g. sink) or destination and return it as a string assigned to `location` in python. Here is the string to extract the location: \n{step}",
        expected_output_format="""
        ```python
            location = "kitchen"
        ```
        """,
    )
    extract_location_agent = Agent(
        name="Location Extractor",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds with only code.
        """,
    )
    extract_location.add_solving_agent(extract_location_agent)
    output = extract_location.run()
    code = extract_code(output)
    try:
        exec_vars = {}
        exec(code, exec_vars)
        log.info(exec_vars.get("location", None))
        location = exec_vars.get("location", None)
    except Exception as e:
        log.error(f"Error executing code: {e}")
        result = {"location": None}
    
    location = get_closest_text(location, list(SEMANTIC_LOCATIONS.keys()))
    result = {"location": {location}}
    navigate_to(SEMANTIC_LOCATIONS[location]["name"], SEMANTIC_LOCATIONS[location]["location"])

    return result, state.update(**result)

@action(reads=["state_machine"], writes=["image"])
def get_image(state: State) -> Tuple[dict, State]:
    image = get_image_from_sim()
    # image = Image.open("shared/data/test1.png")
    image = pil_to_b64(image)
    result = {"image": image}
    return result, state.update(**result)


@action(reads=["image"], writes=["vla_response"])
def ask_vla(
    state: State, vla_prompt: str = "Describe the image."
) -> Tuple[dict, State]:
    image = b64_to_pil(state["image"])
    result = {"vla_response": moondream(image, vla_prompt)["result"]}
    return result, state.update(**result)


@action(reads=["vla_response"], writes=["relevant_vars"])
def get_list_of_objects(state: State) -> Tuple[dict, State]:
    task = Task(
        f"Given the following summary, return just a list in python assigned to `objects_on_table` of the objects on the table. The table is not an object. Summary: \n{state['vla_response']}",
        expected_output_format="""
            ```python
                objects_on_table = ["Object 1", "Object 2", "Object 3"]
            ```
            """,
    )
    analyzer_agent = Agent(
        name="Analyzer",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds with only code.
        Use only the provided functions, do not add any extra code.
        """,
    )
    task.add_solving_agent(analyzer_agent)
    output = task.run()
    code = extract_code(output)
    try:
        exec_vars = {}
        exec(code, exec_vars)
        log.info(exec_vars.get("objects_on_table", None))
        objects_on_table = exec_vars.get("objects_on_table", None)
        relevant_vars = {"objects_on_table": objects_on_table}
    except Exception as e:
        log.error(f"Error executing code: {e}")
    result = {"relevant_vars": relevant_vars}
    return result, state.update(**result)


@action(reads=["relevant_vars", "task", "prompt"], writes=["plan_prompt"])
def create_plan_prompt(state: State) -> Tuple[dict, State]:
    if state["task"] == "Clear the table":
        plan_prompt = f"""Create a plan for a robot to {state["task"]}:
            {state["relevant_vars"]}
            Do not add any extra steps.
        """
    else:
        plan_prompt = "Unknown task"
    result = {"plan_prompt": plan_prompt}
    return result, state.update(**result)


@action(reads=["task", "plan_prompt"], writes=["plan_prompt", "similar_plans"])
def check_for_similar_plans(state: State) -> Tuple[dict, State]:
    plan_prompt = state["plan_prompt"]
    similar_plans = get_closest_text(state["task"], PLANS.values())
    if similar_plans:
        plan_prompt += f"Here is a template to follow: {similar_plans}"
    result = {"plan_prompt": plan_prompt, "similar_plans": similar_plans}
    return result, state.update(**result)


@action(
    reads=["plan_prompt", "similar_plans"], writes=["plan_prompt", "relevant_skills"]
)
def check_for_relevant_skills(state: State) -> Tuple[dict, State]:
    plan_prompt = state["plan_prompt"]
    relevant_skills = get_closest_text(state["similar_plans"], SKILLS, k=2)
    if relevant_skills:
        plan_prompt += f"You can only use the following actions: {relevant_skills}"
    result = {"plan_prompt": plan_prompt, "relevant_skills": relevant_skills}
    return result, state.update(**result)


@action(reads=["plan_prompt", "similar_plans"], writes=["plan"])
def create_plan(state: State) -> Tuple[dict, State]:
    plan_task = Task(
        state["plan_prompt"],
        # expected_output_format="""
        #     1. pick object1
        #     2. place object1
        #     3. pick object2
        #     4. place object2
        #     5. pick object3
        #     6. place object3
        # """
        expected_output_format="A numbered list of steps.",
    )

    # plan_task.register_tool(
    #     name="pick",
    #     func=pick,
    #     description="Robot picks up the provided arg 'object_name'",
    #     example='"pick_success = pick(object_name="Object 1")" --> Returns: True ',
    # )

    # plan_task.register_tool(
    #     name="place",
    #     func=place,
    #     description="Robot places the provided arg 'object_name'",
    #     example='"place_success = place(location="place location")" --> Returns: True ',
    # )

    planner_agent = Agent(
        name="Planner",
        model=DEFAULT_MODEL,
        system_message="""
        You are a planner that breaks down tasks into steps for robots.
        Create a conscise set of steps that a robot can do.
        Do not add any extra steps.
        """
        # + plan_task.generate_tool_prompt(),
    )

    plan_task.add_solving_agent(planner_agent)
    # log.info(plan_task)
    output = plan_task.run()
    log.info(output)
    result = {"plan": output}
    return result, state.update(**result)


@action(reads=["plan"], writes=["code", "exec_vars"])
def convert_plan_to_code(state: State) -> Tuple[dict, State]:
    coder_task = Task(
        f"""Return python code to execute the plan:
            {state["plan"]}
            Convert the plan to code using only the following functions, do not add any extra code or imports.
        """
    )
    coder_task.register_tool(
        name="pick",
        func=pick,
        description="Robot picks up an object",
        example='"pick_success = pick(object_name)" --> Returns: True ',
    )
    coder_task.register_tool(
        name="place",
        func=place,
        description="Robot places an object",
        example='"place_success = place(location="place location")" --> Returns: True ',
    )
    coder_agent = Agent(
        name="Coder",
        model=CODING_MODEL,
        system_message="""
        You are a coder that writes concise and exact code to execute the plan.
        Use only the provided functions. No additional imports.
        """
        + coder_task.generate_tool_prompt(),
    )
    coder_task.add_solving_agent(coder_agent)
    log.info(coder_task)
    output = coder_task.run()
    code = extract_code(output)
    result = {"code": code, "exec_vars": coder_task.get_exec_vars_serialized()}
    return result, state.update(**result)


@action(reads=["code", "exec_vars"], writes=["execution"])
def validate_code(state: State) -> Tuple[dict, State]:
    try:
        exec_vars = state["exec_vars"]
        exec_vars = {k: globals().get(v, None) for k, v in exec_vars.items()}
        # convert to mock functions
        for k, v in exec_vars.items():
            if callable(v):
                exec_vars[k] = globals()[f"{k}_mock"]
        exec_vars["test_mode"] = True
        exec(state["code"], exec_vars)
        execution = "SUCCESS"
    except Exception as e:
        log.error(f"Error executing code: {e}")
        execution = f"Error executing code: {e}"
    result = {"execution": execution}
    return result, state.update(**result)


@action(reads=["code", "execution", "code_attempts"], writes=["code", "code_attempts"])
def iterate_code(state: State) -> Tuple[dict, State]:
    coder_task = Task(
        f"""Given the following error, fix the code. 
        Error: \n{state['execution']}
        Code: \n{state['code']} 
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
        example='"place_success = place(location="place location")" --> Returns: True ',
    )
    coder_agent = Agent(
        name="Coder",
        model=CODING_MODEL,
        system_message="""
        You are a coder that writes concise and exact code to execute the plan.
        Use only the provided functions.
        """
        + coder_task.generate_tool_prompt(),
    )
    coder_task.add_solving_agent(coder_agent)
    log.info(coder_task)
    output = coder_task.run()
    code = extract_code(output)
    result = {"code": code, "code_attempts": state.get("code_attempts", 0) + 1}
    return result, state.update(**result)


@action(reads=["code"], writes=["execution"])
def execute_code(state: State) -> Tuple[dict, State]:
    result = {"execution": "Execution successful!"}
    return result, state.update(**result)


@action(reads=["task", "safe"], writes=["response"])
def prompt_for_more(state: State) -> Tuple[dict, State]:
    result = {
        "response": {
            "content": "None of the response modes I support apply to your question. Please clarify?",
            "type": "text",
            "role": "assistant",
        }
    }
    return result, state.update(**result)


@action(reads=["task_state"], writes=["response"])
def create_response_for_task_state(state: State) -> Tuple[dict, State]:
    if state["task_state"] == "navigate to location":
        result = {
            "response": {
                "content": f"Robot is now at **{state['location']}**.",
                "type": "text",
                "role": "assistant",
            }
        }
    elif state["task_state"] == "get_image":
        result = {
            "response": {
                "content": state["image"],
                "type": "image",
                "role": "assistant",
            }
        }
    elif state["task_state"] == "ask_vla":
        result = {
            "response": {
                "content": state["vla_response"],
                "type": "text",
                "role": "assistant",
            }
        }
    elif state["task_state"] == "get_list_of_objects":
        result = {
            "response": {
                "content": state["relevant_vars"]["objects_on_table"],
                "type": "code",
                "role": "assistant",
            }
        }
    elif state["task_state"] == "create_plan":
        result = {
            "response": {
                "content": state["plan"],
                "type": "text",
                "role": "assistant",
            }
        }
    elif state["task_state"] == "code":
        result = {
            "response": {
                "content": state["code"],
                "type": "code",
                "role": "assistant",
            }
        }
    elif state["task_state"] == "execute_code":
        result = {
            "response": {
                "content": state["execution"],
                "type": "text",
                "role": "assistant",
            }
        }
    elif state["task_state"] == "done":
        result = {
            "response": {
                "content": "I have completed the task.",
                "type": "text",
                "role": "assistant",
            }
        }
    return result, state.update(**result)


@action(
    reads=["current_state", "task_state"],
    writes=["response", "current_state", "code_attempts"],
)
def create_error_response(state: State) -> Tuple[dict, State]:
    result = {
        "response": {
            "content": f"I have failed on {state['task_state']}.",
            "type": "error",
            "role": "assistant",
        },
        "current_state": "DONE",
        "code_attempts": 0,
    }
    return result, state.update(**result)


@action(reads=["response", "current_state"], writes=["chat_history", "current_state"])
def response(state: State) -> Tuple[dict, State]:
    if state["current_state"] == "DONE":
        current_state = "PENDING"
    elif state["current_state"] == "PARSING":
        current_state = "PARSING"
    else:
        current_state = state["current_state"]
    result = {"chat_item": state["response"], "current_state": current_state}
    return result, state.append(chat_history=state["response"]).update(**result)


MAX_CODE_ATTEMPTS = 3


def base_application(
    hooks: List[LifecycleAdapter],
    app_id: str,
    storage_dir: str,
    project_id: str,
):
    if hooks is None:
        hooks = []
    # we're initializing above so we can load from this as well
    # we could also use `with_tracker("local", project=project_id, params={"storage_dir": storage_dir})`
    tracker = LocalTrackingClient(project=project_id, storage_dir=storage_dir)
    sequence_id = None
    return (
        ApplicationBuilder()
        .with_actions(
            prompt=process_prompt,
            determine_task=determine_task,
            response_for_unknown_task=response_for_unknown_task,
            parse_unknown_task=parse_unknown_task,
            convert_plan_to_steps=convert_plan_to_steps,
            confirm_task=confirm_task,
            create_state_machine=create_state_machine,
            output_state_machine=output_state_machine,
            execute_state_machine=execute_state_machine,
            navigate_to_location=navigate_to_location,
            get_image=get_image,
            ask_vla=ask_vla,
            get_list_of_objects=get_list_of_objects,
            create_plan_prompt=create_plan_prompt,
            check_for_similar_plans=check_for_similar_plans,
            check_for_relevant_skills=check_for_relevant_skills,
            create_plan=create_plan,
            code=convert_plan_to_code,
            validate_code=validate_code,
            iterate_code=iterate_code,
            create_error_response=create_error_response,
            execute_code=execute_code,
            create_response=create_response_for_task_state,
            prompt_for_more=prompt_for_more,
            response=response,
        )
        .with_transitions(
            ("prompt", "determine_task", default),
            ("determine_task", "response_for_unknown_task", when(task="unknown")),
            ("response_for_unknown_task", "response", default),
            ("parse_unknown_task", "convert_plan_to_steps", default),
            ("convert_plan_to_steps", "response_for_unknown_task", when(task="unknown")),
            ("convert_plan_to_steps", "response", default),
            ("determine_task", "confirm_task", default),
            ("confirm_task", "response", default),
            ("create_state_machine", "execute_state_machine", default),
            ("create_state_machine", "prompt_for_more", when(state_machine="unknown")),
            ("execute_state_machine", "output_state_machine", when(task_state="not_started")),
            ("output_state_machine", "response", default),
            ("execute_state_machine", "navigate_to_location", when(task_state="navigate to location")),
            ("execute_state_machine", "get_image", when(task_state="get_image")),
            ("execute_state_machine", "ask_vla", when(task_state="ask_vla")),
            (
                "execute_state_machine",
                "get_list_of_objects",
                when(task_state="get_list_of_objects"),
            ),
            (
                "execute_state_machine",
                "create_plan_prompt",
                when(task_state="create_plan"),
            ),
            ("create_plan_prompt", "check_for_similar_plans", default),
            ("check_for_similar_plans", "create_plan", when(similar_plans=None)),
            ("check_for_similar_plans", "check_for_relevant_skills", default),
            ("check_for_relevant_skills", "create_plan", default),
            ("execute_state_machine", "code", when(task_state="code")),
            (
                "execute_state_machine",
                "execute_code",
                when(task_state="execute_code"),
            ),
            ("execute_state_machine", "create_response", when(task_state="done")),
            ("navigate_to_location", "create_response", default),
            ("get_image", "create_response", default),
            ("ask_vla", "create_response", default),
            ("get_list_of_objects", "create_response", default),
            ("create_plan", "create_response", default),
            ("code", "validate_code", default),
            ("validate_code", "create_response", when(execution="SUCCESS")),
            ("validate_code", "iterate_code", default),
            (
                "iterate_code",
                "create_error_response",
                when(code_attempts=MAX_CODE_ATTEMPTS),
            ),
            ("create_error_response", "response", default),
            ("iterate_code", "validate_code", default),
            ("execute_code", "create_response", default),
            ("create_response", "response", default),
            ("response", "parse_unknown_task", when(current_state="PARSING")),
            ("response", "prompt", when(current_state="PENDING")),
            ("response", "execute_state_machine", when(current_state="RUNNING")),
            ("response", "create_state_machine", when(current_state="STARTING")),
            ("prompt_for_more", "response", default),
        )
        # initializes from the tracking log if it does not already exist
        .initialize_from(
            tracker,
            resume_at_next_action=True,  # always resume from entrypoint in the case of failure
            default_state={"chat_history": [], "current_state": "PENDING"},
            default_entrypoint="prompt",
            # fork_from_app_id="000431d2-949d-49ea-b440-302bdb1c6a9d",
            # fork_from_sequence_id=12,
        )
        .with_hooks(*hooks)
        .with_tracker(tracker)
        .with_identifiers(app_id=app_id, sequence_id=sequence_id)
        .build()
    )


def application(
    app_id: Optional[str] = None,
    project_id: str = "roboai",
    storage_dir: Optional[str] = "~/.burr",
    hooks: Optional[List[LifecycleAdapter]] = None,
) -> Application:
    return base_application(hooks, app_id, storage_dir, project_id=project_id)


if __name__ == "__main__":
    app = application()
    # app.visualize(
    #     output_file_path="statemachine", include_conditions=False, view=False, format="png"
    # )
    app.run(halt_after=["response"])
