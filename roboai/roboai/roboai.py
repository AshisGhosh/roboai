import time
from typing import List, Optional, Tuple
from PIL import Image  # noqa: F401
from enum import Enum
from uuid import uuid4

from burr.core import Application, ApplicationBuilder, State, default, when
from burr.core.action import action
from burr.lifecycle import LifecycleAdapter
from burr.tracking import LocalTrackingClient

from shared.utils.llm_utils import (
    get_closest_text_sync as get_closest_text,
    get_most_important_sync as get_most_important,
)

# from shared.utils.isaacsim_client import get_image as get_image_from_sim, pick, place  # noqa: F401
from shared.utils.omnigibson_client import (
    get_image as get_image_from_sim,
    pick,
    place,
    navigate_to,
    get_obj_in_hand,
    wait_until_ready,
)  # noqa: F401
from shared.utils.image_utils import pil_to_b64, b64_to_pil
from shared.utils.gradio_client import moondream_answer_question_from_image as moondream

from task import Task
from agent import Agent
from plans import PLANS
from skills import SKILLS
from semantic_locations import SEMANTIC_LOCATIONS
from role_context import ROBOT_CONTEXT, ROLE_CONTEXT, EMPLOYEE_HANDBOOK
from knowledge_base_utils import KnowledgeBase

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

knowledge_base = KnowledgeBase()

DEFAULT_MODEL = "openrouter/microsoft/phi-3-medium-128k-instruct:free"
# DEFAULT_MODEL = "openrouter/meta-llama/llama-3-8b-instruct:free"
# DEFAULT_MODEL = "openrouter/huggingfaceh4/zephyr-7b-beta:free"
# DEFAULT_MODEL = "ollama/llama3:latest"
# DEFAULT_MODEL = "ollama/phi3"
# CODING_MODEL = "ollama/codegemma:instruct"
CODING_MODEL = DEFAULT_MODEL


class LogColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


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


def exec_code(code, exec_vars, attempts=3):
    success = False
    history = []
    for _ in range(attempts):
        try:
            if history:
                log.warn(
                    f"{LogColors.WARNING}Executing code, retry attempt {len(history)}{LogColors.ENDC}"
                )
                coder_task = Task(
                    f"""Given the following error, fix the syntax. Here is the error: 
                    {history[-1][1]}\n{code}
                    Ensure any explanations are formatted as comments.
                    """,
                    expected_output_format="""
                        ```python
                        # explanatations are only formatted as comments
                        my_variable = "proper_python_syntax"
                        my_list = ["proper_python_syntax"]
                        ```
                    """,
                )
                coder_agent = Agent(
                    name="Coder",
                    model=CODING_MODEL,
                    system_message="You are an expert coder. Only return proper syntax.",
                )
                coder_task.add_solving_agent(coder_agent)
                output = coder_task.run()
                code = extract_code(output)
                log.info(f"{LogColors.OKBLUE}Fixed code: \n{code}{LogColors.ENDC}")
            exec(code, exec_vars)
            success = True
            break
        except Exception as e:
            log.error(f"Error executing code: {e}")
            history.append((code, f"Error executing code: {e}"))
            time.sleep(1)

    return success


@action(reads=["current_state"], writes=["chat_history", "prompt"])
def process_prompt(state: State, prompt: str) -> Tuple[dict, State]:
    result = {"chat_item": {"role": "user", "content": prompt, "type": "text"}}
    if state["current_state"] == "FAILED":
        original_prompt = state["prompt"]
        if isinstance(original_prompt, str):
            prompt = [original_prompt, prompt]
        elif isinstance(original_prompt, list):
            prompt = original_prompt.append(prompt)
    log.info(f"{LogColors.OKCYAN}Prompt: {prompt}{LogColors.ENDC}")
    return result, state.append(chat_history=result["chat_item"]).update(prompt=prompt)


class PromptType(Enum):
    UNKNOWN = "unknown"
    PERFORM_NEW_TASK = "perform new task"
    RESPOND_TO_QUESTION = "respond to question"
    UPDATE_KNOWLEDGE_BASE = "update knowledge base"
    RETRY_EXISTING_TASK = "retry existing task"
    MODIFY_EXISTING_TASK = "modify existing task"

    @classmethod
    def all_values(cls):
        return [prompt.value for prompt in cls]

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, PromptType):
            return self.value == other.value
        return NotImplemented


@action(reads=["prompt"], writes=["current_state", "prompt_cls", "task_state_idx"])
def parse_prompt(state: State) -> Tuple[dict, State]:
    # determine the mode of the prompt
    prompt = state["prompt"]
    if state["current_state"] == "FAILED":
        prompt = state["prompt"][-1]

    options = PromptType.all_values()
    classify_prompt = Task(
        f"""Given the following prompt, classify the type of prompt it is:
        {prompt}
        Options: {options}
        Examples
        - Prompt: "Go to the kitchen and grab the items" Output: "perform new task"
        - Prompt: "What skills do you have?" Output: "respond to question"
        - Prompt: "Don't handle bottles." Output: "update knowledge base"
        - Prompt: "Remember to ignore the red cup." Output: "update knowledge base"
        - Prompt: "You can find the items in the kitchen." Output: "update knowledge base"
        - Prompt: "The beer is in the fridge." Output: "update knowledge base"
        - Prompt: "Beer goes in the fridge." Output: "update knowledge base"
        - Prompt: "I want to retry the task" Output: "retry existing task"
        - Prompt: "Don't pick up the items, just scan the scene" Output: "modify existing task"
        """,
        expected_output_format="A string respresenting the classification.",
    )
    classify_prompt_agent = Agent(
        name="Update Classifier",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds.
        """,
    )
    classify_prompt.add_solving_agent(classify_prompt_agent)
    output = classify_prompt.run()

    prompt_cls = get_closest_text(output, options)
    log.info(f"{LogColors.OKGREEN}Prompt classification: {prompt_cls}{LogColors.ENDC}")

    task_state_idx = 0
    if prompt_cls == PromptType.PERFORM_NEW_TASK:
        current_state = "STARTING"
        content = "Starting a new task..."
    elif prompt_cls == PromptType.RESPOND_TO_QUESTION:
        current_state = "PENDING"
        content = "Responding to a question..."
    elif prompt_cls == PromptType.UPDATE_KNOWLEDGE_BASE:
        current_state = "PENDING"
        content = "Updating knowledge..."
    elif prompt_cls == PromptType.RETRY_EXISTING_TASK:
        current_state = "RUNNING"
        content = "Retrying the task..."
        task_state_idx = state["task_state_idx"] - 1
        if state["current_state"] != "FAILED":
            current_state = "PENDING"
            prompt_cls = PromptType.UNKNOWN
            content = "There is no task to retry."
    elif prompt_cls == PromptType.MODIFY_EXISTING_TASK:
        current_state = "PENDING"
        content = "Modifying existing task...\n\nThis feature is not supported yet."
    else:
        prompt_cls = PromptType.UNKNOWN
        current_state = "PENDING"
        content = "Unknown prompt. Please provide a valid prompt."

    result = {
        "current_state": current_state,
        "prompt_cls": prompt_cls,
        "task_state_idx": task_state_idx,
    }

    chat_item = {
        "content": content,
        "type": "text",
        "role": "assistant",
    }

    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["prompt"], writes=[])
def respond_to_question(state: State) -> Tuple[dict, State]:
    answer_question = Task(
        f"""
        Given the following prompt, answer the question. 
        Here is the prompt: \n{state['prompt']}

        Here is context:
        {ROBOT_CONTEXT}
        {ROLE_CONTEXT}
        {EMPLOYEE_HANDBOOK}
        Locations available: {list(SEMANTIC_LOCATIONS.keys())}
        Skills available: {list(SKILLS.keys())}
        Previous plans: {list(PLANS.keys())}

        Here is additional knowledge you've learned: 
        {knowledge_base.get_knowledge_as_string()}    
    
        If you do not know the answer, return "I do not know."
        """,
        expected_output_format="A string representing the answer.",
    )
    answer_question_agent = Agent(
        name="Answerer",
        model=DEFAULT_MODEL,
        system_message="""
        You are an agent that answers questions.
        """,
    )
    answer_question.add_solving_agent(answer_question_agent)
    output = answer_question.run()
    chat_item = {
        "content": output,
        "type": "text",
        "role": "assistant",
    }
    return {}, state.append(chat_history=chat_item)


@action(reads=["prompt"], writes=[])
def update_knowledge_base(state: State) -> Tuple[dict, State]:
    parse_knowledge = Task(
        f"""Given the following text, extract the knowledge: \n{state['prompt']}

        Do not include any other text that is not the extracted knowledge.

        Examples:
        - Text: "Don't handle bottles." Output: "Don't handle bottles"
        - Text: "Remember to ignore the red cup." Output: "Ignore the red cup"
        - Text: "If you're looking for the cheese, there's a very high chance you could find it in the kitchen." Output: "Cheese in the kitchen"
        """,
        expected_output_format="A string representing the knowledge.",
    )
    parse_knowledge_agent = Agent(
        name="Knowledge Extractor",
        model=DEFAULT_MODEL,
        system_message="""
        You are an agent that extracts knowledge.
        """,
    )
    parse_knowledge.add_solving_agent(parse_knowledge_agent)
    new_knowledge = parse_knowledge.run()

    knowledge_tags = get_most_important(new_knowledge, 2)
    # make tags lower case and strip any non-alphanumeric characters
    knowledge_tags = [tag.lower().strip() for tag in knowledge_tags]

    knowledge_base.add_data(uuid4().hex, new_knowledge, knowledge_tags)
    chat_item = {
        "content": f"""Updated knowledge base with:
                    {new_knowledge}
                    Tags: {knowledge_tags}""",
        "type": "text",
        "role": "assistant",
    }
    return {}, state.append(chat_history=chat_item)


@action(reads=["prompt"], writes=["task"])
def determine_if_task_in_skill_library(state: State) -> Tuple[dict, State]:
    closest_text = get_closest_text(state["prompt"], list(PLANS.keys()), threshold=0.75)
    if closest_text:
        result = {"task": closest_text}
        content = f"Task determined to be **{result['task']}**"
    else:
        result = {"task": "unknown"}
        content = f"Parsing unknown task... **{state['prompt']}**"

    chat_item = {"role": "assistant", "content": content, "type": "text"}
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["prompt"], writes=["task"])
def create_plan_for_unknown_task(state: State) -> Tuple[dict, State]:
    result = {"task": state["prompt"]}
    chat_item = {
        "content": f"Creating plan for **{state['prompt']}**",
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["prompt"], writes=["closest_plans"])
def get_closest_plans(state: State) -> Tuple[dict, State]:
    closest_plans = get_closest_text(
        state["prompt"], list(PLANS.keys()), k=2, threshold=0.75
    )
    chat_item = {
        "content": f"Closest plans: {closest_plans}",
        "type": "text",
        "role": "assistant",
    }
    result = {"closest_plans": closest_plans}
    return result, state.append(chat_history=chat_item).update(
        closest_plans=closest_plans
    )


@action(
    reads=["prompt", "closest_plans"],
    writes=[
        "robot_context",
        "role_context",
        "employee_context",
    ],
)
def get_role_and_location_context(state: State) -> Tuple[dict, State]:
    result = {
        "robot_context": ROBOT_CONTEXT,
        "role_context": ROLE_CONTEXT,
        "employee_context": EMPLOYEE_HANDBOOK,
    }
    chat_item = {
        "content": "Getting role and location context. Creating initial plan.",
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(
    reads=[
        "prompt",
        "closest_plans",
        "robot_context",
        "role_context",
        "employee_context",
    ],
    writes=["plan"],
)
def create_initial_plan(state: State) -> Tuple[dict, State]:
    closest_plans = state["closest_plans"]
    if closest_plans:
        closest_plans = [f"{k}:\n   {PLANS[k]}" for k in closest_plans]
        closest_plans = "\n".join(closest_plans)

    task = Task(
        f"""
        Given the following prompt and current robot state, return a simplified high level plan for a robot to perform. 
        Do not include steps related to confirming successful execution or getting feedback. Do not include steps that indicate to repeat steps.

        Prompt:
        {state['prompt']}
        
        Current robot location: {state['location'] if 'location' in state else 'living room'}

        Examples: 
        {closest_plans if closest_plans else "No examples available."}

        If information is needed, use the skills to get observe or scan the scene.

        Context:
            Robot: 
            {state['robot_context']}
            Role:
            {state['role_context']}
            Employee Handbook:
            {state['employee_context']}
        
        Here is a list of locations that the robot can go to:
        {list(SEMANTIC_LOCATIONS.keys())}

        Here is additional knowledge you've learned:
        {knowledge_base.get_knowledge_as_string()}

        """,
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
    chat_item = {
        "content": f"Initial plan: \n\n{plan}",
        "type": "text",
        "role": "assistant",
    }
    return {"plan": plan}, state.append(chat_history=chat_item).update(plan=plan)


@action(reads=["plan"], writes=["plan"])
def create_robot_grounded_plan(state: State) -> Tuple[dict, State]:
    plan = state["plan"]
    skills_verbose = "\n".join([f"{k} : {v['description']}" for k, v in SKILLS.items()])
    robot_grounded_plan = Task(
        f"""
        Map and consolidate the following steps to the Available Robot Skills and locations. 
        {plan}

        Try to match the number of steps. Do not add any additional steps.

        Available Robot Skills:
        {skills_verbose}

        If there is no match for that step, return 'False'. Be conservative in the matching. There shall only be one skill per step. Summarize if the plan if feasible at the end.
        
        Here is a list of locations that the robot can go to: {list(SEMANTIC_LOCATIONS.keys())}

        If there are pick and place steps following an observation or scanning step, consolidate those steps into a rollout step for a pick and place plan.
        """,
        expected_output_format="A numbered list of steps mapped to single skill each or 'False' followed by a summary if the task is feasible.",
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
    chat_item = {
        "content": f"**Robot grounded plan**: \n\n{robot_grounded_plan_output}",
        "type": "text",
        "role": "assistant",
    }
    return {"plan": robot_grounded_plan_output}, state.append(
        chat_history=chat_item
    ).update(plan=robot_grounded_plan_output)


@action(reads=["plan"], writes=["feasible"])
def determine_if_plan_is_feasibile(state: State) -> Tuple[dict, State]:
    robot_grounded_plan_output = state["plan"]
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

    chat_item = {
        "content": f"Feasible: `{result['feasible']}`",
        "type": "text",
        "role": "assistant",
    }

    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["task", "feasible"], writes=["response", "current_state", "task"])
def convert_plan_to_steps(state: State) -> Tuple[dict, State]:
    plan = state["task"]

    plan_to_list_of_steps = Task(
        f"""Given the following output, take the numbered list and return is as a python list assigned to `list_of_steps`. 
        Do not remove any relevant information. Include information about skills and locations into the correct list item.
        Here is the plan:
        {plan}

        Here is an example:
        Plan:
            1. navigate to location, the location is the kitchen
            2. scan the kitchen for relevant objects
            3. roll out a plan to pick and place the objects
        Output:
            ```python
                list_of_steps = ["navigate to location, the location is the kitchen", "scan the kitchen for relevant objects", "roll out a plan to pick and place the objects"]
            ``` 
        """,
        expected_output_format="""
        ```python
            list_of_steps = ["step 1", "step 2", "step 3"]
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
        exec_code(code, exec_vars)
        log.info(exec_vars.get("list_of_steps", None))
        steps = exec_vars.get("list_of_steps", None)
        content = "Steps:\n\n" + "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(steps)]
        )
    except Exception as e:
        log.error(f"Error executing code: {e}")
        steps = None
        content = "Failed to extract steps. Please check the plan and try again."

    # formatted_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    feasible = state["feasible"]
    current_state = "STARTING" if feasible else "DONE"

    result = {
        "response": {
            "content": content,
            "type": "text" if steps is not None else "error",
            "role": "assistant",
        },
        "current_state": current_state,
        "task": steps,
    }

    return result, state.append(chat_history=result["response"]).update(**result)


def get_closest_state_from_skills(step: str, skills: dict) -> str:
    skill_descriptions = [s["description"] for s in skills.values()]
    closest_description = get_closest_text(step, skill_descriptions)
    state_idx = skill_descriptions.index(closest_description)

    return list(skills.keys())[state_idx]


@action(
    reads=["task"],
    writes=["state_machine", "task_state", "task_state_idx", "current_state", "task"],
)
def create_state_machine(state: State) -> Tuple[dict, State]:
    """
    Create a viable state machine for the task.
    Every task requires:
    * the robot and environment state
    * ensuring the robot has the skills to perform the required steps
    """
    task = state["task"]
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
        state_machine = [get_closest_state_from_skills(step, SKILLS) for step in plan]
        log.info(f"STATE_MACHINE:\n\n{state_machine}\n")

        ### Use symbolic logic to prune plan

        # Ensure that there are only rollout steps after an observation step until the next observation or navigation step
        observation_steps = ["scan the scene"]
        observation_step_idxs = [
            i for i, step in enumerate(state_machine) if step in observation_steps
        ]
        pick_and_place_steps = [
            "rollout pick and place plan",
            "pick object",
            "place in location",
        ]
        pick_and_place_step_idxs = [
            i for i, step in enumerate(state_machine) if step in pick_and_place_steps
        ]
        if len(observation_step_idxs) > 0 and len(pick_and_place_step_idxs) > 0:
            for i, observation_idx in enumerate(observation_step_idxs):
                pick_and_place_exists = False
                if observation_idx + 1 < len(state_machine):
                    while state_machine[observation_idx + 1] in pick_and_place_steps:
                        pick_and_place_exists = observation_idx + 1
                        state_machine.pop(observation_idx + 1)
                        task.pop(observation_idx + 1)
                        print(state_machine[observation_idx + 1 :])
                        if observation_idx + 1 >= len(state_machine):
                            break

                if pick_and_place_exists:
                    state_machine.insert(
                        pick_and_place_exists, "rollout pick and place plan"
                    )
                    task.insert(pick_and_place_exists, "rollout pick and place plan")

            log.info(f"UPDATED STATE_MACHINE (prune for rollout):\n\n{state_machine}\n")

        # Consolidate adjacent roll out steps
        rollout_steps = ["rollout pick and place plan"]
        rollout_step_idxs = [
            i for i, step in enumerate(state_machine) if step in rollout_steps
        ]
        if len(rollout_step_idxs) > 1:
            consolidated_state_machine = []
            for i, s in enumerate(state_machine):
                if s in rollout_steps:
                    if i > 0 and state_machine[i - 1] not in rollout_steps:
                        consolidated_state_machine.append("rollout pick and place plan")
                else:
                    consolidated_state_machine.append(s)
            state_machine = consolidated_state_machine

            log.info(
                f"UPDATED STATE_MACHINE (consolidate rollout steps):\n\n{state_machine}\n"
            )

        result = {
            "state_machine": state_machine,
            "task_state": "not_started",
            "current_state": "RUNNING",
        }
    result["task"] = task
    result["task_state_idx"] = 0

    log.info(f"Task: {task}")
    log.info(f"State machine: {state_machine}")
    output = "Here is the consolidated task:"
    output += "\n\n"
    output += "```\n"
    output += "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(task)])
    output += "\n```"
    output += "\n\n"
    output += "Here is the state machine:"
    output += "\n\n"
    output += "```\n"
    output += "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(state_machine)])
    output += "\n```"
    chat_item = {
        "content": output,
        "type": "text",
        "role": "assistant",
    }

    return result, state.append(chat_history=chat_item).update(**result)


@action(
    reads=["state_machine", "task_state_idx"],
    writes=["task_state", "task_state_idx", "current_state", "state_machine", "task"],
)
def execute_state_machine(state: State) -> Tuple[dict, State]:
    """
    State machine manages the execution of fully observable steps
    """
    task = state["task"]
    current_state = "RUNNING"
    state_machine = state["state_machine"]
    if state["task_state"] == "not_started":
        task_state = state["state_machine"][0]
        task_state_idx = state["task_state_idx"]
    else:
        task_state_idx = state["task_state_idx"] + 1
        if task_state_idx < len(state["state_machine"]):
            task_state = state["state_machine"][task_state_idx]
        else:
            task_state = "done"
            current_state = "DONE"

    result = {
        "task_state": task_state,
        "task_state_idx": task_state_idx,
        "current_state": current_state,
        "state_machine": state_machine,
        "task": task,
    }
    if task_state_idx < len(state_machine):
        content = f"Executing task: **{task[task_state_idx]}**\n\nTask state: `{task_state}`\n\nStep {task_state_idx+1} of {len(state_machine)}"
    else:
        content = f"Task completed: **{state['prompt']}**"
    chat_item = {
        "content": content,
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(
    reads=["state_machine", "task", "task_state", "task_state_idx"], writes=["location"]
)
def navigate_to_location(state: State) -> Tuple[dict, State]:
    step = state["task"][state["task_state_idx"]]
    extract_location = Task(
        f"""
        Given the following step, extract the location (e.g. kitchen), item (e.g. sink) or destination and return it. 
        Here is the string to extract the location: 
        {step}

        Examples:
        - Text: "navigate to the kitchen" Output: "kitchen"

        """,
        expected_output_format="A string representing the location, item or destination.",
    )
    extract_location_agent = Agent(
        name="Location Extractor",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds.
        """,
    )
    extract_location.add_solving_agent(extract_location_agent)
    output = extract_location.run()
    try:
        location = get_closest_text(output, list(SEMANTIC_LOCATIONS.keys()))
        if not navigate_to(
            SEMANTIC_LOCATIONS[location]["name"],
            SEMANTIC_LOCATIONS[location]["location"],
        ):
            raise Exception(f"Error navigating to location: {location}")

        if not wait_until_ready():
            raise Exception(f"Error navigating to location: {location}")
        content = f"Navigating to location: **{location}**"
    except Exception as e:
        log.error(f"Error: {e}")
        location = None
        content = f"{e}"

    result = {"location": location}
    chat_item = {
        "content": content,
        "type": "text" if location is not None else "error",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["state_machine"], writes=[])
def scan_the_scene(state: State) -> Tuple[dict, State]:
    result = {}
    chat_item = {
        "content": "Scanning the scene...",
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["state_machine"], writes=["image"])
def get_image(state: State) -> Tuple[dict, State]:
    try:
        image = get_image_from_sim()
        # image = Image.open("shared/data/test1.png")
        image = pil_to_b64(image)
        result = {"image": image}
        chat_item = {
            "content": image,
            "type": "image",
            "role": "assistant",
        }
    except Exception as e:
        log.error(f"Error getting image: {e}")
        result = {"image": None}
        chat_item = {
            "content": f"Error getting image: {e}",
            "type": "text",
            "role": "assistant",
        }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["image"], writes=["vla_response"])
def ask_vla(
    state: State, vla_prompt: str = "Describe the image."
) -> Tuple[dict, State]:
    image = b64_to_pil(state["image"])
    result = {"vla_response": moondream(image, vla_prompt)["result"]}

    chat_item = {
        "content": f"**Image Description:**:\n\n{result['vla_response']}",
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["vla_response"], writes=["observations"])
def get_list_of_objects(state: State) -> Tuple[dict, State]:
    task = Task(
        f"""Given the following, return a list assigned to `objects_on_table` of the objects on the table. The table is not an object. 
        Summary: 
        {state['vla_response']}
        
        Example:
            Summary:
                There is an object on the table called "Object 1", an object on the table called "Object 2", and an object on the table called "Object 3".
            Output:
                ```
                    objects_on_table = ["Object 1", "Object 2", "Object 3"]
                ```
        Don't use any functions, manually identify the objects on the table from the summary.
        """,
        expected_output_format="""
            ```
                objects_on_table = ["Object 1", "Object 2", "Object 3"]
            ```
            """,
    )
    analyzer_agent = Agent(
        name="Analyzer",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds with lists.
        """,
    )
    task.add_solving_agent(analyzer_agent)
    output = task.run()
    code = extract_code(output)
    try:
        exec_vars = {}
        exec_code(code, exec_vars)
        log.info(exec_vars.get("objects_on_table", None))
        objects_on_table = exec_vars.get("objects_on_table", None)
        observations = {"objects_on_table": objects_on_table}
    except Exception as e:
        log.error(f"Error executing code: {e}")
    result = {"observations": observations}
    chat_item = {
        "content": f"Objects on table: \n\n`{objects_on_table}`",
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(
    reads=["observations", "task_state_idx", "location"],
    writes=["state_machine", "task"],
)
def rollout_pick_and_place_plan(state: State) -> Tuple[dict, State]:
    task_idx = state["task_state_idx"]
    state_machine = state["state_machine"]
    task = state["task"]
    rollout_task = Task(
        f"""Rollout a pick and place plan for the robot given the following objects:
        {state['observations']}
        
        The robot and the objects are at {state['location']}
        
        Here are the locations the robot can go to: 
        {list(SEMANTIC_LOCATIONS.keys())}

        Here is additional knowledge you've learned:
        {knowledge_base.get_knowledge_as_string()}

        Here is an example:
        '{{'objects_on_table': ['cheese', 'milk', 'book']}}
        The robot and the objects are at {{'counter'}}
        Output:
        ```
            pick_and_place_tasks = ["Pick cheese at counter and place at kitchen", "Pick milk at counter and place at kitchen", "Pick book at counter and place at shelf"]
        ```
        Don't use any functions, manually synthesize the pick and place tasks from the summary.

        Here is additional knowledge you've learned, factor this into the plan:
        {knowledge_base.get_knowledge_as_string()}
        """,
        expected_output_format="""
        ```
            pick_and_place_tasks = ["Pick object1 at location and place at destination", "Pick object2 at location and place at destination", "Pick object3 at location and place at destination"]
        ```
        """,
    )
    planner_agent = Agent(
        name="Planner",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds with lists.
        """,
    )
    rollout_task.add_solving_agent(planner_agent)
    output = rollout_task.run()
    code = extract_code(output)
    try:
        exec_vars = {}
        exec_code(code, exec_vars)
        log.info(exec_vars.get("pick_and_place_tasks", None))
        pick_and_place_tasks = exec_vars.get("pick_and_place_tasks", None)
        task = task[: task_idx + 1] + pick_and_place_tasks + task[task_idx + 1 :]
        state_machine = (
            state_machine[: task_idx + 1]
            + ["pick_and_place"] * len(pick_and_place_tasks)
            + state_machine[task_idx + 1 :]
        )
    except Exception as e:
        log.error(f"Error executing code: {e}")
        raise NotImplementedError(
            "error handling for Rollout pick and place plan not implemented"
        )

    result = {"state_machine": state_machine, "task": task}

    log.info(f"Task: {task}")
    log.info(f"State machine: {state_machine}")
    output = "Here is the task:"
    output += "\n\n"
    output += "```\n"
    output += "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(task)])
    output += "\n```"
    output += "\n\n"
    output += "Here is the state machine:"
    output += "\n\n"
    output += "```\n"
    output += "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(state_machine)])
    output += "\n```"
    chat_item = {
        "content": output,
        "type": "text",
        "role": "assistant",
    }

    return result, state.append(chat_history=chat_item).update(**result)


@action(
    reads=["task", "state_machine", "task_state_idx", "location"],
    writes=["obj_to_grasp", "obj_location", "obj_destination"],
)
def pick_and_place(state: State) -> Tuple[dict, State]:
    get_object = Task(
        f"""Given the following, extract the object of interest as a string assigned to `obj_to_grasp`. 
        Here is the string to extract the object:
        {state["task"][state["task_state_idx"]]}

        Here is an example:
            Here is the string to extract the object:
                Pick cheese at counter and place in kitchen
            Output:
            ```
                obj_to_grasp = "cheese"
            ```
        Don't use any functions. Manually identify the object from the summary.
        """,
        expected_output_format="""
        ```
            obj_to_grasp = "object1"
        ```
        """,
    )
    analyzer_agent = Agent(
        name="Analyzer",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds with variables.
        """,
    )
    get_object.add_solving_agent(analyzer_agent)
    output = get_object.run()
    code = extract_code(output)
    try:
        exec_vars = {}
        exec_code(code, exec_vars)
        obj_to_grasp = exec_vars.get("obj_to_grasp", None)
    except Exception as e:
        log.error(f"Error executing code: {e}")

    # Assume object location is current unless previously stored
    location = state["location"]
    if "obj_location" in state:
        location = state["obj_location"]

    get_obj_destination = Task(
        f"""Given the following, extract the destination as a string assigned to `obj_destination`.
        Here is the string to extract the destination:
        {state["task"][state["task_state_idx"]]}
        
        Example:
            Here is the string to extract the destination:
                Pick cheese at counter and place in kitchen
            Output: 'kitchen'
         """,
        expected_output_format="String representing the destination.",
    )
    get_obj_destination_agent = Agent(
        name="Destination Extractor",
        model=DEFAULT_MODEL,
        system_message="""
        You are a helpful agent that concisely responds with variables.
        """,
    )
    get_obj_destination.add_solving_agent(get_obj_destination_agent)
    output = get_obj_destination.run()
    destination = get_closest_text(output, list(SEMANTIC_LOCATIONS.keys()))

    result = {
        "obj_to_grasp": obj_to_grasp,
        "obj_location": location,
        "obj_destination": destination,
    }
    chat_item = {
        "content": f"Pick and place **{obj_to_grasp}** at **{location}**",
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["obj_to_grasp", "obj_location"], writes=["location"])
def navigate_for_pick(state: State) -> Tuple[dict, State]:
    location = state["obj_location"]
    obj_to_grasp = state["obj_to_grasp"]
    location = get_closest_text(location, list(SEMANTIC_LOCATIONS.keys()))
    log.info(f"Pick and place {obj_to_grasp} at {location}")
    try:
        if state["location"] != location:
            log.info(f"Changing location from {state['location']} to {location}")
            if not navigate_to(
                SEMANTIC_LOCATIONS[location]["name"],
                SEMANTIC_LOCATIONS[location]["location"],
            ):
                raise Exception(f"Error navigating to location: {location}")

            if not wait_until_ready():
                raise Exception(f"Error navigating to location: {location}")
        content = f"Navigated to **{location}** to pick **{obj_to_grasp}**"
    except Exception as e:
        log.error(f"Error navigating to location: {e}")
        location = None
        content = f"{e}"

    result = {"location": location}

    chat_item = {
        "content": content,
        "type": "text" if location is not None else "error",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["obj_to_grasp", "obj_location"], writes=["obj_in_hand", "obj_to_grasp"])
def pick_object(state: State) -> Tuple[dict, State]:
    PICK_TIMEOUT = 30.0
    obj_to_grasp = state["obj_to_grasp"]
    print(f"Pick {obj_to_grasp}")

    try:
        if not pick(obj_to_grasp):
            raise Exception(f"Error picking object: {obj_to_grasp}")

        pick_start_time = time.time()
        obj_in_hand = None
        while not obj_in_hand and time.time() - pick_start_time < PICK_TIMEOUT:
            obj_in_hand = get_obj_in_hand()
            time.sleep(0.1)

        if obj_in_hand:
            print(f"Object in hand: {obj_in_hand}")
            obj_to_grasp = None

        if not wait_until_ready():
            raise Exception(f"Error picking object: {obj_to_grasp}")
        content = f"Picked **{obj_in_hand}**"
    except Exception as e:
        log.error(f"Error picking object: {e}")
        obj_in_hand = None
        content = f"{e}"

    result = {"obj_in_hand": obj_in_hand, "obj_to_grasp": obj_to_grasp}

    chat_item = {
        "content": content,
        "type": "text" if obj_in_hand else "error",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["obj_destination", "location"], writes=["location"])
def navigate_for_place(state: State) -> Tuple[dict, State]:
    destination = state["obj_destination"]
    try:
        if state["location"] != destination:
            log.info(f"Changing location from {state['location']} to {destination}")
            if not navigate_to(
                SEMANTIC_LOCATIONS[destination]["name"],
                SEMANTIC_LOCATIONS[destination]["location"],
            ):
                raise Exception(f"Error navigating to destination: {destination}")

            if not wait_until_ready():
                raise Exception(f"Error navigating to destination: {destination}")
        content = f"Navigated to **{destination}** to place **{state['obj_in_hand']}**"
        location = destination
    except Exception as e:
        log.error(f"Error navigating to destination: {e}")
        location = None
        content = f"{e}"

    result = {"location": location}

    chat_item = {
        "content": content,
        "type": "text",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


@action(reads=["obj_in_hand"], writes=["obj_in_hand"])
def place_object(state: State) -> Tuple[dict, State]:
    obj_to_place = state["obj_in_hand"]
    try:
        if not place(SEMANTIC_LOCATIONS[state["location"]]["name"]):
            raise Exception(f"Error placing object: {obj_to_place}")

        if not wait_until_ready():
            raise Exception(f"Error placing object: {obj_to_place}")
        obj_in_hand = None
        content = f"Placed object **{obj_to_place}**"
    except Exception as e:
        log.error(f"Error placing object: {e}")
        obj_in_hand = obj_to_place
        content = f"{e}"

    result = {"obj_in_hand": obj_in_hand}

    chat_item = {
        "content": content,
        "type": "text" if obj_in_hand is None else "error",
        "role": "assistant",
    }
    return result, state.append(chat_history=chat_item).update(**result)


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


@action(
    reads=["current_state"],
    writes=["response", "current_state", "code_attempts"],
)
def create_error_response(state: State) -> Tuple[dict, State]:
    content = "Could not complete the task."
    if "task_state" in state:
        content += f" I have failed on {state['task_state']}."
    result = {
        "response": {
            "content": content,
            "type": "error",
            "role": "assistant",
        },
        "current_state": "FAILED",
        "code_attempts": 0,
    }
    return result, state.append(chat_history=result["response"]).update(**result)


@action(reads=["current_state"], writes=["response", "current_state"])
def finish_and_score_task(state: State) -> Tuple[dict, State]:
    response = {
        "content": "I'm done. Goodbye!",
        "type": "text",
        "role": "assistant",
    }
    current_state = "PENDING"
    result = {"response": response, "current_state": current_state}
    return result, state.append(chat_history=result["response"]).update(**result)


@action(reads=["current_state"], writes=["chat_history", "current_state"])
def response(state: State) -> Tuple[dict, State]:
    if state["current_state"] == "DONE":
        current_state = "PENDING"
        response = {
            "content": "**Task execution successful :+1:**",
            "type": "text",
            "role": "assistant",
        }
    elif state["current_state"] == "FAILED":
        current_state = "FAILED"
        response = {
            "content": "**Task execution failed**\nLet me know if you'd like me to retry or just give me a new task.",
            "type": "error",
            "role": "assistant",
        }
    else:
        current_state = state["current_state"]
        response = {
            "content": "Ready for next prompt.",
            "type": "text",
            "role": "assistant",
        }
    result = {"chat_item": response, "current_state": current_state}
    return result, state.append(chat_history=response).update(**result)


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
            parse_prompt=parse_prompt,
            respond_to_question=respond_to_question,
            update_knowledge_base=update_knowledge_base,
            determine_if_task_in_skill_library=determine_if_task_in_skill_library,
            create_plan_for_unknown_task=create_plan_for_unknown_task,
            get_closest_plans=get_closest_plans,
            get_role_and_location_context=get_role_and_location_context,
            create_initial_plan=create_initial_plan,
            create_robot_grounded_plan=create_robot_grounded_plan,
            determine_if_plan_is_feasibile=determine_if_plan_is_feasibile,
            convert_plan_to_steps=convert_plan_to_steps,
            create_state_machine=create_state_machine,
            execute_state_machine=execute_state_machine,
            navigate_to_location=navigate_to_location,
            scan_the_scene=scan_the_scene,
            get_image=get_image,
            ask_vla=ask_vla,
            get_list_of_objects=get_list_of_objects,
            rollout_pick_and_place_plan=rollout_pick_and_place_plan,
            pick_and_place=pick_and_place,
            navigate_for_pick=navigate_for_pick,
            pick_object=pick_object,
            navigate_for_place=navigate_for_place,
            place_object=place_object,
            finish_and_score_task=finish_and_score_task,
            create_error_response=create_error_response,
            prompt_for_more=prompt_for_more,
            response=response,
        )
        .with_transitions(
            ("prompt", "parse_prompt", default),
            (
                "parse_prompt",
                "determine_if_task_in_skill_library",
                when(prompt_cls=PromptType.PERFORM_NEW_TASK),
            ),
            (
                "parse_prompt",
                "respond_to_question",
                when(prompt_cls=PromptType.RESPOND_TO_QUESTION),
            ),
            (
                "parse_prompt",
                "update_knowledge_base",
                when(prompt_cls=PromptType.UPDATE_KNOWLEDGE_BASE),
            ),
            (
                "parse_prompt",
                "execute_state_machine",
                when(prompt_cls=PromptType.RETRY_EXISTING_TASK),
            ),
            ("parse_prompt", "response", default),
            ("respond_to_question", "response", default),
            ("update_knowledge_base", "response", default),
            (
                "determine_if_task_in_skill_library",
                "create_plan_for_unknown_task",
                when(task="unknown"),
            ),
            ("create_plan_for_unknown_task", "get_closest_plans", default),
            ("get_closest_plans", "get_role_and_location_context", default),
            ("get_role_and_location_context", "create_initial_plan", default),
            ("create_initial_plan", "create_robot_grounded_plan", default),
            ("create_robot_grounded_plan", "determine_if_plan_is_feasibile", default),
            (
                "determine_if_plan_is_feasibile",
                "create_error_response",
                when(feasible=False),
            ),
            ("determine_if_plan_is_feasibile", "convert_plan_to_steps", default),
            (
                "convert_plan_to_steps",
                "create_error_response",
                when(task="unknown"),
            ),
            ("convert_plan_to_steps", "create_state_machine", default),
            ("determine_if_task_in_skill_library", "create_state_machine", default),
            ("create_state_machine", "execute_state_machine", default),
            ("create_state_machine", "prompt_for_more", when(state_machine="unknown")),
            (
                "execute_state_machine",
                "navigate_to_location",
                when(task_state="navigate to location"),
            ),
            (
                "execute_state_machine",
                "scan_the_scene",
                when(task_state="scan the scene"),
            ),
            (
                "execute_state_machine",
                "rollout_pick_and_place_plan",
                when(task_state="rollout pick and place plan"),
            ),
            ("rollout_pick_and_place_plan", "execute_state_machine", default),
            (
                "execute_state_machine",
                "pick_and_place",
                when(task_state="pick_and_place"),
            ),
            ("pick_and_place", "navigate_for_pick", default),
            ("navigate_for_pick", "create_error_response", when(location=None)),
            ("navigate_for_pick", "pick_object", default),
            ("pick_object", "create_error_response", when(obj_in_hand=None)),
            ("pick_object", "navigate_for_place", default),
            ("navigate_for_place", "place_object", default),
            ("place_object", "execute_state_machine", default),
            ("navigate_to_location", "create_error_response", when(location=None)),
            ("navigate_to_location", "execute_state_machine", default),
            ("scan_the_scene", "get_image", default),
            ("get_image", "create_error_response", when(image=None)),
            ("get_image", "ask_vla", default),
            ("ask_vla", "get_list_of_objects", default),
            ("get_list_of_objects", "execute_state_machine", default),
            ("execute_state_machine", "finish_and_score_task", when(task_state="done")),
            ("finish_and_score_task", "prompt", default),
            ("response", "prompt", when(current_state="PENDING")),
            ("response", "execute_state_machine", when(current_state="RUNNING")),
            ("prompt_for_more", "response", default),
            ("create_error_response", "response", default),
            ("response", "prompt", when(current_state="FAILED")),
        )
        # initializes from the tracking log if it does not already exist
        .initialize_from(
            tracker,
            resume_at_next_action=True,  # always resume from entrypoint in the case of failure
            default_state={"chat_history": [], "current_state": "PENDING"},
            default_entrypoint="prompt",
            # fork_from_app_id="670b9f83-d0fa-49ce-b396-dcaba416edc8",
            # fork_from_sequence_id=55,
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
