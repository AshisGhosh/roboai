import io
import base64
from typing import List, Optional, Tuple
from PIL import Image

from burr.core import Application, ApplicationBuilder, State, default, when
from burr.core.action import action
from burr.lifecycle import LifecycleAdapter
from burr.tracking import LocalTrackingClient

from shared.utils.llm_utils import get_closest_text_sync as get_closest_text
from shared.utils.isaacsim_client import get_image as get_image_from_sim, pick, place
from shared.utils.image_utils import pil_to_b64, b64_to_pil
from shared.utils.gradio_client import moondream_answer_question_from_image as moondream

from task import Task
from agent import Agent

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


MODES = {
    "answer_question": "text",
    "generate_image": "image",
    "generate_code": "code",
    "unknown": "text",
}


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
        state["prompt"], ["What is on the table?", "Clear the table"]
    )
    if closest_text:
        result = {"task": closest_text}
    else:
        result = {"task": "unknown"}

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


@action(reads=["task"], writes=["state_machine", "task_state", "current_state"])
def create_state_machine(state: State) -> Tuple[dict, State]:
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
    else:
        result = {
            "state_machine": "unknown",
            "task_state": "unknown",
            "current_state": "PENDING",
        }
    return result, state.update(**result)


@action(reads=["state_machine"], writes=["task_state", "current_state"])
def execute_state_machine(state: State) -> Tuple[dict, State]:
    current_state = "RUNNING"
    if state["task_state"] == "not_started":
        task_state = state["state_machine"][0]
    else:
        task_state_idx = state["state_machine"].index(state["task_state"])
        if task_state_idx < len(state["state_machine"]) - 1:
            task_state = state["state_machine"][task_state_idx + 1]
        else:
            task_state = "done"
            current_state = "DONE"
    result = {"task_state": task_state, "current_state": current_state}
    return result, state.update(**result)


@action(reads=["state_machine"], writes=["image"])
def get_image(state: State) -> Tuple[dict, State]:
    # image = get_image_from_sim()
    image = Image.open("shared/data/test1.png")
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
        f"Given the following summary, return just a list in python of the objects on the table. The table is not an object. Summary: \n{state['vla_response']}",
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
        skills = ["pick", "place", "navigate to toilet", "navigate to kitchen", "call support"]
        # relevant_skills = get_closest_text(state["task"], skills, k=2)
        # relevant_skills = ["pick", "place"]
        relevant_skills = None
        plan_prompt = f"""Create a plan for a robot to {state["task"]}:
            {state["relevant_vars"]}
            Do not add any extra steps.
        """
        if relevant_skills:
            plan_prompt += f"You can only use the following actions: {relevant_skills}"
    else:
        plan_prompt = "Unknown task"
    result = {"plan_prompt": plan_prompt}
    return result, state.update(**result)


@action(reads=["plan_prompt"], writes=["plan"])
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
        example='"place_success = place(object_name)" --> Returns: True ',
    )
    coder_agent = Agent(
        name="Coder",
        model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
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
        exec_vars = {
            k: globals().get(v, None) for k, v in exec_vars.items()
        }
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
        f"Given the following code and error, fix the code. Code: \n{state['code']} Error: \n{state['execution']}"
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
    code = extract_code(output)
    result = {"code": code, "code_attempts": state.get("code_attempts", 0) + 1}
    return result, state.update(**result)


@action(reads=["code"], writes=["execution"])
def execute_code(state: State) -> Tuple[dict, State]:
    result = {"execution": "HI IM AN EXECUTION"}
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
    if state["task_state"] == "get_image":
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
    # app_id="89dbcceb-05e2-4ab3-afee-2d5f2d627177"
    # sequence_id=20
    # persisted_state = tracker.load(partition_key=None,
    #                                app_id=app_id, 
    #                            )
    # state_values = persisted_state['state'].get_all()
    return (
        ApplicationBuilder()
        .with_actions(
            prompt=process_prompt,
            determine_task=determine_task,
            confirm_task=confirm_task,
            create_state_machine=create_state_machine,
            execute_state_machine=execute_state_machine,
            get_image=get_image,
            ask_vla=ask_vla,
            get_list_of_objects=get_list_of_objects,
            create_plan_prompt=create_plan_prompt,
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
            ("determine_task", "prompt_for_more", when(task="unknown")),
            ("determine_task", "confirm_task", default),
            ("confirm_task", "response", default),
            ("create_state_machine", "execute_state_machine", default),
            ("create_state_machine", "prompt_for_more", when(state_machine="unknown")),
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
            ("create_plan_prompt", "create_plan", default),
            ("execute_state_machine", "code", when(task_state="code")),
            (
                "execute_state_machine",
                "execute_code",
                when(task_state="execute_code"),
            ),
            ("execute_state_machine", "create_response", when(task_state="done")),
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
        )
        .with_hooks(*hooks)
        .with_tracker(tracker)
        .with_identifiers(app_id=app_id)
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
