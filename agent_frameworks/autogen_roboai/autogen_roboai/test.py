import autogen
from autogen import AssistantAgent, UserProxyAgent

import tempfile
from autogen.coding import LocalCommandLineCodeExecutor

from typing_extensions import Annotated


import logging

logging.basicConfig(level=logging.INFO)

filter_dict = {"tags": ["zephyr"]}
config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict
)
assert len(config_list) == 1

llm_config = {
    "config_list": config_list,
    "timeout": 120,
}

task = "Create a list of steps for a robot to clear the table."

# create an AssistantAgent instance named "assistant" with the LLM configuration.
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="""
        You are a helpful assistant who can break down tasks into steps. 
        Please help the user with their task.
        Use the functions provided to learn more about the task.
        Respond with 'TERMINATE' when you are done.
        """,
)

# Create a temporary directory to store the code files.
temp_dir = tempfile.TemporaryDirectory()

# Create a local command line code executor.
executor = LocalCommandLineCodeExecutor(
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    system_message="A proxy for the user for executing code.",
    code_execution_config={"executor": executor},
    is_termination_msg=lambda x: "content" in x
    and x["content"] is not None
    and "TERMINATE" in x["content"]
    and "``" not in x["content"],
)


@user_proxy.register_for_execution()
@assistant.register_for_llm(
    name="identify_objs_on_table",
    description="Python function to get a list of objects on the table.",
)
def identify_objs_on_table(
    message: Annotated[
        str, "Message to ask the inspector for the objects on the table."
    ],
) -> str:
    logging.info("Asked for objects.")
    return "Milk, Cereal, a Can."


# inspector = AssistantAgent(
#     name="inspector",
#     llm_config=llm_config,
#     system_message="You are an inspector who can identify objects in a scene. There is 'Milk', 'Cereal' and a 'Can' on the table. Please respond with 'TERMINATE' when you are done."
# )

# user_inspector = UserProxyAgent(
#     name="user_inspector",
#     human_input_mode="NEVER",
#     is_termination_msg=lambda x: "content" in x
#         and x["content"] is not None
#         and "TERMINATE" in x["content"]
# )

# @user_inspector.register_for_execution()
# @inspector.register_for_llm(
#     name="identify_objects",
#     description="Identify objects in the scene.",
# )
# def identify_objects(message: Annotated[str, "Message to identify objects in the scene."]):
#     return "Milk, Cereal, a Can."

user_proxy.initiate_chat(assistant, message=task)
# logging.info(f"Chat result: {chat_result}")
