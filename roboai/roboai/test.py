import logging

from litellm import completion

from roboai.agent import Agent
from roboai.task import Task

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

logging.basicConfig(level=logging.WARN)

log = logging.getLogger("roboai")
log.setLevel(logging.DEBUG)

# litellm.success_callback = ["langfuse"]

# litellm.set_verbose=True


def test_task():
    planner_agent = Agent(
        name="Planner",
        model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
        system_message="""You are a planner that breaks down tasks into steps for robots.
                            Create a set of steps that a robot with wheels and one arm can do.
                            """,
    )
    # task_handler = Agent(
    #                     name="Task Handler",
    #                     model="openrouter/huggingfaceh4/zephyr-7b-beta:free",
    #                     system_message="""
    #                         You are a task handler that can handle tasks for robots.
    #                         """
    #                     )

    task_handler = "Create a plan to clear the table"
    task = Task(task_handler, [planner_agent])
    task.run()


def test():
    messages = [{"content": "Hello, how are you?", "role": "user"}]
    response = completion(
    model="openrouter/microsoft/phi-3-medium-128k-instruct:free", messages=messages
    )
    print(response)


def test_agent():
    agent = Agent(name="test", model="openrouter/huggingfaceh4/zephyr-7b-beta:free")
    response = agent.chat("Hello, how are you?")
    print(response)
    print(agent.get_last_response())
    print(agent.get_last_response_obj())
    agent.clear_messages()
    print(agent.messages)
    response = agent.chat("What is the capital of China?")
    print(response)
    print(agent.get_last_response())
    print(agent.get_last_response_obj())
    agent.clear_messages()
    print(agent.messages)
    response = agent.chat("And India?")
    print(response)
    print(agent.get_last_response())
    print(agent.get_last_response_obj())
    agent.clear_messages()
    print(agent.messages)


if __name__ == "__main__":
    test()
    # test_agent()
    # test_task()
