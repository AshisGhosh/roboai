import langroid as lr
import langroid.language_models as lm

# set up LLM
llm_cfg = lm.OpenAIGPTConfig( # or OpenAIAssistant to use Assistant API 
  # any model served via an OpenAI-compatible API
#   chat_model="litellm/openrouter/mistralai/mistral-7b-instruct:free"
  chat_model="litellm/openrouter/huggingfaceh4/zephyr-7b-beta:free"
)
# # use LLM directly
# mdl = lm.OpenAIGPT(llm_cfg)
# response = mdl.chat("What is the capital of Ontario?", max_tokens=10)

# # use LLM in an Agent
# agent_cfg = lr.ChatAgentConfig(llm=llm_cfg)
# agent = lr.ChatAgent(agent_cfg)
# agent.llm_response("What is the capital of China?") 
# response = agent.llm_response("And India?") # maintains conversation state 

# wrap Agent in a Task to run interactive loop with user (or other agents)
# task = lr.Task(agent, name="Bot", system_message="You are a helpful assistant")
# task.run("Hello") # kick off with user saying "Hello"

# 2-Agent chat loop: Teacher Agent asks questions to Student Agent
agent_cfg = lr.ChatAgentConfig(llm=llm_cfg)
robot_agent = lr.ChatAgent(agent_cfg)

robot_task = lr.Task(
    robot_agent, name="Robot",
    system_message="""
        You are a robot and have a high level task. 
        You must ask the planner to break it down into steps you can do.
        Your skills involve 'pick' and 'place' actions.  
        """,
    # done_if_response=[Entity.LLM],
    interactive=False,
)
planner_agent = lr.ChatAgent(agent_cfg)
planner_task = lr.Task(
    planner_agent, name="Planner",
    system_message="""
    Concisely return numbered steps of a plan for a robot.
    The plan can only involve 'pick' and 'place' actions.
    If the plan is valid, respond with 'DONE'.
    """,
    single_round=True,
    interactive=False,
)

robot_task.add_sub_task(planner_task)
robot_task.run("The task is to clear the table, it has the following objects: 'Milk', 'Cereal', and a 'Can'.")