from crewai import Crew, Process, Agent, Task
from crewai_tools import tool


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

@tool("Get objects on the table.")
def get_objects_on_the_table() -> str:
    """Get objects on the table"""
    return "Milk, Cereal" # string to be sent back to the agent

# Define your agents
planner = Agent(
  role='Planner',
  goal='Create plans for robots.',
  backstory='An experienced planner that breaks down tasks into steps for robots.',
  tools = [],
  verbose=True,
  allow_delegation=False
)
analyst = Agent(
  role='Scene Analyzer',
  goal='Identify objects in the scene.',
  backstory='An experienced analyst that can identify objects in a scene.',
  tools = [get_objects_on_the_table],
  verbose=True,
  allow_delegation=False
)


# Define the tasks in sequence
planner_task = Task(description='Create a plan for a robot to clear the table.',
                    agent=planner, 
                    expected_output="List of steps for a robot.")
analysis_task = Task(description='List the objects that are on the table', 
                     agent=analyst, 
                     expected_output="List of objects.")

# Form the crew with a sequential process
crew = Crew(
  agents=[planner, analyst],
  tasks=[analysis_task, planner_task],
  process=Process.sequential,
  verbose=2
)
crew.kickoff()  