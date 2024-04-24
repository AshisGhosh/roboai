import carb
import numpy as np
from enum import Enum

from roboai.robot import RobotStatus 

class TaskClass(Enum):
    CONTROL_TASK = 0
    DATA_TASK = 1
    ASYNC_DATA_TASK = 2

class TaskManagerStatus(Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3

class Task:
    def __init__(self, name, function, task_class, *args, **kwargs):
        self.name = name
        self.function = function
        self.task_class = task_class
        self.args = args
        self.kwargs = kwargs

    def execute(self):
        try:
            return self.function(*self.args, **self.kwargs)
        except Exception as e:
            carb.log_error(f"Error executing task {self.name}: {e}")
    
    def __str__(self):
        return f"Task: {self.name}\n    Function: {self.function}\n    Args: {self.args}\n    Kwargs: {self.kwargs}"

class TaskManager:
    def __init__(self, robot_actor):
        self.robot_actor = robot_actor
        self.planner = Planner(robot_actor)
        self.tasks = []
        self._current_task = None
        self.status = TaskManagerStatus.PENDING
    
    def do_tasks(self):        
        if self._current_task is None:
            if len(self.tasks) > 0:
                self._current_task = self.tasks.pop(0)
                carb.log_warn(f"Executing task {self._current_task.name}")
                self.status = TaskManagerStatus.RUNNING
            else:
                self.status = TaskManagerStatus.COMPLETED
                return self.status
        
        status = self._current_task.execute() 
        if status == RobotStatus.COMPLETED:
            self._current_task = None
            self.status = TaskManagerStatus.PENDING
            if len(self.tasks) == 0:
                carb.log_warn("All tasks completed")
                self.status = TaskManagerStatus.COMPLETED
        elif status == RobotStatus.FAILED:
            carb.log_error(f"Task {self._current_task.name} failed")
            self._current_task = None
            self.status = TaskManagerStatus.FAILED
        
        return self.status

    def add_task(self, task):
        self.tasks.append(task)      
    
    def test_task(self):
        self.add_task(
            Task(
                name="Test Task",
                function=self.robot_actor.move_pos,
                task_class=TaskClass.CONTROL_TASK,
                pos=np.array([0.3, 0.3, 0.3])
            )
        )
