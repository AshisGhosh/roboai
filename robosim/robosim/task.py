from enum import Enum

import logging
logging.basicConfig(level=logging.INFO)

class TaskClass(Enum):
    CONTROL_TASK = 0
    DATA_TASK = 1
    ASYNC_DATA_TASK = 2

class TaskStatus(Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3

class TaskFactory:
    def __init__(self):
        self._creators = {}

    def register_task(self, creator, task_class=TaskClass.CONTROL_TASK):
        self.register_task_type(
            creator.__name__,
            lambda name, *args, **kwargs: Task(name, creator, task_class, *args, **kwargs)
        )

    def register_task_type(self, task_type, creator):
        self._creators[task_type] = creator

    def create_task(self, task_type, task_name=None, *args, **kwargs):
        creator = self._creators.get(task_type)
        if not creator:
            raise ValueError(f"Task type {task_type} not registered.")
        if task_name is not None:
            # Use the provided task_name or fallback to a default naming convention
            return creator(task_name, *args, **kwargs)
        else:
            return creator(task_type, *args, **kwargs)
    
    def get_task_types(self):
        return self._creators.keys()
    

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
            logging.error(f"Error executing task {self.name}: {e}")
    
    def __str__(self):
        return f"Task: {self.name}\n    Function: {self.function}\n    Args: {self.args}\n    Kwargs: {self.kwargs}"