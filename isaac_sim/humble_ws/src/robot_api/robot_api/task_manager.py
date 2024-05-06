from enum import Enum
from collections import deque
from uuid import uuid4
import threading
import asyncio
from pathlib import Path
from nicegui import Client, app, ui, ui_run

# generic ros libraries
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from roboai_interfaces.action import MoveArm


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ABORTED = "ABORTED"
    PAUSED = "PAUSED"


class Task:
    def __init__(self, name, logger=None) -> None:
        self.name = name
        self.status = TaskStatus.PENDING
        self.uuid = uuid4()
        self.logger = logger
        self.result = None
        self.log(f"TASK ({self.uuid}): {self.name} created; Status: {self.status}")

    def run(self) -> None:
        pass

    def abort(self) -> None:
        self.status = TaskStatus.ABORTED

    def update_status(self, status) -> None:
        self.status = status
        self.log(f"TASK ({self.uuid}): {self.name} updated to: {status}")

    def log(self, message) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def __str__(self) -> str:
        return f"{self.name}; Status: {self.status}"


class MoveArmTask(Task):
    def __init__(self, name, goal: str | list[float], logger=None) -> None:
        super().__init__(name, logger)
        self.goal = goal
        self._action_client = None
        self.goal_handle = None

    def run(self, action_client: ActionClient) -> None:
        self._action_client = action_client
        self.log(f"Moving arm to goal: {self.goal}")
        self.update_status(TaskStatus.RUNNING)
        try:
            self.send_goal()
        except Exception as e:
            self.log(f"Error while sending goal: {e}")
            self.update_status(TaskStatus.FAILURE)

    def abort(self) -> None:
        if self.goal_handle:
            self.goal_handle.cancel_goal()
            self._action_client._cancel_goal(self.goal_handle)
        super().abort()

    def send_goal(self) -> None:
        goal_msg = MoveArm.Goal()
        if isinstance(self.goal, str):
            goal_msg.configuration_goal = self.goal
        elif isinstance(self.goal, list):
            if len(self.goal) == 6:
                goal_msg.joint_goal = self.goal
            elif len(self.goal) == 7:
                goal_msg.cartesian_goal = self.goal
            else:
                raise ValueError(f"Invalid goal: {self.goal}")
        else:
            raise ValueError(f"Invalid goal: {self.goal}")

        self._action_client.wait_for_server(timeout_sec=1)
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.log("Goal rejected :(")
            self.update_status(TaskStatus.FAILURE)
            return

        self.log("Goal accepted :)")
        # Wait for the result
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future) -> None:
        self.result = future.result().result
        print(f"Result received: {self.result}, {type(self.result.status)}")
        if self.result.status == "SUCCEEDED":
            self.log(f"Result received: {self.result.message}")
            self.update_status(TaskStatus.SUCCESS)
        else:
            self.log(f"Action did not succeed with status: {self.result.status}")
            self.update_status(TaskStatus.FAILURE)


class TaskManager(Node):
    def __init__(self) -> None:
        super().__init__("task_manager")
        self.tasks = deque()
        self.current_task = None
        self.task_history = []
        self.move_arm_action_client = ActionClient(self, MoveArm, "/move_arm")
        self.setup_gui()

        self.get_logger().info("Task Manager initialized")

        self.add_task(MoveArmTask("Move to extended", "extended"))
        self.add_task(MoveArmTask("Move to ready", "ready"))

    def setup_gui(self) -> None:
        with Client.auto_index_client:
            ui.label("Task Manager").style("font-size: 24px")
            self.grid = ui.aggrid(
                {
                    "domLayout": "autoHeight",
                    "defaultColDef": {"flex": 1},
                    "columnDefs": [
                        {"headerName": "Name", "field": "name", "sortable": True},
                        {"headerName": "Status", "field": "status", "sortable": True},
                    ],
                    "rowData": [],
                }
            )
            self.update_grid()

            ui.button("Add Task", on_click=self.add_task)
            ui.button("Run Tasks", on_click=lambda: asyncio.create_task(self.run_tasks()))
            ui.button("Abort Current Task", on_click=lambda: asyncio.create_task(self.abort_current_task()))
            ui.button("Clear Tasks", on_click=self.clear_tasks)

    def update_grid(self) -> None:
        task_dict = [
            {"name": task.name, "status": task.status} for task in self.task_history
        ] + [{"name": task.name, "status": task.status} for task in self.tasks]

        self.grid.options["rowData"] = task_dict
        self.get_logger().info(f"{task_dict}")
        self.grid.update()

    def add_task(self, task: Task) -> None:
        task.logger = self.get_logger()
        self.tasks.append(task)
        self.update_grid()
        self.get_logger().info(f"Task added: {task}")

    def remove_task(self, task: Task) -> None:
        self.tasks.remove(task)
        self.update_grid()
        self.get_logger().info(f"Task removed: {task}")

    def clear_tasks(self) -> None:
        self.tasks.clear()
        self.update_grid()
        self.get_logger().info("Tasks cleared")

    async def run_tasks(self) -> None:
        while self.tasks:
            self.set_current_task(self.tasks.popleft())

            if isinstance(self.current_task, MoveArmTask):
                self.current_task.run(self.move_arm_action_client)
            else:
                self.current_task.run()

            self.update_grid()

            while self.current_task.status == TaskStatus.RUNNING:
                await asyncio.sleep(0.1)
                if self.current_task.status == TaskStatus.ABORTED:
                    break

        self.set_current_task(None)
        self.get_logger().info("All tasks completed")

    def set_current_task(self, task: Task | None) -> None:
        self.current_task = task
        if task:
            self.task_history.append(task)
            self.get_logger().info(f"Current task: {task}")
        self.update_grid()

    async def abort_current_task(self) -> None:
        if self.current_task:
            self.current_task.abort()
            self.get_logger().info(f"Current task aborted: {self.current_task}")
        else:
            self.get_logger().info("No current task to abort")

    def get_tasks(self) -> deque[Task]:
        return self.tasks

    def destroy_node(self) -> None:
        self.move_arm_action_client.destroy()
        self.abort_current_task()
        super().destroy_node()


def main() -> None:
    # NOTE: This function is defined as the ROS entry point in setup.py, but it's empty to enable NiceGUI auto-reloading
    # https://github.com/zauberzeug/nicegui/blob/main/examples/ros2/ros2_ws/src/gui/gui/node.py
    pass


def ros_main() -> None:
    rclpy.init()
    task_manager = TaskManager()
    executor = MultiThreadedExecutor()
    executor.add_node(task_manager)
    try:
        executor.spin()
    finally:
        task_manager.destroy_node()
        rclpy.shutdown()


app.on_startup(lambda: threading.Thread(target=ros_main).start())
ui_run.APP_IMPORT_STRING = f"{__name__}:app"  # ROS2 uses a non-standard module name, so we need to specify it here
ui.run(uvicorn_reload_dirs=str(Path(__file__).parent.resolve()), favicon="🤖")
