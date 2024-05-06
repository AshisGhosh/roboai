from enum import Enum
from collections import deque
from uuid import uuid4

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
    def __init__(self, name, logger=None):
        self.name = name
        self.status = TaskStatus.PENDING
        self.uuid = uuid4()
        self.logger = logger
        self.result = None
        self.log(f"TASK ({self.uuid}): {self.name} created; Status: {self.status}")

    def run(self):
        pass

    def abort(self):
        self.status = TaskStatus.ABORTED

    def update_status(self, status):
        self.status = status
        self.log(f"TASK ({self.uuid}): {self.name} updated to: {status}")

    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def __str__(self):
        return f"{self.name}; Status: {self.status}"


class MoveArmTask(Task):
    def __init__(self, name, goal: str | list[float], logger=None):
        super().__init__(name, logger)
        self.goal = goal
        self._action_client = None
        self.goal_handle = None

    def run(self, action_client: ActionClient):
        self._action_client = action_client
        self.log(f"Moving arm to goal: {self.goal}")
        self.status = TaskStatus.RUNNING
        try:
            self.send_goal()
        except Exception as e:
            self.log(f"Error while sending goal: {e}")
            self.update_status(TaskStatus.FAILURE)

    def abort(self):
        if self.goal_handle:
            self.goal_handle.cancel_goal()
        # self._action_client.cancel_goal()
        super().abort()

    def send_goal(self):
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

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.log("Goal rejected :(")
            self.update_status(TaskStatus.FAILURE)
            return

        self.log("Goal accepted :)")
        # Wait for the result
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.result = future.result().result
        print(f"Result received: {self.result}, {type(self.result.status)}")
        if self.result.status == "SUCCEEDED":
            self.log(f"Result received: {self.result.message}")
            self.update_status(TaskStatus.SUCCESS)
        else:
            self.log(f"Action did not succeed with status: {self.result.status}")
            self.update_status(TaskStatus.FAILURE)


class TaskManager(Node):
    def __init__(self):
        super().__init__("task_manager")
        self.tasks = deque()
        self.current_task = None
        self.move_arm_action_client = ActionClient(self, MoveArm, "/move_arm")

        self.get_logger().info("Task Manager initialized")

        # self.task_thread = Thread(target=self.run_tasks)
        # self.task_thread.start()

    def add_task(self, task):
        task.logger = self.get_logger()
        self.tasks.append(task)
        self.get_logger().info(f"Task added: {task}")

    def remove_task(self, task):
        self.tasks.remove(task)
        self.get_logger().info(f"Task removed: {task}")

    def clear_tasks(self):
        self.tasks.clear()
        self.get_logger().info("Tasks cleared")

    def run_tasks(self):
        self.add_task(MoveArmTask("Move to extended", "extended"))
        self.add_task(MoveArmTask("Move to ready", "ready"))
        while self.tasks:
            self.current_task = self.tasks.popleft()

            if isinstance(self.current_task, MoveArmTask):
                self.current_task.run(self.move_arm_action_client)
            else:
                self.current_task.run()

            while self.current_task.status == TaskStatus.RUNNING:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.current_task.status == TaskStatus.ABORTED:
                    break

        self.current_task = None
        self.get_logger().info("All tasks completed")

    def abort_current_task(self):
        if self.current_task:
            self.current_task.abort()

    def get_tasks(self):
        return self.tasks

    def destroy_node(self):
        self.move_arm_action_client.destroy()
        self.abort_current_task()
        self.task_thread.join()
        super().destroy_node()


def main():
    rclpy.init()
    task_manager = TaskManager()
    executor = MultiThreadedExecutor()
    executor.add_node(task_manager)
    try:
        executor.spin()
    finally:
        task_manager.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
