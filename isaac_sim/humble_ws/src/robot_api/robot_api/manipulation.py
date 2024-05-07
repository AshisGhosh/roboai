import time
from typing import List
from dataclasses import dataclass

# generic ros libraries
import rclpy

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    PlanRequestParameters,
)

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import Constraints
from rclpy.action import ActionServer, CancelResponse, GoalResponse, ActionClient
from control_msgs.action import GripperCommand
from roboai_interfaces.action import MoveArm, ControlGripper


@dataclass
class ArmState:
    JOINT_VALUES = "joint_values"
    POSE = "pose"
    name: str
    type: str
    joint_values: List[float | int] | None = None
    pose: List[float | int] | None = None

    def __post_init__(self):
        if self.joint_values is not None and self.pose is not None:
            raise ValueError("Only one of joint_values or pose can be set")
        if self.joint_values is None and self.pose is None:
            raise ValueError("Either joint_values or pose must be set")

    @classmethod
    def get_pose_as_pose(self, pose_array: List[float | int]) -> Pose:
        pose = Pose()
        pose.position.x = pose_array[0]
        pose.position.y = pose_array[1]
        pose.position.z = pose_array[2]
        pose.orientation.x = pose_array[3]
        pose.orientation.y = pose_array[4]
        pose.orientation.z = pose_array[5]
        pose.orientation.w = pose_array[6]
        return pose


# Arm states
LOOK_DOWN_QUAT = [0.924, -0.383, 0.0, 0.0]
PICK_CENTER = ArmState(
    name="pick_center", type=ArmState.POSE, pose=[0.5, 0.0, 0.5, *LOOK_DOWN_QUAT]
)
DROP = ArmState(name="drop", type=ArmState.POSE, pose=[0.5, -0.5, 0.5, *LOOK_DOWN_QUAT])
ARM_STATE_LIST = [PICK_CENTER, DROP]

ARM_STATES = {state.name: state for state in ARM_STATE_LIST}


class ManipulationAPI(Node):
    def __init__(
        self,
        robot_arm_planning_component="panda_arm",
        robot_arm_eef_link="panda_link8",
    ):
        super().__init__("manipulation_api")
        moveit_config = self._get_moveit_config()

        self.robot_arm_planning_component_name = robot_arm_planning_component
        self.robot_arm_eef_link = robot_arm_eef_link

        self.robot = MoveItPy(node_name="moveit_py", config_dict=moveit_config)
        self.robot_arm_planning_component = self.robot.get_planning_component(
            self.robot_arm_planning_component_name
        )

        self._action_server = ActionServer(
            self,
            MoveArm,
            "move_arm",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self._last_gripper_result = None
        self._gripper_action_client = ActionClient(
            self, GripperCommand, "/panda_hand_controller/gripper_cmd"
        )
        self._gripper_action_server = ActionServer(
            self,
            ControlGripper,
            "control_gripper",
            execute_callback=self.gripper_execute_callback,
            goal_callback=self.gripper_goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("Manipulation API initialized")

    def _get_moveit_config(self):
        from moveit_configs_utils import MoveItConfigsBuilder
        from ament_index_python.packages import get_package_share_directory

        moveit_config = (
            MoveItConfigsBuilder(
                robot_name="panda", package_name="moveit_resources_panda_moveit_config"
            )
            .robot_description(file_path="config/panda.urdf.xacro")
            .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
            .moveit_cpp(
                file_path=get_package_share_directory("robot_api")
                + "/config/moveit_franka_python.yaml"
            )
            .to_moveit_configs()
        )
        moveit_config = moveit_config.to_dict()
        return moveit_config

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info("Received goal request")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        # TODO: Implement cancel
        self.get_logger().info("Received cancel request")
        self.get_logger().error("Cancel not implemented")
        return CancelResponse.REJECT

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f"Received goal: {goal}")
        result = MoveArm.Result()
        status = self.move_arm(
            goal.configuration_goal,
            goal.cartesian_goal,
            goal.joint_goal,
            goal.constraints_goal,
            None,
        )
        self.get_logger().info(f"Move arm status: {status.status}")
        result.status = status.status
        if result.status == "SUCCEEDED":
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def gripper_goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info("Received gripper goal request")
        return GoalResponse.ACCEPT

    def gripper_execute_callback(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f"Received gripper goal: {goal}")
        result = ControlGripper.Result()
        status = self.control_gripper(goal.goal_state)
        self.get_logger().info(f"Gripper status: {status}")
        result.status = status
        if result.status == "SUCCEEDED":
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def plan(
        self,
        goal_state,
        start_state=None,
    ):
        self.get_logger().info("Planning trajectory")
        self.get_logger().info(
            f"Goal state: {goal_state} type {type(goal_state)}: {isinstance(goal_state, str)}"
        )
        plan_request_parameters = None

        if start_state is None:
            self.robot_arm_planning_component.set_start_state_to_current_state()
        else:
            self.robot_arm_planning_component.set_start_state(robot_state=start_state)

        if isinstance(goal_state, str):
            self.get_logger().info(f"Setting goal state to {goal_state}")
            self.robot_arm_planning_component.set_goal_state(
                configuration_name=goal_state
            )
        elif isinstance(goal_state, RobotState):
            self.robot_arm_planning_component.set_goal_state(robot_state=goal_state)
        elif isinstance(goal_state, PoseStamped):
            self.robot_arm_planning_component.set_goal_state(
                pose_stamped_msg=goal_state, pose_link=self.robot_arm_eef_link
            )
            plan_request_parameters = PlanRequestParameters(
                self.robot, "pilz_industrial_motion_planner"
            )
            plan_request_parameters.planning_time = 1.0
            plan_request_parameters.planning_attempts = 1
            plan_request_parameters.max_velocity_scaling_factor = 0.1
            plan_request_parameters.max_acceleration_scaling_factor = 0.1
            plan_request_parameters.planning_pipeline = "pilz_industrial_motion_planner"
            plan_request_parameters.planner_id = "PTP"
        elif isinstance(goal_state, Constraints):
            self.robot_arm_planning_component.set_goal_state(
                motion_plan_constraints=[goal_state]
            )
        elif isinstance(goal_state, list):
            self.robot_arm_planning_component.set_goal_state(
                motion_plan_constraints=goal_state
            )
        else:
            raise ValueError("Invalid goal state type")

        self.get_logger().info(
            f"Planning trajectory for goal of type {type(goal_state)}"
        )
        start_time = time.time()
        if plan_request_parameters is not None:
            plan_result = self.robot_arm_planning_component.plan(
                single_plan_parameters=plan_request_parameters
            )
        else:
            plan_result = self.robot_arm_planning_component.plan()
        end_time = time.time()
        self.get_logger().info("Planning completed")
        self.get_logger().info(f"Planning time: {end_time - start_time}")

        return plan_result

    def execute(self, trajectory):
        self.get_logger().info("Executing trajectory")
        return self.robot.execute(trajectory, controllers=[])
        # execution_manager = self.robot.get_trajactory_execution_manager()

        # current_status = execution_manager.get_execution_status()
        # # https://moveit.picknik.ai/main/api/html/structmoveit__controller__manager_1_1ExecutionStatus.html

    def move_arm(
        self,
        configuration_goal=None,
        cartesian_goal=None,
        joint_goal=None,
        constraints_goal=None,
        start_state=None,
    ):
        self.get_logger().info("Moving arm")

        if start_state is not None:
            raise NotImplementedError("Custom start state not implemented")

        if configuration_goal not in [None, ""]:
            if configuration_goal in ["extended", "ready"]:
                goal_state = configuration_goal
            elif configuration_goal in ARM_STATES:
                goal_state = ARM_STATES[configuration_goal]
                if goal_state.type == ArmState.JOINT_VALUES:
                    return self.move_arm(joint_goal=goal_state.joint_values)
                elif goal_state.type == ArmState.POSE:
                    return self.move_arm(cartesian_goal=goal_state.pose)
            else:
                raise ValueError("Invalid configuration goal")
        elif cartesian_goal is not None:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "panda_link0"
            pose_stamped.pose = ArmState.get_pose_as_pose(cartesian_goal)
            goal_state = pose_stamped
        elif joint_goal is not None:
            robot_model = self.robot.get_robot_model()
            robot_state = RobotState(robot_model)
            robot_state.set_joint_group_positions(
                self.robot_arm_planning_component_name, joint_goal
            )
            goal_state = robot_state
        elif constraints_goal is not None:
            raise NotImplementedError("Constraints goal not implemented")
            goal_state = constraints_goal
        else:
            raise ValueError("No goal state provided")

        plan_result = self.plan(goal_state)
        return self.execute(plan_result.trajectory)

    def gripper_send_goal(self, goal: str):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 0.04 if goal == "open" else 0.0

        self._gripper_action_client.wait_for_server(timeout_sec=1)
        future = self._gripper_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().error("Goal rejected :(")
            return

        self.get_logger().info("Goal accepted :)")
        # Wait for the result
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.gripper_get_result_callback)

    def gripper_get_result_callback(self, future) -> None:
        result = future.result().result
        self.get_logger().info(
            f"Result received: {result}, reached goal: {result.reached_goal}, stalled: {result.stalled}"
        )
        if result.reached_goal or result.stalled:
            self.get_logger().info("Gripper command success.")
        else:
            self.get_logger().error("Gripper command failed.")
        self._last_gripper_result = result

    def control_gripper(self, goal: str):
        self._last_gripper_result = None
        if goal not in ["open", "close"]:
            raise ValueError("Invalid gripper goal")

        self.get_logger().info(f"Control gripper to {goal}")
        self.gripper_send_goal(goal)
        while self._last_gripper_result is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"Result: {self._last_gripper_result}")
        return "SUCCEEDED"


def main():
    rclpy.init()
    node = ManipulationAPI()
    try:
        while True:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")

    node.destroy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
