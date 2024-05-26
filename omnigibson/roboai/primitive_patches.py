import numpy as np
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError


def _simplified_place_with_predicate(
    self, obj, predicate, near_poses=None, near_poses_threshold=None
):
    """
    Yields action for the robot to navigate to the object if needed, then to place it

    Args:
        obj (StatefulObject): Object for robot to place the object in its hand on
        predicate (object_states.OnTop or object_states.Inside): Determines whether to place on top or inside

    Returns:
        np.array or None: Action array for one step for the robot to place or None if place completed
    """
    obj_in_hand = self._get_obj_in_hand()
    if obj_in_hand is None:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "You need to be grasping an object first to place it somewhere.",
        )

    # Find a spot to put it
    # obj_pose = self._sample_pose_with_object_and_predicate(
    #     predicate,
    #     obj_in_hand,
    #     obj,
    #     near_poses=near_poses,
    #     near_poses_threshold=near_poses_threshold,
    # )
    obj_pose = (np.array([-0.7, 0.5, 0.5]), np.array([0.0, 0.0, 0.0, 1.0]))

    # Get close, release the object.
    # yield from self._navigate_if_needed(obj, pose_on_obj=obj_pose)
    yield from self._release()

    # Actually move the object to the spot and step a bit to settle it.
    obj_in_hand.set_position_orientation(*obj_pose)
    # yield from self._settle_robot()


def _quick_settle_robot(self):
    """
    Yields a no op action for a few steps to allow the robot and physics to settle

    Returns:
        np.array or None: Action array for one step for the robot to do nothing
    """
    print("Settling robot")
    for _ in range(10):
        empty_action = self._empty_action()
        yield self._postprocess_action(empty_action)

    print("Settled robot")
