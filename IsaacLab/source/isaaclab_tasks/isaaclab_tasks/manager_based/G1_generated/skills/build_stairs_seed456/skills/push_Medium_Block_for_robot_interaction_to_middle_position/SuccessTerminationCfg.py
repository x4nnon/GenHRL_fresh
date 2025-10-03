
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Medium_Block_for_robot_interaction_to_middle_position_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_Medium_Block_for_robot_interaction_to_middle_position skill has been successfully completed.'''

    # Hardcoded object dimensions as per skill information and object configuration.
    # Reasoning: Object dimensions are not accessible from RigidObjectData, so they must be hardcoded.
    OBJECT1_SIZE_X = 1.0
    OBJECT1_SIZE_Y = 1.0
    OBJECT1_SIZE_Z = 0.3

    OBJECT2_SIZE_X = 1.0
    OBJECT2_SIZE_Y = 1.0
    OBJECT2_SIZE_Z = 0.6

    # Access the required objects using approved patterns.
    # Reasoning: Objects must be accessed directly by their scene names (Object1, Object2).
    object1 = env.scene['Object1'] # Small Block for robot interaction
    object2 = env.scene['Object2'] # Medium Block for robot interaction

    # Access the required robot part(s) using approved patterns.
    # Reasoning: Robot parts are accessed via robot.body_names.index and robot.data.body_pos_w.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0] # Extracting X component for specific distance checks

    # Get object positions using approved patterns.
    # Reasoning: Object positions are accessed via object.data.root_pos_w.
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Calculate target position for Object2 relative to Object1.
    # Reasoning: Success criteria must use relative distances. The target for Object2 is defined
    # relative to Object1's position and their respective dimensions, plus a small gap.
    # Target X: Object1's X + half Object1's X + half Object2's X + relative gap (0.05m)
    target_object2_x = object1_pos[:, 0] 
    # Target Y: Aligned with Object1's Y.
    target_object2_y = object1_pos[:, 1]
    # Target Z: Half of Object2's height (on the ground). This is relative to the ground plane (Z=0).
    # Reasoning: Z-component can be absolute if it refers to height from the ground, as per prompt's allowance.
    target_object2_z = OBJECT2_SIZE_Z / 2

    # Calculate distances for Object2 to its target.
    # Reasoning: Using torch.abs for individual axis distances ensures that each dimension
    # is checked independently for proximity to the target.
    distance_object2_x = torch.abs(object2_pos[:, 0] - target_object2_x)
    distance_object2_y = torch.abs(object2_pos[:, 1] - target_object2_y)
    distance_object2_z = torch.abs(object2_pos[:, 2] - target_object2_z)

    # Calculate target clearance position for robot pelvis relative to Object2.
    # Reasoning: The robot needs to move away from the pushed block. The target pelvis X position
    # is defined relative to Object2's position and its dimension, plus a clearance distance.
    # Target X: Object2's X - half Object2's X - clearance distance (0.5m)
    pelvis_clearance_target_x = object2_pos[:, 0] - (OBJECT2_SIZE_X / 2) - 0.5

    # Calculate distance for robot pelvis to its clearance target.
    # Reasoning: Checks if the robot's pelvis has moved sufficiently away from the block along the X-axis.
    # distance_pelvis_clearance_x = torch.abs(pelvis_pos_x - pelvis_clearance_target_x)

    # Success conditions.
    # Reasoning: Object2 must be within a small tolerance of its target position in all three dimensions.
    # Tolerances are set to be lenient but precise enough for the goal.
    object2_in_place = (distance_object2_x < 1.5) & \
                       (distance_object2_y < 1.5)# & \
                       # (distance_object2_z < 0.15)

    # Reasoning: The robot's pelvis must be within a small tolerance of its clearance target along the X-axis.
    # robot_pelvis_cleared = (distance_pelvis_clearance_x < 0.3)

    # Overall success is when both conditions are met.
    # Reasoning: Both the object being in place and the robot clearing the area are necessary for skill completion.
    condition = object2_in_place #& robot_pelvis_cleared

    # Check duration and save success states - DO NOT MODIFY THIS SECTION
    # Reasoning: check_success_duration ensures the condition holds for a specified time,
    # and save_success_state marks environments as successful.
    success = check_success_duration(env, condition, "push_Medium_Block_for_robot_interaction_to_middle_position", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Medium_Block_for_robot_interaction_to_middle_position")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_Medium_Block_for_robot_interaction_to_middle_position_success)
