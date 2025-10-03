
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Large_Block_into_stair_base_position_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_Large_Block_into_stair_base_position skill has been successfully completed.'''
    # REASONING: Accessing the robot object using the approved pattern.
    robot = env.scene["robot"]

    # REASONING: Accessing Object3 (Large Block) using the approved pattern.
    object3 = env.scene['Object3']

    # REASONING: Getting the index of the 'pelvis' robot part using the approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    # REASONING: Getting the position of the 'pelvis' robot part using the approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # REASONING: Hardcoding Object3's height from the object configuration (z=0.9m), as allowed by rule 6.
    object3_height = 0.9
    # REASONING: Calculating the stable Z position (center) of Object3 when it's on the ground.
    object3_stable_z = object3_height / 2.0

    # REASONING: Hardcoding a conceptual default Z height for the robot's pelvis when standing stably, as allowed for fixed robot posture targets.
    robot_default_pelvis_z = 0.7

    # REASONING: Calculating the Z distance of Object3 from its stable ground position.
    # This is a relative distance to a conceptual stable height.
    distance_object3_z = object3.data.root_pos_w[:, 2] - object3_stable_z

    # REASONING: Calculating the Z distance of the robot's pelvis from its default standing height.
    # This is a relative distance to a conceptual stable height.
    distance_pelvis_z = pelvis_pos[:, 2] - robot_default_pelvis_z

    # REASONING: Calculating the relative X distance between Object3 and the robot's pelvis.
    # The goal is for Object3 to be approximately 1.0m ahead of the robot's pelvis in X.
    relative_x_distance = object3.data.root_pos_w[:, 0] - pelvis_pos[:, 0]

    # REASONING: Calculating the relative Y distance between Object3 and the robot's pelvis.
    # The goal is for them to be aligned in Y (distance close to 0).
    relative_y_distance = object3.data.root_pos_w[:, 1] - pelvis_pos[:, 1]

    # REASONING: Condition for Object3 being at its stable Z height. Using a lenient threshold (0.1m).
    condition_object3_z = torch.abs(distance_object3_z) < 0.1

    # REASONING: Condition for the robot's pelvis being at its default standing height. Using a lenient threshold (0.1m).
    condition_pelvis_z = torch.abs(distance_pelvis_z) < 0.1

    # REASONING: Condition for Object3 being approximately 1.0m ahead of the robot's pelvis in X.
    # Using a reasonable threshold (0.3m) to allow for some variability in final robot position.
    condition_relative_x = torch.abs(relative_x_distance - 1.0) < 0.3

    # REASONING: Condition for Object3 and the robot's pelvis being aligned in Y.
    # Using a reasonable threshold (0.3m) for alignment.
    condition_relative_y = torch.abs(relative_y_distance) < 0.3

    # REASONING: Combining all conditions for overall success. All conditions must be met.
    success_condition = condition_object3_z & condition_pelvis_z & condition_relative_x & condition_relative_y

    # REASONING: Checking if the success condition has been maintained for a sufficient duration (0.5 seconds), as required.
    success = check_success_duration(env, success_condition, "push_Large_Block_into_stair_base_position", duration=0.5)

    # REASONING: Saving the success state for environments that have successfully completed the skill, as required.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Large_Block_into_stair_base_position")

    return success

class SuccessTerminationCfg:
    # REASONING: Defining the success termination configuration, linking to the implemented success function.
    success = DoneTerm(func=push_Large_Block_into_stair_base_position_success)
