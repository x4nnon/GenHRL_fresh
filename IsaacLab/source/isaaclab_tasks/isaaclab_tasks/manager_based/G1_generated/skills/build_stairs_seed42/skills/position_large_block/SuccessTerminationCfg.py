
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def position_large_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the position_large_block skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access the required objects using approved patterns
    # Object2 is the Medium Block for robot interaction.
    object2 = env.scene['Object2']
    # Object3 is the Large Block for robot interaction.
    object3 = env.scene['Object3']

    # Access object positions using approved patterns
    # This adheres to the rule: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Hardcode the Large Block's height as per the object configuration and rules.
    # This adheres to the rule: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    # Object3 (Large Block) measures x=1m y=1m and z=0.9m.
    object3_height = 0.9

    # Calculate the target Z position for Object3's center.
    # For the block to be on the ground, its center Z should be half its height.
    # This is an allowed absolute Z reference for ground positioning, as per the rule: z is the only absolute position allowed.
    target_object3_z_center = object3_height / 2.0

    # Calculate relative distances for Object3 to its target position based on Object2.
    # This adheres to the rule: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts.
    # Target X for Object3: 1.0m behind Object2's center. This is a relative offset from Object2's position.
    # Target Y for Object3: Aligned with Object2's Y. This is a relative alignment.
    # Target Z for Object3: Half its height (on the ground). This is a relative distance to a hardcoded ground reference.
    # All distances are calculated using torch.abs for individual components, ensuring relative checks.
    distance_x_obj3 = torch.abs(object3_pos[:, 0] - (object2_pos[:, 0] - 1.0))
    distance_y_obj3 = torch.abs(object3_pos[:, 1] - object2_pos[:, 1])
    distance_z_obj3 = torch.abs(object3_pos[:, 2] - target_object3_z_center)

    # Define success thresholds. These are lenient thresholds for positioning.
    # This adheres to the rule: USE LENIENT THRESHOLDS and REASONABLE TOLERANCES.
    # Thresholds are applied separately to X, Y, and Z components, as per the rule: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY.
    threshold_xy = 0.15 # 15cm tolerance for X and Y positioning.
    threshold_z = 0.15  # 15cm tolerance for Z positioning (on the ground).

    # Success condition: Object3 is within the defined thresholds of its target relative position.
    # All conditions must be met for success.
    condition = (distance_x_obj3 < threshold_xy) & \
                (distance_y_obj3 < threshold_xy) & \
                (distance_z_obj3 < threshold_z)

    # Check success duration and save success states.
    # This adheres to the rules: ALWAYS use check_success_duration and save_success_state.
    # A duration of 0.5 seconds is used to ensure stability of the success condition.
    success = check_success_duration(env, condition, "position_large_block", duration=0.5)

    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "position_large_block")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=position_large_block_success)
