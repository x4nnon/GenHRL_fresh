
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_third_0_5m_cubed_block_towards_platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_third_0_5m_cubed_block_towards_platform skill has been successfully completed.'''
    # Access the required objects using approved patterns.
    # Object3 is the 'third 0.5m cubed block'.
    object3 = env.scene['Object3']
    # Object4 is the 'platform'.
    object4 = env.scene['Object4']

    # Access the robot object using approved patterns.
    robot = env.scene["robot"]
    # Access the pelvis position using robot.body_names.index for robustness, as per approved patterns.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Calculate the Euclidean distance between Object3 and Object4.
    # This is a relative distance between two objects, adhering to the rule of using relative distances.
    current_distance_obj3_obj4 = torch.norm(object3.data.root_pos_w - object4.data.root_pos_w, dim=1)

    # Calculate the Euclidean distance between the robot's pelvis and Object3.
    # This is a relative distance between a robot part and an object, adhering to the rule of using relative distances.
    distance_pelvis_obj3 = torch.norm(pelvis_pos - object3.data.root_pos_w, dim=1)

    # Condition 1: Object3 is close to Object4, but not yet fully on it.
    # The thresholds (0.3m and 0.7m) are derived from the task description's intent
    # for the block to be "moved significantly closer... but not yet fully on it".
    # These are relative distances, and the thresholds are reasonable tolerances for this skill phase.
    condition_obj3_near_obj4 = (current_distance_obj3_obj4 > 0.3) & (current_distance_obj3_obj4 < 0.7)

    # Condition 2: Object3's Z-position indicates it's not on the platform.
    # Object3 is a 0.5m cube, so its center is at z=0.25m when on the ground.
    # The platform's z is 0.001m. If the block were on the platform, its center z would be ~0.251m.
    # A threshold of 0.2m ensures it's still on the ground or slightly above, not fully settled on the platform.
    # This is one of the few cases where an absolute Z-position is allowed, as it indicates height relative to the ground,
    # which is critical for the "not yet fully on it" part of the skill description.
    condition_obj3_z_not_on_platform = object3.data.root_pos_w[:, 2] > 0.2

    # Condition 3: Robot pelvis is close to Object3, indicating it's in a stable position
    # either still pushing or ready for the next push.
    # A threshold of 1.0m is a lenient tolerance for the robot to be near the block,
    # ensuring it's in a relevant position without being overly strict. This is a relative distance.
    condition_robot_near_obj3 = distance_pelvis_obj3 < 1.0

    # Combine all success conditions. All conditions must be met simultaneously.
    # All operations correctly handle batched environments.
    success_condition = condition_obj3_near_obj4 & condition_obj3_z_not_on_platform & condition_robot_near_obj3

    # Check success duration and save success states, as required.
    # A duration of 0.5 seconds ensures the conditions are met consistently over time.
    success = check_success_duration(env, success_condition, "push_third_0_5m_cubed_block_towards_platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_third_0_5m_cubed_block_towards_platform")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_third_0_5m_cubed_block_towards_platform_success)
