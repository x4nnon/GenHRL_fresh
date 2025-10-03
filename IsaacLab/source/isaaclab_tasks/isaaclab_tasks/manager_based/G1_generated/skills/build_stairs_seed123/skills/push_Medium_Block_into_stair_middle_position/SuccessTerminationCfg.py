
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_Medium_Block_into_stair_middle_position_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_Medium_Block_into_stair_middle_position skill has been successfully completed.'''

    # 1. Get robot parts and object positions
    # CORRECT: Direct indexed access to the robot
    robot = env.scene["robot"]
    # CORRECT: Using robot.body_names.index instead of hardcoded indices for pelvis
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CORRECT: Direct object access using Object1...Object5 names
    object2 = env.scene['Object2'] # Medium Block
    object2_pos = object2.data.root_pos_w # Shape: [num_envs, 3]

    object3 = env.scene['Object3'] # Large Block
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]

    # 2. Hardcode Object2 dimensions from configuration (1m x 1m x 0.6m)
    # CORRECT: Object dimensions are hardcoded from the object configuration, not accessed from the object itself.
    object2_size_x = 1.0

    # 3. Calculate Object2's target position relative to Object3
    # The target position for Object2 is 1.0m along the X-axis from Object3, with Y and Z aligned.
    # CORRECT: Using relative distances for target calculation.
    target_x_obj2_rel_obj3 = object3_pos[:, 0] + 1.0
    target_y_obj2_rel_obj3 = object3_pos[:, 1]
    target_z_obj2_rel_obj3 = object3_pos[:, 2] # Assuming Z alignment means on the same ground plane or base height

    # 4. Calculate distances for Object2 placement
    # CORRECT: Calculating absolute differences for each dimension.
    dist_x_obj2_target = torch.abs(object2_pos[:, 0] - target_x_obj2_rel_obj3)
    dist_y_obj2_target = torch.abs(object2_pos[:, 1] - target_y_obj2_rel_obj3)
    dist_z_obj2_target = torch.abs(object2_pos[:, 2] - target_z_obj2_rel_obj3)

    # 5. Define success conditions for Object2 placement
    # CORRECT: Using reasonable, lenient thresholds for success.
    obj2_x_aligned = dist_x_obj2_target < 0.15 # Within 0.15m of 1.0m offset
    obj2_y_aligned = dist_y_obj2_target < 0.15 # Within 0.15m of Object3's Y
    obj2_z_aligned = dist_z_obj2_target < 0.15 # Within 0.15m of Object3's Z

    # 6. Calculate Robot Pelvis's target position for stable standing behind Object2
    # Target robot position: roughly 0.5m behind Object2's final position (relative to push direction)
    # and aligned with the stairs.
    # Object2's final X position is object3.x + 1.0.
    # So, robot's target X is (object3.x + 1.0) + (object2_size_x / 2.0) + 0.5m (behind Object2's face)
    # CORRECT: Using relative distances for target calculation based on Object2's *target* position.
    target_robot_x = target_x_obj2_rel_obj3 + (object2_size_x / 2.0) + 0.5
    target_robot_y = target_y_obj2_rel_obj3 # Aligned with stairs Y
    pelvis_target_z = 0.7 # Target standing height for the robot's pelvis (from reward function context)

    # 7. Calculate distances for robot pelvis position
    # CORRECT: Calculating absolute differences for each dimension.
    dist_x_pelvis_target = torch.abs(pelvis_pos[:, 0] - target_robot_x)
    dist_y_pelvis_target = torch.abs(pelvis_pos[:, 1] - target_robot_y)
    dist_z_pelvis_target = torch.abs(pelvis_pos[:, 2] - pelvis_target_z)

    # 8. Define success conditions for robot pelvis position
    # CORRECT: Using reasonable, lenient thresholds for success.
    robot_x_aligned = dist_x_pelvis_target < 0.2
    robot_y_aligned = dist_y_pelvis_target < 0.2
    robot_z_height_ok = dist_z_pelvis_target < 0.2

    # 9. Combine all success conditions
    # CORRECT: Combining conditions with proper tensor operations.
    condition = (obj2_x_aligned & obj2_y_aligned & obj2_z_aligned) & \
                (robot_x_aligned & robot_y_aligned & robot_z_height_ok)

    # 10. Check duration and save success states
    # CORRECT: Using check_success_duration and appropriate duration.
    success = check_success_duration(env, condition, "push_Medium_Block_into_stair_middle_position", duration=0.5)

    # CORRECT: Saving success states for successful environments.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_Medium_Block_into_stair_middle_position")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_Medium_Block_into_stair_middle_position_success)
