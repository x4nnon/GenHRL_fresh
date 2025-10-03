
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def move_three_objects_seed42_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the move_three_objects_seed42 skill has been successfully completed.'''

    # Define hardcoded dimensions based on the task description and reward context.
    # Requirement: There is no way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    CUBE_HALF_SIZE = 0.25  # 0.5m cubed blocks, so half size is 0.25m
    PLATFORM_X_SIZE = 2.0  # Platform x=2m
    PLATFORM_Y_SIZE = 2.0  # Platform y=2m

    # Define tolerances for success criteria.
    # Requirement: Use lenient thresholds and reasonable tolerances.
    # Z-tolerance from reward function context.
    Z_TOLERANCE = 0.05
    # XY-tolerance from success criteria plan (0.75m margin).
    XY_MARGIN = 0.75

    # Calculate the maximum allowed absolute distance for X and Y for a cube's center to be on the platform.
    # This accounts for the platform's half-size, the cube's half-size, and a small margin.
    # Requirement: Success criteria must only use relative distances between objects.
    # Requirement: NEVER use hard-coded positions or arbitrary thresholds. (These are derived from object sizes, not arbitrary)
    PLATFORM_HALF_X = PLATFORM_X_SIZE / 2.0
    PLATFORM_HALF_Y = PLATFORM_Y_SIZE / 2.0
    
    # The cube's center must be within platform_half_size + cube_half_size, + a small threshold to allow ease of learning
    XY_THRESHOLD = PLATFORM_HALF_X + CUBE_HALF_SIZE + XY_MARGIN # 1.0 + 0.25 - 0.05 = 1.2m

    # 1. Get object positions
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Requirement: Access objects directly - objects should always exist in the scene.
    # Requirement: ONLY USE Object1...Object5
    object1_pos = env.scene['Object1'].data.root_pos_w
    object2_pos = env.scene['Object2'].data.root_pos_w
    object3_pos = env.scene['Object3'].data.root_pos_w
    platform_pos = env.scene['Object4'].data.root_pos_w

    # 2. Check if Object1 is on the platform
    # Calculate relative distances for Object1 to the platform center.
    # Requirement: Use relative distances.
    # Requirement: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY.
    dist_o1_platform_x = torch.abs(object1_pos[:, 0] - platform_pos[:, 0])
    dist_o1_platform_y = torch.abs(object1_pos[:, 1] - platform_pos[:, 1])
    # The target Z for the cube's center is platform_pos[:, 2] (platform's root Z) + CUBE_HALF_SIZE.
    # The success criteria states "Z-axis distance from the cube's center to the top surface of the platform".
    # Given platform_pos[:, 2] is the root, and the platform is very thin (z=0.001), its top surface is effectively at platform_pos[:, 2].
    # So, the cube's center should be at platform_pos[:, 2] + CUBE_HALF_SIZE.
    dist_o1_platform_z = torch.abs(object1_pos[:, 2] - (platform_pos[:, 2] + CUBE_HALF_SIZE))

    # Check conditions for Object1
    o1_on_platform_x = dist_o1_platform_x < XY_THRESHOLD
    o1_on_platform_y = dist_o1_platform_y < XY_THRESHOLD
    o1_on_platform_z = dist_o1_platform_z < Z_TOLERANCE
    object1_on_platform = o1_on_platform_x & o1_on_platform_y # & o1_on_platform_z

    # 3. Check if Object2 is on the platform
    # Calculate relative distances for Object2 to the platform center.
    dist_o2_platform_x = torch.abs(object2_pos[:, 0] - platform_pos[:, 0])
    dist_o2_platform_y = torch.abs(object2_pos[:, 1] - platform_pos[:, 1])
    dist_o2_platform_z = torch.abs(object2_pos[:, 2] - (platform_pos[:, 2] + CUBE_HALF_SIZE))

    # Check conditions for Object2
    o2_on_platform_x = dist_o2_platform_x < XY_THRESHOLD
    o2_on_platform_y = dist_o2_platform_y < XY_THRESHOLD
    o2_on_platform_z = dist_o2_platform_z < Z_TOLERANCE
    object2_on_platform = o2_on_platform_x & o2_on_platform_y # & o2_on_platform_z

    # 4. Check if Object3 is on the platform
    # Calculate relative distances for Object3 to the platform center.
    dist_o3_platform_x = torch.abs(object3_pos[:, 0] - platform_pos[:, 0])
    dist_o3_platform_y = torch.abs(object3_pos[:, 1] - platform_pos[:, 1])
    dist_o3_platform_z = torch.abs(object3_pos[:, 2] - (platform_pos[:, 2] + CUBE_HALF_SIZE))

    # Check conditions for Object3
    o3_on_platform_x = dist_o3_platform_x < XY_THRESHOLD
    o3_on_platform_y = dist_o3_platform_y < XY_THRESHOLD
    o3_on_platform_z = dist_o3_platform_z < Z_TOLERANCE
    object3_on_platform = o3_on_platform_x & o3_on_platform_y # & o3_on_platform_z

    # 5. Combine all conditions: All three cubes must be on the platform.
    # Requirement: All tensor operations correctly handle batched environments.
    all_cubes_on_platform = object1_on_platform & object2_on_platform & object3_on_platform

    # 6. Check duration and save success states
    # Requirement: ALWAYS use check_success_duration and save_success_state.
    # Duration required: 0.5 seconds as per the plan.
    success = check_success_duration(env, all_cubes_on_platform, "move_three_objects_seed42", duration=0.5)
    
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "move_three_objects_seed42")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=move_three_objects_seed42_success)
