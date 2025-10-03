
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def push_second_0_5m_cubed_block_onto_platform_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the push_second_0_5m_cubed_block_onto_platform skill has been successfully completed.'''

    # Access the robot object
    # REQUIREMENT: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Access the required objects using approved patterns
    # REQUIREMENT: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # REQUIREMENT: Access objects directly - objects should always exist in the scene
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object4 = env.scene['Object4'] # platform
    object1 = env.scene['Object1'] # first 0.5m cubed block
    object3 = env.scene['Object3'] # third 0.5m cubed block

    obj2_pos = object2.data.root_pos_w
    obj4_pos = object4.data.root_pos_w
    obj1_pos = object1.data.root_pos_w
    obj3_pos = object3.data.root_pos_w

    # Define object dimensions (hardcoded from object configuration as per requirements)
    # REQUIREMENT: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    block_half_size = 0.5 / 2.0 # 0.25m for 0.5m cubed block
    platform_x_size = 2.0
    platform_y_size = 2.0
    # platform_base_z = 0.001 # From task description, platform is at z=0.001 - This is not used directly, obj4_pos[:, 2] is used instead.

    # --- Success Criterion 1: Block Placement (Object2 fully on Object4) ---
    # REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # REQUIREMENT: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY, INCLUDING THEIR THRESHOLDS.

    # Calculate horizontal distances between Object2 center and Object4 center
    dist_x_obj2_obj4 = torch.abs(obj2_pos[:, 0] - obj4_pos[:, 0])
    dist_y_obj2_obj4 = torch.abs(obj2_pos[:, 1] - obj4_pos[:, 1])

    # Check if Object2 is within platform's horizontal boundaries
    # Tolerance added to allow for slight overlap/edge cases, but still ensuring containment.
    # (platform_size / 2.0 - block_half_size) gives the maximum center-to-center distance for full containment.
    # Adding 0.05m tolerance for leniency.
    block_on_platform_x = dist_x_obj2_obj4 <= (platform_x_size / 2.0 - block_half_size + 0.05)
    block_on_platform_y = dist_y_obj2_obj4 <= (platform_y_size / 2.0 - block_half_size + 0.05)

    # Check if Object2 is resting on the platform's surface (z-position)
    # Target Z for Object2's center is platform_base_z + block_half_size.
    # Using a tolerance of 0.075m for the z-height check.
    target_obj2_z = obj4_pos[:, 2] + block_half_size
    block_on_platform_z = torch.abs(obj2_pos[:, 2] - target_obj2_z) < 0.075

    # Combined condition for Object2 being fully on Object4
    object2_fully_on_platform = block_on_platform_x & block_on_platform_y & block_on_platform_z

    # --- Success Criterion 2: Robot Position (Pelvis not on platform and near other blocks) ---

    # Check if robot pelvis is NOT on the platform
    # Pelvis is considered off the platform if it's outside horizontal bounds OR below a certain Z height relative to platform.
    # Using a slightly larger buffer (0.1m) for the robot's horizontal clearance from the platform edges.
    pelvis_outside_platform_x = torch.abs(pelvis_pos[:, 0] - obj4_pos[:, 0]) > (platform_x_size / 2.0 + 0.1)
    pelvis_outside_platform_y = torch.abs(pelvis_pos[:, 1] - obj4_pos[:, 1]) > (platform_y_size / 2.0 + 0.1)
    # Pelvis is considered off the platform if its Z is below the platform's surface + a small buffer (0.1m).
    # This uses a relative Z check against the platform's Z position.
    pelvis_below_platform_z = pelvis_pos[:, 2] < (obj4_pos[:, 2] + 0.1)

    robot_not_on_platform = pelvis_outside_platform_x | pelvis_outside_platform_y | pelvis_below_platform_z

    # Check if robot pelvis is within a reasonable distance of the other blocks (Object1 and Object3)
    # Calculate the average x and y positions of Object1 and Object3 as a target for the pelvis.
    avg_obj1_obj3_x = (obj1_pos[:, 0] + obj3_pos[:, 0]) / 2.0
    avg_obj1_obj3_y = (obj1_pos[:, 1] + obj3_pos[:, 1]) / 2.0

    # Calculate relative distances from pelvis to the average position of the other blocks
    dist_x_pelvis_next_blocks = torch.abs(pelvis_pos[:, 0] - avg_obj1_obj3_x)
    dist_y_pelvis_next_blocks = torch.abs(pelvis_pos[:, 1] - avg_obj1_obj3_y)

    # Robot is considered ready for next skill if it's within 1.5m horizontally of the other blocks' average position.
    # This uses lenient thresholds for secondary conditions.
    robot_near_other_blocks = (dist_x_pelvis_next_blocks < 1.5) & (dist_y_pelvis_next_blocks < 1.5)

    # Combine all success conditions
    # The block must be on the platform AND the robot must not be on the platform AND the robot must be near the other blocks.
    condition = object2_fully_on_platform & robot_not_on_platform & robot_near_other_blocks

    # Check duration and save success states
    # REQUIREMENT: ALWAYS use check_success_duration and save_success_state
    # Duration required: 0.5 seconds as per plan.
    success = check_success_duration(env, condition, "push_second_0_5m_cubed_block_onto_platform", duration=0.5)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "push_second_0_5m_cubed_block_onto_platform")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=push_second_0_5m_cubed_block_onto_platform_success)
