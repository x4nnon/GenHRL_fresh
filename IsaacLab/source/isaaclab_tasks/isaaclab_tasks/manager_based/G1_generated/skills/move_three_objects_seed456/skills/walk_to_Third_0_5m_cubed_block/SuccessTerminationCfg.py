
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_Third_0_5m_cubed_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_Third_0_5m_cubed_block skill has been successfully completed.'''
    # Access the robot object.
    # REQUIREMENT: Always access robot using env.scene["robot"].
    robot = env.scene["robot"]
    
    # Get the index for the 'pelvis' body part.
    # REQUIREMENT: Always access robot parts using robot.body_names.index('part_name').
    pelvis_idx = robot.body_names.index('pelvis')
    # Get the world position of the robot's pelvis.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # Access Object3 (Third 0.5m cubed block).
    # REQUIREMENT: Always access object positions using env.scene['ObjectName'].data.root_pos_w.
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w
    
    # Hardcode Object3 dimensions as per requirements (0.5m cubed block).
    # REQUIREMENT: Object dimensions must be hardcoded from the object configuration.
    object3_half_height = 0.25 # Half of 0.5m block height
    
    # Calculate the horizontal distance (XY plane) between the robot's pelvis and Object3's center.
    # REQUIREMENT: Use relative distances between objects and robot parts.
    # REQUIREMENT: Handle tensor operations for batched environments.
    distance_x = object3_pos[:, 0] - pelvis_pos[:, 0]
    distance_y = object3_pos[:, 1] - pelvis_pos[:, 1]
    current_xy_distance = torch.sqrt(distance_x**2 + distance_y**2)
    
    # Calculate the Z-axis distance between the robot's pelvis and Object3's center.
    # The object's center Z is object3_pos[:, 2].
    # The pelvis Z is pelvis_pos[:, 2].
    # The target relative Z-distance is 0.45m (pelvis Z 0.7m - object center Z 0.25m).
    # REQUIREMENT: Use relative distances. Z-height of pelvis relative to object center.
    current_z_distance_relative_to_object_center = pelvis_pos[:, 2] - object3_pos[:, 2]
    
    # Define success thresholds.
    # REQUIREMENT: Use lenient thresholds and reasonable tolerances.
    # Horizontal distance threshold: pelvis within 0.5m of Object3's center.
    xy_threshold = 0.5
    # Target relative Z-distance: pelvis should be 0.45m above Object3's center.
    target_relative_z_distance = 0.45
    # Z-axis tolerance: allow for +/- 0.1m deviation from the target relative Z-distance.
    z_tolerance = 0.1
    
    # Check the horizontal distance condition.
    condition_xy = current_xy_distance <= xy_threshold
    
    # Check the Z-axis distance condition.
    condition_z = torch.abs(current_z_distance_relative_to_object_center - target_relative_z_distance) <= z_tolerance
    
    # Combine all conditions for overall success.
    # REQUIREMENT: Combine conditions with proper tensor operations.
    overall_condition = condition_xy & condition_z
    
    # Check success duration and save success states.
    # REQUIREMENT: Always use check_success_duration and save_success_state.
    # Duration required: 0.5 seconds as per the plan.
    success = check_success_duration(env, overall_condition, "walk_to_Third_0_5m_cubed_block", duration=0.5)
    
    # Save success states for environments that have met the success criteria.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_Third_0_5m_cubed_block")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_Third_0_5m_cubed_block_success)
