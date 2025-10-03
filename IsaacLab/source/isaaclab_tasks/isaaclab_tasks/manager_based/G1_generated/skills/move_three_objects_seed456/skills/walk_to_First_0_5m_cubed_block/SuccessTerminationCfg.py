
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_First_0_5m_cubed_block_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_First_0_5m_cubed_block skill has been successfully completed.'''
    
    # Access the robot object.
    # REQUIREMENT: ALWAYS access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    
    # Get the index for the 'pelvis' body part.
    pelvis_idx = robot.body_names.index('pelvis')
    
    # Get the world position of the robot's pelvis.
    # REQUIREMENT: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    
    # Access the target object (Object1: First 0.5m cubed block).
    # REQUIREMENT: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # REQUIREMENT: Access objects directly - objects should always exist in the scene
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w
    
    # Access other objects for collision avoidance checks.
    object2 = env.scene['Object2']
    object2_pos = object2.data.root_pos_w
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w
    object4 = env.scene['Object4']
    object4_pos = object4.data.root_pos_w
    
    # --- Success Condition 1: Pelvis is horizontally within optimal pushing distance of Object1 ---
    # Calculate horizontal distance (XY plane) between robot's pelvis and Object1's center.
    # REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    horizontal_distance_pelvis_obj1 = torch.norm(pelvis_pos[:, :2] - object1_pos[:, :2], dim=1)
    
    # Define the target range for horizontal distance.
    # Object1 is a 0.5m cubed block, so its half-size is 0.25m.
    # Target pushing distance from reward is 0.3m.
    # Success threshold: between 0.2m and 0.4m.
    # REQUIREMENT: Use lenient thresholds and reasonable tolerances. These thresholds are derived from the reward function's target.
    condition_obj1_horizontal = (horizontal_distance_pelvis_obj1 >= 0.2) & (horizontal_distance_pelvis_obj1 <= 0.4)
    
    # --- Success Condition 2: Pelvis is at an appropriate height relative to Object1 ---
    # Calculate Z-distance between robot's pelvis and Object1's center.
    # Object1's center is at 0.25m Z (since it's 0.5m tall and on the ground).
    # Target pelvis Z is 0.7m. Relative Z should be around 0.7 - 0.25 = 0.45m.
    # Success threshold: between 0.4m and 0.9m.
    # REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # REQUIREMENT: Use lenient thresholds and reasonable tolerances. These thresholds are derived from the reward function's target.
    z_distance_pelvis_obj1 = torch.abs(pelvis_pos[:, 2] - object1_pos[:, 2])
    condition_obj1_z = (z_distance_pelvis_obj1 >= 0.4) & (z_distance_pelvis_obj1 <= 0.9)
    
    # --- Success Condition 3: Pelvis is not too close to Object2, Object3, or Object4 ---
    # Calculate horizontal distances to other objects.
    # Success threshold: greater than 0.2m for all.
    # REQUIREMENT: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # REQUIREMENT: Use lenient thresholds and reasonable tolerances. These thresholds are derived from the reward function's collision avoidance.
    horizontal_distance_pelvis_obj2 = torch.norm(pelvis_pos[:, :2] - object2_pos[:, :2], dim=1)
    condition_obj2_avoidance = horizontal_distance_pelvis_obj2 > 0.2
    
    horizontal_distance_pelvis_obj3 = torch.norm(pelvis_pos[:, :2] - object3_pos[:, :2], dim=1)
    condition_obj3_avoidance = horizontal_distance_pelvis_obj3 > 0.2
    
    horizontal_distance_pelvis_obj4 = torch.norm(pelvis_pos[:, :2] - object4_pos[:, :2], dim=1)
    condition_obj4_avoidance = horizontal_distance_pelvis_obj4 > 0.2
    
    # Combine all conditions. All must be true for success.
    # REQUIREMENT: All tensor operations correctly handle batched environments.
    condition = (condition_obj1_horizontal & 
                 condition_obj1_z & 
                 condition_obj2_avoidance & 
                 condition_obj3_avoidance & 
                 condition_obj4_avoidance)
    
    # Check success duration and save success states.
    # REQUIREMENT: ALWAYS use check_success_duration and save_success_state
    # REQUIREMENT: Success duration properly checked with appropriate duration value (0.5s as per plan).
    success = check_success_duration(env, condition, "walk_to_First_0_5m_cubed_block", duration=0.5)
    
    # REQUIREMENT: Success states saved for successful environments.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_First_0_5m_cubed_block")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_First_0_5m_cubed_block_success)
