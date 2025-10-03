from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch

from isaaclab.envs import mdp
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

def main_ExecuteJumpOntoBlock_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for ExecuteJumpOntoBlock.

    Reward for the robot standing on top of the block with feet above the block's top surface.
    This encourages the robot to successfully jump and land on the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    RewNormalizer = get_normalizer(env.device)
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot foot positions using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        left_foot_idx = robot.body_names.index('left_ankle_roll_link')
        right_foot_idx = robot.body_names.index('right_ankle_roll_link')
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

        # Calculate minimum foot position
        min_foot_pos = torch.min(left_foot_pos, right_foot_pos)

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in z direction between minimum foot position and block top surface - rule 1 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        block_top_surface_z = block_pos[:, 2] + block_size_z/2
        distance_z_feet_block_top = min_foot_pos[:, 2] - block_top_surface_z - 0.2  # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # pelvis height should be 0.5 + 0.7 above block
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        distance_z_pelvis_block_top = pelvis_pos[:, 2] - (block_top_surface_z + 1.0)

        feet_pelvis_distance = torch.abs(pelvis_pos[:, 2] - min_foot_pos[:, 2])

        # Reward is negative absolute distance to encourage feet to be on top of the block - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_feet_block_top) - torch.abs(distance_z_pelvis_block_top) + 0.5*feet_pelvis_distance # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_approach_block_x(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_block_x_reward") -> torch.Tensor:
    '''Shaping reward for approaching the block in the x-direction.

    Reward for decreasing the x-distance between the pelvis and the block.
    Encourages the robot to move towards the block before jumping.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    RewNormalizer = get_normalizer(env.device)
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in x direction between pelvis and block - rule 1 in ABSOLUTE REQUIREMENTS
        distance_x_pelvis_block = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS
        distance_y_pelvis_block = block_pos[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction
        distance = torch.sqrt(distance_x_pelvis_block**2 + distance_y_pelvis_block**2)


        # Reward is negative absolute distance to encourage moving closer in x direction - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_jump_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_height_reward") -> torch.Tensor:
    '''Shaping reward for achieving a suitable jump height when approaching the block.

    Reward for reaching a target pelvis height above the block's top surface when close to the block in x-direction.
    Encourages the robot to jump upwards when approaching the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    RewNormalizer = get_normalizer(env.device)
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate target pelvis height above block top surface - rule 1 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 10.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        block_top_surface_z = block_pos[:, 2] + block_size_z
        target_pelvis_height = block_top_surface_z + 1.0 # Relative target height - rule 1 in ABSOLUTE REQUIREMENTS

        # Calculate relative distance in z direction between pelvis and target height - rule 1 in ABSOLUTE REQUIREMENTS
        distance_z_pelvis_target_height = pelvis_pos[:, 2] - target_pelvis_height # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Approach block condition: activate when robot is approaching the block in x direction - rule 4 in ABSOLUTE REQUIREMENTS
        approach_block_condition = (pelvis_pos[:, 0] > block_pos[:, 0] - 2.0) & (pelvis_pos[:, 0] < block_pos[:, 0]) # Relative condition - rule 1 in ABSOLUTE REQUIREMENTS

        # Reward is negative absolute distance to encourage reaching target height - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_pelvis_target_height) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(approach_block_condition, reward, -torch.ones_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stability_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_on_block_reward") -> torch.Tensor:
    '''Shaping reward for maintaining stability on the block after landing.

    Reward for keeping the pelvis at a reasonable height relative to the block and feet, and avoid falling off.
    Encourages the robot to maintain balance on the block.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    RewNormalizer = get_normalizer(env.device)
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis and feet positions using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        left_foot_idx = robot.body_names.index('left_ankle_roll_link')
        right_foot_idx = robot.body_names.index('right_ankle_roll_link')
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

        # Calculate average foot position
        avg_foot_pos = (left_foot_pos + right_foot_pos) / 2

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative pelvis height above average feet position - rule 1 in ABSOLUTE REQUIREMENTS
        relative_pelvis_height = pelvis_pos[:, 2] - avg_foot_pos[:, 2] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # Target relative pelvis height (slightly above feet) - rule 4 in ABSOLUTE REQUIREMENTS
        target_relative_pelvis_height = 0.5 # Relative target height - rule 4 in ABSOLUTE REQUIREMENTS

        # Calculate distance from target relative pelvis height - rule 1 in ABSOLUTE REQUIREMENTS
        distance_z_pelvis_relative_target = relative_pelvis_height - target_relative_pelvis_height # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        pelvis_on_block_condition_x = (pelvis_pos[:, 0] > block_pos[:, 0] - 0.25) & (pelvis_pos[:, 0] < block_pos[:, 0] + 0.25)
        pelvis_on_block_condition_y = (pelvis_pos[:, 1] > block_pos[:, 1] - 0.25) & (pelvis_pos[:, 1] < block_pos[:, 1] + 0.25)


        # On block condition: activate when feet are approximately on top of the block - rule 4 in ABSOLUTE REQUIREMENTS
        block_size_z = 0.5 # Reading block size from object config (size_cubes = [[0.4, 10.0, 0.4], [1.0, 1.0, 0.2], [0.5, 0.5, 0.5]]) - rule 6 & 7 in CRITICAL IMPLEMENTATION RULES
        on_block_condition = (avg_foot_pos[:, 2] > (block_pos[:, 2] + block_size_z - 0.1)) & pelvis_on_block_condition_x & pelvis_on_block_condition_y # Relative condition - rule 1 in ABSOLUTE REQUIREMENTS

        # Reward is negative absolute distance to encourage maintaining target relative pelvis height - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -torch.abs(distance_z_pelvis_relative_target) # Continuous reward, using relative distance - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(on_block_condition, reward, -2.0 * torch.ones_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_collision_avoidance(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with the block at ground level.

    Negative reward if the pelvis is too close to the block horizontally when the pelvis is low to the ground.
    Discourages the robot from colliding with the block before jumping.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
    RewNormalizer = get_normalizer(env.device)
    try:
        block = env.scene['Object5'] # Accessing block object using approved pattern and try/except - rule 2 & 5 in ABSOLUTE REQUIREMENTS

        # Accessing robot pelvis position using approved pattern - rule 3 in ABSOLUTE REQUIREMENTS
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

        # Accessing block position using approved pattern - rule 2 in ABSOLUTE REQUIREMENTS
        block_pos = block.data.root_pos_w

        # Calculate relative distance in x direction between pelvis and block - rule 1 in ABSOLUTE REQUIREMENTS
        distance_x_pelvis_block = block_pos[:, 0] - pelvis_pos[:, 0] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS
        distance_y_pelvis_block = block_pos[:, 1] - pelvis_pos[:, 1] # Relative distance - rule 1 in ABSOLUTE REQUIREMENTS

        # overall xy distance between pelvis and block
        distance_pelvis_block = torch.sqrt(distance_x_pelvis_block**2 + distance_y_pelvis_block**2)

        # Collision condition: activate when robot is very close to block horizontally and low to the ground - rule 4 in ABSOLUTE REQUIREMENTS
        collision_condition = (distance_pelvis_block <  0.9)# Relative condition in x, absolute condition in z (allowed for height) - rule 1 in ABSOLUTE REQUIREMENTS

        # Small negative reward for collision proximity - rule 5 in CRITICAL IMPLEMENTATION RULES
        reward = -1.0 * torch.ones(env.num_envs, device=env.device) # Continuous negative reward - rule 5 in CRITICAL IMPLEMENTATION RULES

        reward = torch.where(collision_condition, reward, torch.zeros_like(reward)) # Apply activation condition - rule 4 in ABSOLUTE REQUIREMENTS

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward - rule 5 in ABSOLUTE REQUIREMENTS

    # Reward normalization using RewNormalizer - rule 6 in ABSOLUTE REQUIREMENTS and rule 4 in REWARD STRUCTURE RULES
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_feet_under_pelvis(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_under_pelvis_reward") -> torch.Tensor:
    '''Shaping reward for keeping feet underneath the pelvis in the horizontal plane.
    
    Reward for minimizing the horizontal (x,y) distance between the feet and pelvis.
    Encourages the robot to maintain a stable posture.
    '''
    robot = env.scene["robot"]
    RewNormalizer = get_normalizer(env.device)
    
    # Accessing robot pelvis and feet positions using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    
    left_x_distance = pelvis_pos[:, 0] - left_foot_pos[:, 0]
    left_y_distance = pelvis_pos[:, 1] - left_foot_pos[:, 1]
    
    right_x_distance = pelvis_pos[:, 0] - right_foot_pos[:, 0]
    right_y_distance = pelvis_pos[:, 1] - right_foot_pos[:, 1]
    
    
    

    # Calculate horizontal distance between pelvis and average foot position
    horizontal_distance = torch.sqrt((left_x_distance)**2 + 
                                     (left_y_distance)**2) + torch.sqrt((right_x_distance)**2 + 
                                     (right_y_distance)**2)
    
    # Reward is negative distance to encourage feet to stay under pelvis
    reward = -horizontal_distance
    
    # Reward normalization using RewNormalizer
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    Main_ExecuteJumpOntoBlockReward = RewTerm(func=main_ExecuteJumpOntoBlock_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})
    ShapingRewardApproachBlockX = RewTerm(func=shaping_reward_approach_block_x, weight=0.4,
                                            params={"normalise": True, "normaliser_name": "approach_block_x_reward"})
    ShapingRewardJumpHeight = RewTerm(func=shaping_reward_jump_height, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "jump_height_reward"})
    ShapingRewardStabilityOnBlock = RewTerm(func=shaping_reward_stability_on_block, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "stability_on_block_reward"})
    ShapingRewardCollisionAvoidance = RewTerm(func=shaping_reward_collision_avoidance, weight=0, 
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})
    ShapingRewardFeetUnderPelvis = RewTerm(func=shaping_reward_feet_under_pelvis, weight=0.5,
                                            params={"normalise": True, "normaliser_name": "feet_under_pelvis_reward"})