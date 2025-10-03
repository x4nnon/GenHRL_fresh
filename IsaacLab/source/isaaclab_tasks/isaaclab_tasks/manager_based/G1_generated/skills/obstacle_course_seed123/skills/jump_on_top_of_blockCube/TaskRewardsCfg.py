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

def main_jump_on_top_of_blockCube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'jump_on_top_of_blockCube' skill.
    This reward encourages the robot to approach the block, gain height to jump over it,
    and then land stably on its top surface.
    """
    # Get normalizer instance (MANDATORY NORMALIZATION)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts (APPROVED ACCESS PATTERNS)
    block_cube = env.scene['Object5'] # Accessing object using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern

    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    block_cube_pos = block_cube.data.root_pos_w # Accessing object position using approved pattern

    # Block dimensions (hardcoded from object configuration: 0.5m cubed) (HARDCODED DIMENSIONS)
    block_height = 0.5
    block_half_size = block_height / 2 # 0.25

    # Block's top surface z-coordinate (relative to its root position)
    block_top_z = block_cube_pos[:, 2] + block_half_size

    # Average foot position for combined feet measurements
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # 1. Horizontal Approach Reward: Encourages the robot to move towards the block's center horizontally.
    # Uses relative distances between pelvis and block center. (RELATIVE DISTANCES)
    pelvis_dist_x = torch.abs(pelvis_pos[:, 0] - block_cube_pos[:, 0])
    pelvis_dist_y = torch.abs(pelvis_pos[:, 1] - block_cube_pos[:, 1])
    horizontal_approach_reward = -(pelvis_dist_x + pelvis_dist_y) # Continuous negative reward, minimized as robot gets closer. (CONTINUOUS REWARD)

    # 2. Feet Above Block Reward (Jump Encouragement): Rewards gaining height when near the block.
    # This helps bridge the gap between approaching and landing.
    is_near_block_horizontally = (pelvis_dist_x < 1.0) & (pelvis_dist_y < 1.0) # Within 1m horizontally of the block
    feet_above_block_reward = torch.where(
        is_near_block_horizontally & (avg_foot_pos_z > block_top_z),
        (avg_foot_pos_z - block_top_z) * 2.0, # Reward for feet being above block top, scaled.
        0.0
    )

    # 3. Landing and Stability Reward: Activated when feet are horizontally aligned and vertically close to the block top.
    horizontal_tolerance = 0.1 # Small buffer for landing
    feet_on_block_x_condition = (avg_foot_pos_x > block_cube_pos[:, 0] - block_half_size - horizontal_tolerance) & \
                                (avg_foot_pos_x < block_cube_pos[:, 0] + block_half_size + horizontal_tolerance)
    feet_on_block_y_condition = (avg_foot_pos_y > block_cube_pos[:, 1] - block_half_size - horizontal_tolerance) & \
                                (avg_foot_pos_y < block_cube_pos[:, 1] + block_half_size + horizontal_tolerance)
    feet_horizontally_aligned = feet_on_block_x_condition & feet_on_block_y_condition

    feet_to_block_top_z = avg_foot_pos_z - block_top_z
    feet_vertically_on_block = torch.abs(feet_to_block_top_z) < 0.1 # Within 0.1m of block top

    # Reward for feet being vertically aligned with the block's top surface
    feet_on_block_vertical_reward = -torch.abs(feet_to_block_top_z) * 10.0 # Strong penalty for vertical deviation

    # Target pelvis height when standing stably on the block (relative to block top)
    target_pelvis_z_on_block = block_top_z + 0.7 # Assuming 0.7m is a stable standing height for the robot's pelvis
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block) * 5.0 # Penalize deviation from target height

    # Combine landing and stability rewards, activated only when feet are properly positioned
    landing_and_stability_reward = torch.where(
        feet_horizontally_aligned & feet_vertically_on_block,
        feet_on_block_vertical_reward + pelvis_height_reward + 10.0, # Bonus for being on block and stable
        0.0
    )

    # Total primary reward (CONTINUOUS REWARD)
    reward = horizontal_approach_reward + feet_above_block_reward + landing_and_stability_reward

    # Mandatory reward normalization (MANDATORY NORMALIZATION)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to prevent the robot's body parts (pelvis, feet) from colliding with the sides of the block
    or falling off the block.
    """
    # Get normalizer instance (MANDATORY NORMALIZATION)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts (APPROVED ACCESS PATTERNS)
    block_cube = env.scene['Object5'] # Accessing object using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern

    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    block_cube_pos = block_cube.data.root_pos_w # Accessing object position using approved pattern

    # Block dimensions (hardcoded from object configuration: 0.5m cubed) (HARDCODED DIMENSIONS)
    block_half_size = 0.25
    block_top_z = block_cube_pos[:, 2] + block_half_size

    # Pelvis collision avoidance: Penalize if pelvis is too close to the sides of the block (not on top)
    pelvis_dist_x_to_block_center = torch.abs(pelvis_pos[:, 0] - block_cube_pos[:, 0]) # RELATIVE DISTANCES
    pelvis_dist_y_to_block_center = torch.abs(pelvis_pos[:, 1] - block_cube_pos[:, 1])

    # Define a safe zone around the block. If robot parts are too close to the sides, penalize.
    # A buffer of 0.1m for collision clearance.
    safe_distance_x = block_half_size + 0.1
    safe_distance_y = block_half_size + 0.1

    # Penalize if pelvis is horizontally within the block's bounds and below its top surface (i.e., hitting the side)
    pelvis_collision_penalty = torch.where(
        (pelvis_dist_x_to_block_center < safe_distance_x) &
        (pelvis_dist_y_to_block_center < safe_distance_y) &
        (pelvis_pos[:, 2] < block_top_z - 0.05), # If pelvis is below top surface (with a small margin)
        -10.0 * (1.0 - (pelvis_dist_x_to_block_center / safe_distance_x).clamp(0,1)) - \
        10.0 * (1.0 - (pelvis_dist_y_to_block_center / safe_distance_y).clamp(0,1)), # Continuous penalty based on how deep it is (CONTINUOUS REWARD)
        0.0
    )

    # Feet collision avoidance: Penalize if feet are outside the block's horizontal bounds when they are near the block's height.
    # This is crucial for preventing falling off.
    feet_near_block_height_condition = (torch.abs(left_foot_pos[:, 2] - block_top_z) < 0.2) | \
                                       (torch.abs(right_foot_pos[:, 2] - block_top_z) < 0.2) # If either foot is near block height (0.2m tolerance)

    # Check if feet are horizontally outside the block's top surface bounds (with a small tolerance)
    horizontal_off_block_tolerance = 0.05 # Small margin for being off the block
    left_foot_off_x = torch.abs(left_foot_pos[:, 0] - block_cube_pos[:, 0]) > block_half_size + horizontal_off_block_tolerance
    left_foot_off_y = torch.abs(left_foot_pos[:, 1] - block_cube_pos[:, 1]) > block_half_size + horizontal_off_block_tolerance
    right_foot_off_x = torch.abs(right_foot_pos[:, 0] - block_cube_pos[:, 0]) > block_half_size + horizontal_off_block_tolerance
    right_foot_off_y = torch.abs(right_foot_pos[:, 1] - block_cube_pos[:, 1]) > block_half_size + horizontal_off_block_tolerance

    feet_horizontally_off_block = (left_foot_off_x | left_foot_off_y) | (right_foot_off_x | right_foot_off_y)

    feet_collision_penalty = torch.where(
        feet_near_block_height_condition & feet_horizontally_off_block,
        -20.0, # Strong penalty for feet being off the block when at landing height (CONTINUOUS REWARD - although a step, it's a strong deterrent)
        0.0
    )

    reward = pelvis_collision_penalty + feet_collision_penalty

    # Mandatory reward normalization (MANDATORY NORMALIZATION)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_stability_on_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain an upright and stable posture
    (pelvis height) while its feet are on top of the block.
    """
    # Get normalizer instance (MANDATORY NORMALIZATION)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts (APPROVED ACCESS PATTERNS)
    block_cube = env.scene['Object5'] # Accessing object using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern

    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    block_cube_pos = block_cube.data.root_pos_w # Accessing object position using approved pattern

    # Block dimensions (hardcoded from object configuration: 0.5m cubed) (HARDCODED DIMENSIONS)
    block_half_size = 0.25
    block_top_z = block_cube_pos[:, 2] + block_half_size

    # Average foot position
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # Condition: Feet are on the block (horizontally and vertically within tolerance) (RELATIVE DISTANCES)
    horizontal_tolerance = 0.1
    feet_on_block_x_condition = (avg_foot_pos_x > block_cube_pos[:, 0] - block_half_size - horizontal_tolerance) & \
                                (avg_foot_pos_x < block_cube_pos[:, 0] + block_half_size + horizontal_tolerance)
    feet_on_block_y_condition = (avg_foot_pos_y > block_cube_pos[:, 1] - block_half_size - horizontal_tolerance) & \
                                (avg_foot_pos_y < block_cube_pos[:, 1] + block_half_size + horizontal_tolerance)
    feet_horizontally_aligned = feet_on_block_x_condition & feet_on_block_y_condition

    feet_vertically_on_block = torch.abs(avg_foot_pos_z - block_top_z) < 0.1 # Within 0.1m of block top

    is_on_block_condition = feet_horizontally_aligned & feet_vertically_on_block

    # Target pelvis height when standing on the block (relative to block top)
    target_pelvis_z = block_top_z + 0.7 # 0.7m above the block's top surface for stable standing

    # Reward for pelvis being at the target height, only active when on the block (CONTINUOUS REWARD)
    pelvis_height_on_block_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z) * 2.0 # Penalize deviation from target height

    reward = torch.where(is_on_block_condition, pelvis_height_on_block_reward, 0.0)

    # Mandatory reward normalization (MANDATORY NORMALIZATION)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for jumping on top of the block and stabilizing (PROPER WEIGHTS)
    MainJumpOnTopBlockCubeReward = RewTerm(func=main_jump_on_top_of_blockCube_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "main_jump_on_top_block_reward"})

    # Shaping reward for collision avoidance with the block and preventing falling off (PROPER WEIGHTS)
    CollisionAvoidanceBlockReward = RewTerm(func=collision_avoidance_block_reward, weight=0.6,
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_block_reward"})

    # Shaping reward for maintaining stable pelvis height when on the block (PROPER WEIGHTS)
    PelvisStabilityOnBlockReward = RewTerm(func=pelvis_stability_on_block_reward, weight=0.4,
                                            params={"normalise": True, "normaliser_name": "pelvis_stability_on_block_reward"})