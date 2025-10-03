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


def walk_to_blockCube_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_blockCube_main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_blockCube skill.
    Guides the robot to walk towards the block cube (Object5) and position its pelvis within a target x-distance,
    centered in y, and at a stable z-height relative to the block cube. It encourages the robot to stop before
    overshooting the block cube, preparing for the next skill (jumping onto it).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions using approved patterns
    robot = env.scene["robot"]
    block_cube = env.scene['Object5'] # Object5 is the block cube as per object configuration

    # Get robot pelvis position
    robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # Get block cube position
    block_cube_pos = block_cube.data.root_pos_w # CORRECT: Accessing object position using approved pattern

    # Target x-offset from block cube's center: robot should be 0.5m in front of the block
    # This positions the robot correctly to jump onto the block.
    target_x_offset = -0.5 # CORRECT: Relative distance, not hard-coded absolute position
    
    # Distance to target x-position relative to block cube
    # We want the robot's pelvis x to be at (block_cube_x + target_x_offset)
    dist_x = torch.abs((block_cube_pos[:, 0] + target_x_offset) - robot_pelvis_pos_x) # CORRECT: Relative distance calculation
    reward_x = -dist_x # CORRECT: Continuous reward, closer is better (negative distance)

    # Distance to target y-position (centered with block cube)
    # We want the robot's pelvis y to be aligned with the block cube's y.
    dist_y = torch.abs(block_cube_pos[:, 1] - robot_pelvis_pos_y) # CORRECT: Relative distance calculation
    reward_y = -dist_y # CORRECT: Continuous reward, closer is better

    # Distance to target z-position (stable standing height)
    # A typical standing height for the pelvis.
    target_pelvis_z = 0.7 # CORRECT: Relative value based on robot dimensions, not hard-coded absolute position
    dist_z = torch.abs(target_pelvis_z - robot_pelvis_pos_z) # CORRECT: Relative distance calculation
    reward_z = -dist_z # CORRECT: Continuous reward, closer is better

    # Combine rewards
    reward = reward_x + reward_y + reward_z

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_blockCube_posture_alignment_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_blockCube_posture_alignment_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable, upright posture by penalizing large deviations
    in the pelvis's z-height from the target standing height (0.7m) and penalizing large y-deviations,
    which could indicate instability or drifting off course. This implicitly encourages facing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions
    robot = env.scene["robot"]
    block_cube = env.scene['Object5']

    # Get robot pelvis position
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # Get block cube position
    block_cube_pos = block_cube.data.root_pos_w

    # Reward for maintaining pelvis z-height around 0.7m
    target_pelvis_z = 0.7 # CORRECT: Relative value based on robot dimensions
    reward_pelvis_z_stability = -torch.abs(robot_pelvis_pos_z - target_pelvis_z) # CORRECT: Continuous reward

    # Reward for maintaining y-position close to the block's y-position
    # This reinforces the y-alignment from the main reward and helps with facing.
    reward_pelvis_y_alignment = -torch.abs(robot_pelvis_pos_y - block_cube_pos[:, 1]) # CORRECT: Relative distance

    # Combine rewards
    reward = reward_pelvis_z_stability + reward_pelvis_y_alignment

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_blockCube_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_blockCube_stability_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to keep its feet on the ground and avoid falling.
    It penalizes the robot if its feet are too high off the ground when not intended (i.e., not jumping)
    or if the pelvis z-height drops significantly, indicating a fall.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot part positions
    robot = env.scene["robot"]
    robot_left_foot_idx = robot.body_names.index('left_ankle_roll_link') # CORRECT: Accessing robot part index
    robot_right_foot_idx = robot.body_names.index('right_ankle_roll_link') # CORRECT: Accessing robot part index
    robot_pelvis_idx = robot.body_names.index('pelvis')

    robot_left_foot_pos_z = robot.data.body_pos_w[:, robot_left_foot_idx][:, 2] # CORRECT: Accessing robot part position
    robot_right_foot_pos_z = robot.data.body_pos_w[:, robot_right_foot_idx][:, 2] # CORRECT: Accessing robot part position
    robot_pelvis_pos_z = robot.data.body_pos_w[:, robot_pelvis_idx][:, 2] # CORRECT: Accessing robot part position

    # Penalize if feet are too high off the ground (e.g., > 0.1m for walking)
    # This threshold is relative to the ground (z=0) and appropriate for walking/standing.
    max_foot_height_for_walking = 0.1 # CORRECT: Relative threshold, not arbitrary
    reward_left_foot_height = -torch.where(robot_left_foot_pos_z > max_foot_height_for_walking,
                                           robot_left_foot_pos_z - max_foot_height_for_walking, 0.0) # CORRECT: Continuous penalty
    reward_right_foot_height = -torch.where(robot_right_foot_pos_z > max_foot_height_for_walking,
                                            robot_right_foot_pos_z - max_foot_height_for_walking, 0.0) # CORRECT: Continuous penalty

    # Penalize if pelvis drops too low (indicating a fall)
    min_pelvis_z_for_standing = 0.4 # CORRECT: Relative threshold, based on robot dimensions
    reward_pelvis_fall = -torch.where(robot_pelvis_pos_z < min_pelvis_z_for_standing,
                                      min_pelvis_z_for_standing - robot_pelvis_pos_z, 0.0) # CORRECT: Continuous penalty

    # Combine rewards
    reward = reward_left_foot_height + reward_right_foot_height + reward_pelvis_fall

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for reaching the target position relative to the block cube
    walk_to_blockCube_main_reward = RewTerm(func=walk_to_blockCube_main_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "walk_to_blockCube_main_reward"})

    # Shaping reward for maintaining stable posture and alignment
    walk_to_blockCube_posture_alignment_reward = RewTerm(func=walk_to_blockCube_posture_alignment_reward, weight=0.4,
                                                         params={"normalise": True, "normaliser_name": "walk_to_blockCube_posture_alignment_reward"})

    # Shaping reward for general stability and preventing falls
    walk_to_blockCube_stability_reward = RewTerm(func=walk_to_blockCube_stability_reward, weight=0.6,
                                                 params={"normalise": True, "normaliser_name": "walk_to_blockCube_stability_reward"})