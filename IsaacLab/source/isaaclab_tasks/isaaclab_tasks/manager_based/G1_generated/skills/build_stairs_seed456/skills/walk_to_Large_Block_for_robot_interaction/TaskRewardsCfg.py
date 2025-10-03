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


def walk_to_large_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_large_block_main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_Large_Block_for_robot_interaction skill.
    Rewards the robot for moving its pelvis to the side of the Large Block (Object3)
    that is opposite to the Medium Block (Object2), at a target horizontal distance,
    and for maintaining a stable upright posture. This positions the robot to push
    the Large Block towards the Medium Block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL RULE: Access objects directly using their scene names (Object1, Object2, Object3)
    large_block = env.scene['Object3']
    large_block_pos = large_block.data.root_pos_w # CORRECT: Accessing object position using approved pattern
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w

    # Access the required robot part(s)
    # CRITICAL RULE: Access robot parts using their names and indices
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern

    # Target horizontal distance from pelvis to block center (e.g., 0.8m for pushing)
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds. This is a target distance, not a position.
    target_horizontal_distance = 0.8 # meters

    # Compute the desired target position on the opposite side of the Large Block relative to the Medium Block.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    pelvis_xy = pelvis_pos[:, :2]
    large_xy = large_block_pos[:, :2]
    medium_xy = medium_block_pos[:, :2]
    # Direction pointing away from the Medium Block, anchored at the Large Block
    away_vec = large_xy - medium_xy
    away_norm = torch.norm(away_vec, dim=1, keepdim=True).clamp_min(1e-6)
    away_dir = away_vec / away_norm
    desired_xy = large_xy + away_dir * target_horizontal_distance

    # Reward is negative distance from pelvis to desired opposite-side target position
    # CRITICAL RULE: Rewards should be continuous and positive where possible. This is continuous.
    position_error = torch.norm(pelvis_xy - desired_xy, dim=1)
    side_position_reward = -position_error

    # Additional shaping to avoid local minima on the wrong side of the block
    rel_xy = pelvis_xy - large_xy
    proj_along = torch.sum(rel_xy * away_dir, dim=1)  # signed distance along desired push direction
    proj_norm = proj_along / (target_horizontal_distance + 1e-6)
    alignment_reward = torch.tanh(proj_norm)  # in [-1, 1], positive when on the correct side

    lateral_vec = rel_xy - away_dir * proj_along.unsqueeze(1)
    lateral_error = torch.norm(lateral_vec, dim=1)
    lateral_penalty = -0.1 * lateral_error

    # Hinge penalty when on the wrong side of the large block (proj_along < 0)
    wrong_side_penalty = -0.5 * torch.relu(-proj_along)

    # Target pelvis z-position for standing (0.7m).
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds. This is a target height, not a position.
    target_pelvis_z = 0.7 # meters
    pelvis_z_deviation = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)
    z_reward = -pelvis_z_deviation

    # Combine rewards mirroring the successful medium-block implementation
    reward = side_position_reward # + 0.5 * alignment_reward + lateral_penalty + wrong_side_penalty + 0.2 * z_reward

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    This reward encourages the robot to avoid collisions with all objects (Object1, Object2, Object3).
    It applies a negative reward if any specified robot part gets too close to any of the blocks.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL RULE: Access objects directly using their scene names
    small_block = env.scene['Object1']
    medium_block = env.scene['Object2']
    large_block = env.scene['Object3']

    # Object dimensions (Small: x=1m y=1m z=0.3m, Medium: x=1m y=1m z=0.6m, Large: x=1m y=1m z=0.9m)
    # CRITICAL RULE: Hardcode object dimensions from the configuration
    # Using the largest dimension for a conservative collision radius for simplicity.
    # A general clearance threshold for proximity penalty.
    clearance_threshold = 0.6 # A bit more than half the largest block dimension (0.9/2 = 0.45) + some buffer

    # Access the required robot part(s)
    # CRITICAL RULE: Access robot parts using their names and indices
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_palm_idx = robot.body_names.index('left_palm_link')
    right_palm_idx = robot.body_names.index('right_palm_link')
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')

    # Combine positions of relevant robot parts into a single tensor for batch processing
    # CRITICAL RULE: All operations must work with batched environments
    robot_parts_pos = torch.cat([
        robot.data.body_pos_w[:, pelvis_idx].unsqueeze(1),
        robot.data.body_pos_w[:, left_palm_idx].unsqueeze(1),
        robot.data.body_pos_w[:, right_palm_idx].unsqueeze(1),
        robot.data.body_pos_w[:, left_ankle_idx].unsqueeze(1),
        robot.data.body_pos_w[:, right_ankle_idx].unsqueeze(1)
    ], dim=1) # Shape: [num_envs, num_parts, 3]

    objects = [small_block, medium_block, large_block]
    collision_reward = torch.zeros_like(env.episode_length_buf, dtype=torch.float32) # Initialize reward tensor

    for obj in objects:
        obj_pos = obj.data.root_pos_w # Shape: [num_envs, 3]

        # Calculate Euclidean distance from each robot part to the object's center
        # CRITICAL RULE: Use relative distances
        # Expand obj_pos to match robot_parts_pos dimensions for broadcasting
        distances_to_obj = torch.norm(robot_parts_pos - obj_pos.unsqueeze(1), dim=2) # Shape: [num_envs, num_parts]

        # Find the minimum distance from any robot part to the object center for each environment
        min_dist_to_obj = torch.min(distances_to_obj, dim=1).values # Shape: [num_envs]

        # Apply negative reward if too close
        # CRITICAL RULE: Rewards should be continuous
        collision_condition = (min_dist_to_obj < clearance_threshold)
        # Penalize more as distance decreases below threshold
        collision_reward += torch.where(collision_condition, - (clearance_threshold - min_dist_to_obj) * 2.0, 0.0)

    reward = collision_reward

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def posture_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain an upright and stable posture.
    It penalizes large deviations of the pelvis's z-height from a target standing height
    and encourages the feet to be close to the ground (z=0).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    # CRITICAL RULE: Access robot parts using their names and indices
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')

    # Access robot part positions
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2]
    left_ankle_pos_z = robot.data.body_pos_w[:, left_ankle_idx, 2]
    right_ankle_pos_z = robot.data.body_pos_w[:, right_ankle_idx, 2]

    # Target pelvis height for standing
    target_pelvis_z = 0.7 # A reasonable standing height for the pelvis

    # Penalize deviation from target pelvis height
    # CRITICAL RULE: Rewards should be continuous and negative for deviation
    pelvis_height_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Penalize feet being too far from the ground (z=0)
    # A small offset (e.g., 0.05m) can be used to account for foot thickness/ground contact.
    ground_clearance = 0.05 # Target Z-height for feet to be considered on the ground
    feet_on_ground_reward = -torch.abs(left_ankle_pos_z - ground_clearance) - torch.abs(right_ankle_pos_z - ground_clearance)

    reward = pelvis_height_reward + feet_on_ground_reward

    # CRITICAL RULE: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    """
    Configuration for the reward terms for the walk_to_Large_Block_for_robot_interaction skill.
    """
    # Main reward: Encourages the robot to reach the target pushing position near the Large Block
    # CRITICAL RULE: Main reward weight should be around 1.0
    walk_to_large_block_main_reward = RewTerm(
        func=walk_to_large_block_main_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "walk_to_large_block_main_reward"}
    )

    # Shaping Reward 1: Penalizes collisions or close proximity to any of the blocks
    # CRITICAL RULE: Shaping rewards typically have lower weights (< 1.0)
    collision_avoidance_reward = RewTerm(
        func=collision_avoidance_reward,
        weight=0.0, # As per reward design plan
        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"}
    )

    # Shaping Reward 2: Encourages stable and upright posture
    # CRITICAL RULE: Shaping rewards typically have lower weights (< 1.0)
    posture_stability_reward = RewTerm(
        func=posture_stability_reward,
        weight=0.2, # A reasonable weight for posture stability
        params={"normalise": True, "normaliser_name": "posture_stability_reward"}
    )