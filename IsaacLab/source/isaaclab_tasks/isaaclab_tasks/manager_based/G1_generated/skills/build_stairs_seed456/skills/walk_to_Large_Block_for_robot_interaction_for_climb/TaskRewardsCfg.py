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


def walk_to_large_block_position_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_large_block_position_reward") -> torch.Tensor:
    """
    Primary reward for guiding the robot's pelvis to a target position directly in front of the Large Block (Object3),
    ready for a jump. This reward combines alignment in x, y, and z dimensions.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object using approved patterns.
    robot = env.scene["robot"]
    object_large_block = env.scene['Object3']

    # Get robot pelvis position using approved pattern.
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # Hardcode Large Block dimensions from the object configuration.
    # Object3: "Large Block for robot interaction" -> x=1m y=1m z=0.9m
    large_block_x_dim = 1.0
    large_block_y_dim = 1.0
    # Define target clearance and stable standing height. These are relative values.
    target_clearance_x = 0.5  # Desired distance from the front face of the block
    target_pelvis_z = 0.7     # Stable standing height for the robot

    # Calculate target position relative to the Large Block's root position.
    # The block's root is assumed to be at its center.
    # Target X: block_center_x - (block_x_dim / 2) - target_clearance_x
    # This ensures the robot is positioned in front of the block's face.
    target_x_pos = object_large_block.data.root_pos_w[:, 0] - (large_block_x_dim / 2.0) - target_clearance_x
    # Target Y: Center Y of the block.
    target_y_pos = object_large_block.data.root_pos_w[:, 1]

    # Calculate rewards for each dimension. Using negative absolute difference for continuous reward.
    # Reward is maximized when the robot's pelvis is exactly at the target position.
    reward_x = -torch.abs(robot_pelvis_pos_x - target_x_pos)
    reward_y = -torch.abs(robot_pelvis_pos_y - target_y_pos)
    reward_z = -torch.abs(robot_pelvis_pos_z - target_pelvis_z)

    # Combine individual dimension rewards.
    reward = reward_x + reward_y + reward_z

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def feet_on_ground_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_on_ground_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to keep its feet on the ground, promoting stable walking.
    It penalizes if either foot is too high off the ground.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access robot using approved pattern.
    robot = env.scene["robot"]

    # Get foot positions using approved patterns.
    robot_left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    robot_right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    robot_left_foot_pos_z = robot.data.body_pos_w[:, robot_left_foot_idx][:, 2]
    robot_right_foot_pos_z = robot.data.body_pos_w[:, robot_right_foot_idx][:, 2]

    # Define ground level and maximum allowed foot height.
    ground_level_z = 0.0  # Assuming ground is at z=0
    max_foot_height = 0.1 # Maximum allowed foot height off the ground for walking

    # Calculate penalty for each foot.
    # The penalty is based on the absolute distance from the ground, but only applied if the foot is too high.
    # The reward should be negative when the foot is high, and 0 otherwise.
    # The original skeleton had a slight logical error in applying the penalty.
    # It should be a penalty (negative value) when the condition is met, and 0 otherwise.
    # The absolute difference from ground_level_z is a good measure of height.
    # We want to penalize if robot_left_foot_pos_z > max_foot_height.
    # A simple penalty could be -(robot_left_foot_pos_z - max_foot_height) if it's above, or just -robot_left_foot_pos_z.
    # The prompt's skeleton used -torch.abs(robot_left_foot_pos_z - ground_level_z) as the base penalty.
    # Let's stick to the skeleton's base penalty and apply it conditionally.
    # If foot is high, penalty is -abs(current_z - ground_z). If not high, reward is 0.
    
    # Calculate the base penalty for each foot (negative of its height above ground).
    # This value is always negative or zero.
    base_penalty_left_foot = -robot_left_foot_pos_z
    base_penalty_right_foot = -robot_right_foot_pos_z

    # Apply penalty only if the foot's z-position exceeds the maximum allowed height.
    # This makes the reward conditional, penalizing only when the feet are lifted too much.
    condition_left_foot_high = robot_left_foot_pos_z > max_foot_height
    condition_right_foot_high = robot_right_foot_pos_z > max_foot_height

    # If condition is true, apply the base penalty. Otherwise, reward is 0.0.
    reward_left_foot_on_ground = torch.where(condition_left_foot_high, base_penalty_left_foot, torch.tensor(0.0, device=env.device))
    reward_right_foot_on_ground = torch.where(condition_right_foot_high, base_penalty_right_foot, torch.tensor(0.0, device=env.device))

    # Combine penalties.
    reward = reward_left_foot_on_ground + reward_right_foot_on_ground

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_blocks_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_blocks_reward") -> torch.Tensor:
    """
    Shaping reward that encourages collision avoidance between the robot's pelvis and all blocks (Object1, Object2, Object3).
    It penalizes the robot if its pelvis gets too close to any block's bounding box.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access robot and objects using approved patterns.
    robot = env.scene["robot"]
    object_small_block = env.scene['Object1']
    object_medium_block = env.scene['Object2']
    object_large_block = env.scene['Object3']

    # Get robot pelvis position using approved pattern.
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Hardcode block dimensions from the object configuration.
    # Small Block: x=1m y=1m z=0.3m
    # Medium Block: x=1m y=1m z=0.6m
    # Large Block: x=1m y=1m z=0.9m
    blocks_data = [
        (object_small_block, 1.0, 1.0, 0.3),
        (object_medium_block, 1.0, 1.0, 0.6),
        (object_large_block, 1.0, 1.0, 0.9)
    ]

    collision_penalty = torch.zeros_like(robot_pelvis_pos[:, 0]) # Initialize penalty tensor for batch environments
    safe_distance_threshold = 0.2 # Minimum safe distance from block surface

    for block_obj, block_x_dim, block_y_dim, block_z_dim in blocks_data:
        block_center_x = block_obj.data.root_pos_w[:, 0]
        block_center_y = block_obj.data.root_pos_w[:, 1]
        block_center_z = block_obj.data.root_pos_w[:, 2]

        # Calculate the signed distance from the pelvis to the *surface* of the block in each dimension.
        # A negative value means overlap, a positive value means outside.
        dist_x = torch.abs(robot_pelvis_pos[:, 0] - block_center_x) - (block_x_dim / 2.0)
        dist_y = torch.abs(robot_pelvis_pos[:, 1] - block_center_y) - (block_y_dim / 2.0)
        dist_z = torch.abs(robot_pelvis_pos[:, 2] - block_center_z) - (block_z_dim / 2.0)

        # Find the minimum distance to any surface of the current block.
        # This represents how close the pelvis is to "colliding" with the block.
        min_dist_to_surface = torch.min(torch.stack([dist_x, dist_y, dist_z]), dim=0).values

        # Apply a penalty if the minimum distance to the surface is less than the safe_distance_threshold.
        # The penalty is stronger the closer the robot is to the block (or if it's inside).
        # The reward is negative when too close, and 0 otherwise.
        current_block_penalty = torch.where(min_dist_to_surface < safe_distance_threshold,
                                            -(safe_distance_threshold - min_dist_to_surface),
                                            torch.tensor(0.0, device=env.device))

        collision_penalty += current_block_penalty

    reward = collision_penalty

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for positioning the robot in front of the Large Block.
    WalkToLargeBlockPositionReward = RewTerm(func=walk_to_large_block_position_reward, weight=1.0,
                                             params={"normalise": True, "normaliser_name": "walk_to_large_block_position_reward"})

    # Shaping reward for keeping the robot's feet on the ground during walking.
    FeetOnGroundReward = RewTerm(func=feet_on_ground_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "feet_on_ground_reward"})

    # Shaping reward for avoiding collisions with all blocks.
    CollisionAvoidanceBlocksReward = RewTerm(func=collision_avoidance_blocks_reward, weight=0.6,
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_blocks_reward"})