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

# Hardcoded dimensions for the Large Block (Object3) based on the object configuration
# "Object3": "Large Block for robot interaction"
# Large Block measures x=1m, y=1m, z=0.9m
LARGE_BLOCK_X_SIZE = 1.0
LARGE_BLOCK_Y_SIZE = 1.0
LARGE_BLOCK_HEIGHT = 0.9

def ascend_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "ascend_large_block_reward") -> torch.Tensor:
    """
    Primary reward for guiding the robot to approach the Large Block, jump onto its top surface,
    and land stably. It combines horizontal approach with vertical ascent and stable landing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions using approved patterns
    robot = env.scene["robot"]
    object_large_block = env.scene['Object3'] # Object3 is the Large Block

    # Get robot part indices and positions
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2.0

    # Get Large Block's root position
    large_block_root_pos = object_large_block.data.root_pos_w

    # Calculate target positions relative to the Large Block
    # Target horizontal position is the center of the large block
    target_x = large_block_root_pos[:, 0]
    target_y = large_block_root_pos[:, 1]
    # Target Z for feet on top surface of the block
    target_z_on_block = large_block_root_pos[:, 2] + LARGE_BLOCK_HEIGHT / 2.0
    # Target Z for pelvis when standing stably on the block (approx. 0.7m above block surface)
    pelvis_stable_z_target = target_z_on_block + 0.7

    # Phase 1: Approach horizontally
    # Reward for reducing horizontal distance to the block's center
    horizontal_distance_x = torch.abs(pelvis_pos[:, 0] - target_x)
    horizontal_distance_y = torch.abs(pelvis_pos[:, 1] - target_y)
    approach_reward = - (horizontal_distance_x + horizontal_distance_y) # Continuous negative reward for distance

    # Phase 2 & 3: Jump and land on block
    # Condition for being "over" the block horizontally (pelvis within block's x,y bounds)
    is_over_block_x = (pelvis_pos[:, 0] > target_x - LARGE_BLOCK_X_SIZE / 2.0) & \
                      (pelvis_pos[:, 0] < target_x + LARGE_BLOCK_X_SIZE / 2.0)
    is_over_block_y = (pelvis_pos[:, 1] > target_y - LARGE_BLOCK_Y_SIZE / 2.0) & \
                      (pelvis_pos[:, 1] < target_y + LARGE_BLOCK_Y_SIZE / 2.0)
    is_over_block_horizontally = is_over_block_x & is_over_block_y

    # Reward for feet being at or above the block's top surface, encouraging precise landing
    vertical_landing_reward = -torch.abs(avg_foot_pos_z - target_z_on_block) # Continuous negative reward for vertical distance

    # Reward for pelvis stability once on the block, encouraging standing upright
    pelvis_stability_reward = -torch.abs(pelvis_pos[:, 2] - pelvis_stable_z_target) # Continuous negative reward for pelvis height deviation

    # Combine rewards based on phases using torch.where for batch compatibility
    # If not yet over the block horizontally, prioritize horizontal approach
    # Once over the block horizontally, prioritize vertical landing and stability
    reward = torch.where(
        is_over_block_horizontally,
        vertical_landing_reward * 0.7 + pelvis_stability_reward * 0.3, # Weighted for landing and stability
        approach_reward # Prioritize horizontal approach
    )

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to prevent the robot's lower body parts (knees, feet) from colliding with the Large Block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions using approved patterns
    robot = env.scene["robot"]
    object_large_block = env.scene['Object3'] # Object3 is the Large Block

    # Get robot part indices and positions
    left_knee_idx = robot.body_names.index('left_knee_link')
    right_knee_idx = robot.body_names.index('right_knee_link')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Get Large Block's root position
    large_block_root_pos = object_large_block.data.root_pos_w

    # Calculate block boundaries based on its root position and hardcoded dimensions
    block_center_x = large_block_root_pos[:, 0]
    block_center_y = large_block_root_pos[:, 1]
    block_bottom_z = large_block_root_pos[:, 2] - LARGE_BLOCK_HEIGHT / 2.0
    block_top_z = large_block_root_pos[:, 2] + LARGE_BLOCK_HEIGHT / 2.0

    # Define a small clearance buffer around the block to penalize getting too close
    clearance_buffer = 0.1

    # Check for collision for left knee
    is_left_knee_colliding_x = (left_knee_pos[:, 0] > block_center_x - LARGE_BLOCK_X_SIZE / 2.0 - clearance_buffer) & \
                              (left_knee_pos[:, 0] < block_center_x + LARGE_BLOCK_X_SIZE / 2.0 + clearance_buffer)
    is_left_knee_colliding_y = (left_knee_pos[:, 1] > block_center_y - LARGE_BLOCK_Y_SIZE / 2.0 - clearance_buffer) & \
                              (left_knee_pos[:, 1] < block_center_y + LARGE_BLOCK_Y_SIZE / 2.0 + clearance_buffer)
    is_left_knee_colliding_z = (left_knee_pos[:, 2] > block_bottom_z - clearance_buffer) & \
                              (left_knee_pos[:, 2] < block_top_z + clearance_buffer)
    left_knee_collision = is_left_knee_colliding_x & is_left_knee_colliding_y & is_left_knee_colliding_z

    # Check for collision for right knee
    is_right_knee_colliding_x = (right_knee_pos[:, 0] > block_center_x - LARGE_BLOCK_X_SIZE / 2.0 - clearance_buffer) & \
                               (right_knee_pos[:, 0] < block_center_x + LARGE_BLOCK_X_SIZE / 2.0 + clearance_buffer)
    is_right_knee_colliding_y = (right_knee_pos[:, 1] > block_center_y - LARGE_BLOCK_Y_SIZE / 2.0 - clearance_buffer) & \
                               (right_knee_pos[:, 1] < block_center_y + LARGE_BLOCK_Y_SIZE / 2.0 + clearance_buffer)
    is_right_knee_colliding_z = (right_knee_pos[:, 2] > block_bottom_z - clearance_buffer) & \
                               (right_knee_pos[:, 2] < block_top_z + clearance_buffer)
    right_knee_collision = is_right_knee_colliding_x & is_right_knee_colliding_y & is_right_knee_colliding_z

    # Check for collision for left foot (only if not intended to be on top, i.e., below the top surface)
    is_left_foot_colliding_x = (left_foot_pos[:, 0] > block_center_x - LARGE_BLOCK_X_SIZE / 2.0 - clearance_buffer) & \
                              (left_foot_pos[:, 0] < block_center_x + LARGE_BLOCK_X_SIZE / 2.0 + clearance_buffer)
    is_left_foot_colliding_y = (left_foot_pos[:, 1] > block_center_y - LARGE_BLOCK_Y_SIZE / 2.0 - clearance_buffer) & \
                              (left_foot_pos[:, 1] < block_center_y + LARGE_BLOCK_Y_SIZE / 2.0 + clearance_buffer)
    is_left_foot_colliding_z = (left_foot_pos[:, 2] > block_bottom_z - clearance_buffer) & \
                              (left_foot_pos[:, 2] < block_top_z + clearance_buffer)
    # Add condition to exclude feet that are already on top of the block
    left_foot_collision = is_left_foot_colliding_x & is_left_foot_colliding_y & is_left_foot_colliding_z & \
                          (left_foot_pos[:, 2] < block_top_z - 0.05) # 0.05m buffer to allow feet to be slightly above bottom of top surface

    # Check for collision for right foot (only if not intended to be on top)
    is_right_foot_colliding_x = (right_foot_pos[:, 0] > block_center_x - LARGE_BLOCK_X_SIZE / 2.0 - clearance_buffer) & \
                               (right_foot_pos[:, 0] < block_center_x + LARGE_BLOCK_X_SIZE / 2.0 + clearance_buffer)
    is_right_foot_colliding_y = (right_foot_pos[:, 1] > block_center_y - LARGE_BLOCK_Y_SIZE / 2.0 - clearance_buffer) & \
                               (right_foot_pos[:, 1] < block_center_y + LARGE_BLOCK_Y_SIZE / 2.0 + clearance_buffer)
    is_right_foot_colliding_z = (right_foot_pos[:, 2] > block_bottom_z - clearance_buffer) & \
                               (right_foot_pos[:, 2] < block_top_z + clearance_buffer)
    # Add condition to exclude feet that are already on top of the block
    right_foot_collision = is_right_foot_colliding_x & is_right_foot_colliding_y & is_right_foot_colliding_z & \
                           (right_foot_pos[:, 2] < block_top_z - 0.05)

    # Negative reward for any collision
    # Using -1.0 for a clear penalty when collision occurs
    reward = torch.where(left_knee_collision | right_knee_collision | left_foot_collision | right_foot_collision, -1.0, 0.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_stability_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_shaping_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable, upright posture (pelvis at a reasonable height)
    when not actively on the Large Block. This helps prevent the robot from crawling or falling.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions using approved patterns
    robot = env.scene["robot"]
    object_large_block = env.scene['Object3'] # Object3 is the Large Block

    # Get robot part indices and positions
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2.0

    # Get Large Block's root position
    large_block_root_pos = object_large_block.data.root_pos_w
    block_top_z = large_block_root_pos[:, 2] + LARGE_BLOCK_HEIGHT / 2.0

    # Target pelvis height for standing, relative to ground (0.7m is a typical standing height for this robot)
    target_pelvis_z_standing = 0.7

    # Condition: Robot is not yet on the block (feet clearly below block top surface)
    # Using a small buffer (0.05m) to ensure the robot is truly off the block
    is_not_on_block = (avg_foot_pos_z < block_top_z - 0.05)

    # Reward for maintaining pelvis height when not on the block
    # This reward is only active when the robot is not on the block, otherwise it's 0.0
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_standing) # Continuous negative reward for height deviation

    reward = torch.where(is_not_on_block, pelvis_height_reward, torch.tensor(0.0, device=env.device))

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
    """
    Configuration for the reward terms used in the Ascend_Large_Block_for_robot_interaction skill.
    """
    # Primary reward for ascending the large block
    AscendLargeBlockReward = RewTerm(func=ascend_large_block_reward, weight=1.0,
                                     params={"normalise": True, "normaliser_name": "ascend_large_block_reward"})

    # Shaping reward for collision avoidance with the large block
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.5,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining pelvis stability when not on the block
    PelvisStabilityShapingReward = RewTerm(func=pelvis_stability_shaping_reward, weight=0.3,
                                           params={"normalise": True, "normaliser_name": "pelvis_stability_shaping_reward"})