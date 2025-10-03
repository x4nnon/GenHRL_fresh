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

def walk_to_Small_Block_for_push_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_block_reward") -> torch.Tensor:
    """
    Primary reward for the robot to walk to a stable standing position adjacent to the Small Block (Object1)
    to prepare for pushing it.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Requirement: Access objects directly using approved pattern
    object1_pos = object1.data.root_pos_w # Requirement: Access object positions using approved pattern

    # Access the required robot part(s)
    robot = env.scene["robot"] # Requirement: Access robot object using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Requirement: Access robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Requirement: Access robot part position using approved pattern

    # Object1 dimensions (from skill info: x=1m, y=1m, z=0.3m)
    # Requirement: Hardcode object dimensions from config, do not access from object directly
    object1_half_width_y = 0.5 # Half of Object1's Y dimension (1m / 2)
    object1_half_depth_x = 0.5 # Half of Object1's X dimension (1m / 2)

    # Target relative position for pelvis to be adjacent to Object1 for pushing
    # Assume robot approaches from -Y side of Object1 to push it in +Y direction
    # Target Y position: Object1_center_y - (object1_half_width_y + robot_body_depth_offset)
    # A reasonable robot body depth offset for pushing might be around 0.2m (to be close but not inside)
    robot_clearance_y = 0.2 # Clearance for robot body from block
    target_y_offset_from_block_center = object1_half_width_y + robot_clearance_y # 0.5m (half block Y) + 0.2m (robot clearance) = 0.7m
    target_pelvis_z = 0.7 # Stable standing height for pelvis (relative to ground z=0)

    # Calculate distances relative to Object1's center and target pelvis Z
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # X-distance: Pelvis X should align with Object1 X
    distance_x_pelvis_to_object1_x = torch.abs(pelvis_pos[:, 0] - object1_pos[:, 0])
    # Y-distance: Pelvis Y should be at target_y_offset_from_block_center *behind* Object1's Y center
    # If pelvis_y is object1_y - target_y_offset, then (object1_y - pelvis_y) should be target_y_offset
    distance_y_pelvis_to_object1_y = torch.abs((object1_pos[:, 1] - pelvis_pos[:, 1]) - target_y_offset_from_block_center)
    # Z-distance: Pelvis Z should be at target_pelvis_z
    distance_z_pelvis_to_target_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Primary reward: Penalize deviation from target relative positions
    # Requirement: Rewards should be continuous and negative for deviation
    reward = -distance_x_pelvis_to_object1_x \
             -distance_y_pelvis_to_object1_y \
             -distance_z_pelvis_to_target_z

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def feet_on_ground_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_on_ground_reward") -> torch.Tensor:
    """
    This reward encourages the robot to keep its feet on the ground, preventing it from lifting off or falling.
    It penalizes the feet being too high off the ground.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"] # Requirement: Access robot object using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Requirement: Access robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Requirement: Access robot part index using approved pattern
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2] # Requirement: Access robot part position using approved pattern
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2] # Requirement: Access robot part position using approved pattern

    # Target foot height (on ground, assuming ground is z=0)
    # A small positive value to account for foot thickness/contact point
    target_foot_z = 0.05

    # Reward for keeping feet near ground level
    # Requirement: ALL rewards MUST ONLY use relative distances (here, relative to target_foot_z)
    # Requirement: Rewards should be continuous and negative for deviation
    reward_left_foot = -torch.abs(left_foot_pos_z - target_foot_z)
    reward_right_foot = -torch.abs(right_foot_pos_z - target_foot_z)

    reward = reward_left_foot + reward_right_foot

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    This reward encourages collision avoidance between the robot's main body (pelvis) and Object1,
    and also encourages avoiding collisions with the ground.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Requirement: Access objects directly using approved pattern
    object1_pos = object1.data.root_pos_w # Requirement: Access object positions using approved pattern

    # Access the required robot part(s)
    robot = env.scene["robot"] # Requirement: Access robot object using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Requirement: Access robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Requirement: Access robot part position using approved pattern

    # Object1 dimensions (from skill info: x=1m, y=1m, z=0.3m)
    # Requirement: Hardcode object dimensions from config, do not access from object directly
    object1_half_height = 0.15 # Half of Object1's Z dimension (0.3m / 2)
    object1_half_width_y = 0.5 # Half of Object1's Y dimension (1m / 2)
    object1_half_depth_x = 0.5 # Half of Object1's X dimension (1m / 2)

    # Calculate relative positions for collision check
    # Requirement: ALL rewards MUST ONLY use relative distances
    rel_pos_x = pelvis_pos[:, 0] - object1_pos[:, 0]
    rel_pos_y = pelvis_pos[:, 1] - object1_pos[:, 1]
    rel_pos_z = pelvis_pos[:, 2] - object1_pos[:, 2]

    # Define thresholds for collision (e.g., robot body radius ~0.2m)
    # If pelvis is within these bounds, it's too close or colliding
    robot_body_radius = 0.2 # Approximate radius of robot pelvis for collision check

    # Collision thresholds for Object1 (block)
    # Pelvis should not be within the block's volume plus robot's radius
    collision_threshold_x = object1_half_depth_x + robot_body_radius
    collision_threshold_y = object1_half_width_y + robot_body_radius
    collision_threshold_z_upper = object1_half_height + robot_body_radius # Pelvis should not be above block + clearance
    collision_threshold_z_lower = object1_half_height - robot_body_radius # Pelvis should not be below block's center (i.e., inside it)

    # Condition for collision with Object1 (pelvis too close or inside block)
    # Penalize if pelvis is too close in X, Y, or Z relative to the block's volume
    # This checks if the pelvis is within the bounding box of the block + robot_body_radius
    collision_condition_x = torch.abs(rel_pos_x) < collision_threshold_x
    collision_condition_y = torch.abs(rel_pos_y) < collision_threshold_y
    # For Z, we want to penalize if pelvis is too high (above block) or too low (inside/below block)
    collision_condition_z_too_high = rel_pos_z > collision_threshold_z_upper
    collision_condition_z_too_low = rel_pos_z < -collision_threshold_z_lower # Pelvis below block's center

    # Combined collision condition for Object1
    # A collision is considered if the pelvis is within the X and Y bounds AND within the Z bounds (either too high or too low)
    collision_with_object1 = (collision_condition_x & collision_condition_y & (collision_condition_z_too_high | collision_condition_z_too_low))

    # Reward for avoiding collision (large negative if colliding)
    # Requirement: Rewards should be continuous (or at least not binary 0/1 for success/failure, but here a penalty is fine)
    # Using torch.where for a sharp penalty on collision
    reward_collision_with_object1 = torch.where(collision_with_object1, -10.0, 0.0)

    # Also penalize if pelvis is too low (e.g., falling through ground)
    # Assuming ground is at z=0, pelvis should not go below a certain threshold
    pelvis_min_z_threshold = 0.1 # If pelvis goes below this, it's likely falling or clipping ground
    # Requirement: Z is the only absolute position allowed for ground checks, used sparingly
    reward_ground_collision = torch.where(pelvis_pos[:, 2] < pelvis_min_z_threshold, -5.0, 0.0)

    reward = reward_collision_with_object1 + reward_ground_collision

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for reaching the target position relative to Object1
    WalkToSmallBlockReward = RewTerm(func=walk_to_Small_Block_for_push_reward, weight=1.0,
                                     params={"normalise": True, "normaliser_name": "walk_to_block_reward"})

    # Shaping reward for keeping feet on the ground
    FeetOnGroundReward = RewTerm(func=feet_on_ground_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "feet_on_ground_reward"})

    # Shaping reward for avoiding collisions with Object1 and the ground
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.5,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})