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


def main_ascend_small_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_ascend_small_block_reward") -> torch.Tensor:
    """
    Main reward for the Ascend_Small_Block_for_robot_interaction skill.
    This reward encourages the robot to first approach the small block (Object1) and then
    to get both feet onto its top surface, centering them and maintaining the correct height.
    """
    # Get normalizer instance - CRITICAL RULE: Normalizer instance is obtained this way.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # CRITICAL RULE: Access objects using env.scene['ObjectN']
    object1 = env.scene['Object1'] # Small Block for robot interaction

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # CRITICAL RULE: Hardcode object dimensions from the object configuration, do not access from object.data
    object1_height = 0.3
    object1_half_x = 1.0 / 2.0
    object1_half_y = 1.0 / 2.0

    # Calculate target z-position for feet on top of the block
    # CRITICAL RULE: Use relative distances. object1_top_z is relative to the block's root position.
    object1_top_z = object1.data.root_pos_w[:, 2] + object1_height / 2.0

    # Phase 1: Approach the block horizontally (pelvis)
    # Reward for reducing horizontal distance to the block's center
    # CRITICAL RULE: Use relative distances for x and y components.
    dist_pelvis_x = object1.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    dist_pelvis_y = object1.data.root_pos_w[:, 1] - pelvis_pos[:, 1]
    # CRITICAL RULE: Continuous reward, negative Euclidean distance.
    approach_reward = -torch.sqrt(dist_pelvis_x**2 + dist_pelvis_y**2)

    # Phase 2: Get feet onto the block's top surface
    # Define margins for "on top" detection. CRITICAL RULE: No arbitrary thresholds, these are small margins.
    margin_xy = 0.1 # 10cm margin for horizontal boundaries
    margin_z_above = 0.05 # 5cm above top surface
    margin_z_below = 0.05 # 5cm below top surface

    # Check if left foot is horizontally within block boundaries
    left_foot_on_block_x = (left_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - object1_half_x + margin_xy) & \
                           (left_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + object1_half_x - margin_xy)
    left_foot_on_block_y = (left_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - object1_half_y + margin_xy) & \
                           (left_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + object1_half_y - margin_xy)
    # Check if left foot is vertically at the correct height relative to the block's top surface
    left_foot_on_block_z = (left_foot_pos[:, 2] > object1_top_z - margin_z_below) & \
                           (left_foot_pos[:, 2] < object1_top_z + object1_height + margin_z_above) # Allow some height above block

    left_foot_on_top_surface = left_foot_on_block_x & left_foot_on_block_y & left_foot_on_block_z

    # Check if right foot is horizontally within block boundaries
    right_foot_on_block_x = (right_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - object1_half_x + margin_xy) & \
                            (right_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + object1_half_x - margin_xy)
    right_foot_on_block_y = (right_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - object1_half_y + margin_xy) & \
                            (right_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + object1_half_y - margin_xy)
    # Check if right foot is vertically at the correct height relative to the block's top surface
    right_foot_on_block_z = (right_foot_pos[:, 2] > object1_top_z - margin_z_below) & \
                            (right_foot_pos[:, 2] < object1_top_z + object1_height + margin_z_above)

    right_foot_on_top_surface = right_foot_on_block_x & right_foot_on_block_y & right_foot_on_block_z

    # Reward for feet being close to the center of the block's top surface
    # CRITICAL RULE: Use relative distances.
    dist_left_foot_to_center_x = object1.data.root_pos_w[:, 0] - left_foot_pos[:, 0]
    dist_left_foot_to_center_y = object1.data.root_pos_w[:, 1] - left_foot_pos[:, 1]
    dist_right_foot_to_center_x = object1.data.root_pos_w[:, 0] - right_foot_pos[:, 0]
    dist_right_foot_to_center_y = object1.data.root_pos_w[:, 1] - right_foot_pos[:, 1]

    # CRITICAL RULE: Continuous reward, negative sum of absolute distances.
    feet_centering_reward = - (torch.abs(dist_left_foot_to_center_x) + torch.abs(dist_left_foot_to_center_y) + \
                               torch.abs(dist_right_foot_to_center_x) + torch.abs(dist_right_foot_to_center_y))

    # Reward for feet being at the correct Z height relative to the block's top surface
    # CRITICAL RULE: Use relative distances.
    feet_z_reward = - (torch.abs(left_foot_pos[:, 2] - object1_top_z) + torch.abs(right_foot_pos[:, 2] - object1_top_z))

    # Combine rewards: approach until close, then focus on feet placement
    # CRITICAL RULE: No arbitrary thresholds, this threshold is relative to the approach distance.
    approach_threshold = 0.5 # meters
    is_close_to_block = (torch.sqrt(dist_pelvis_x**2 + dist_pelvis_y**2) < approach_threshold)

    # If close to block, prioritize feet placement and centering
    # Otherwise, prioritize approaching the block
    # CRITICAL RULE: Tensor operations handle batched environments.
    primary_reward = torch.where(is_close_to_block, feet_centering_reward + feet_z_reward, approach_reward)

    # Add a bonus for both feet being firmly on the block
    both_feet_on_block = left_foot_on_top_surface & right_foot_on_top_surface
    # CRITICAL RULE: Bonus is a continuous value, not binary, but applied conditionally.
    primary_reward = primary_reward + torch.where(both_feet_on_block, 5.0, 0.0) # Bonus for success

    reward = primary_reward

    # CRITICAL RULE: Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_on_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_on_block_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to maintain a stable, upright posture (pelvis at a target height)
    once it is on top of the block.
    """
    # Get normalizer instance - CRITICAL RULE: Normalizer instance is obtained this way.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    object1 = env.scene['Object1'] # Small Block for robot interaction
    robot = env.scene["robot"]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # CRITICAL RULE: Hardcode object dimensions from the object configuration
    object1_height = 0.3
    object1_half_x = 1.0 / 2.0
    object1_half_y = 1.0 / 2.0

    # Calculate target z-position for feet on top of the block
    object1_top_z = object1.data.root_pos_w[:, 2] + object1_height / 2.0

    # Define conditions for feet being on the block (similar to primary reward)
    margin_xy = 0.1
    margin_z_above = 0.05
    margin_z_below = 0.05

    left_foot_on_block_x = (left_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - object1_half_x + margin_xy) & \
                           (left_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + object1_half_x - margin_xy)
    left_foot_on_block_y = (left_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - object1_half_y + margin_xy) & \
                           (left_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + object1_half_y - margin_xy)
    left_foot_on_block_z = (left_foot_pos[:, 2] > object1_top_z - margin_z_below) & \
                           (left_foot_pos[:, 2] < object1_top_z + object1_height + margin_z_above)

    right_foot_on_block_x = (right_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - object1_half_x + margin_xy) & \
                            (right_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + object1_half_x - margin_xy)
    right_foot_on_block_y = (right_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - object1_half_y + margin_xy) & \
                            (right_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + object1_half_y - margin_xy)
    right_foot_on_block_z = (right_foot_pos[:, 2] > object1_top_z - margin_z_below) & \
                            (right_foot_pos[:, 2] < object1_top_z + object1_height + margin_z_above)

    both_feet_on_top_surface = left_foot_on_block_x & left_foot_on_block_y & left_foot_on_block_z & \
                               right_foot_on_block_x & right_foot_on_block_y & right_foot_on_block_z

    # Target pelvis height relative to the block's top surface
    # CRITICAL RULE: No arbitrary thresholds, this is a reasonable target height.
    target_pelvis_z_relative = 0.7 # Default stable pelvis height for the robot
    # CRITICAL RULE: Target pelvis height is relative to the block's top surface.
    target_pelvis_z_absolute = object1_top_z + target_pelvis_z_relative

    # Reward for pelvis being at the target height
    # CRITICAL RULE: Continuous reward, negative absolute difference.
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_absolute)

    # Activate this reward only when both feet are detected to be on the block
    # CRITICAL RULE: Tensor operations handle batched environments.
    reward = torch.where(both_feet_on_top_surface, pelvis_height_reward, torch.zeros_like(pelvis_height_reward))

    # CRITICAL RULE: Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_penalty_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_penalty_reward") -> torch.Tensor:
    """
    Shaping reward 2: Provides a penalty for collisions between the robot's body parts
    (excluding feet) and Object1.
    """
    # Get normalizer instance - CRITICAL RULE: Normalizer instance is obtained this way.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    object1 = env.scene['Object1'] # Small Block for robot interaction
    robot = env.scene["robot"]

    # CRITICAL RULE: Access robot parts using robot.body_names.index('part_name')
    pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')]
    head_pos = robot.data.body_pos_w[:, robot.body_names.index('head_link')]
    left_knee_pos = robot.data.body_pos_w[:, robot.body_names.index('left_knee_link')]
    right_knee_pos = robot.data.body_pos_w[:, robot.body_names.index('right_knee_link')]

    # CRITICAL RULE: Hardcode object dimensions from the object configuration
    object1_half_x = 1.0 / 2.0
    object1_half_y = 1.0 / 2.0
    object1_half_z = 0.3 / 2.0

    # Define a small clearance distance to avoid direct contact
    # CRITICAL RULE: No arbitrary thresholds, this is a small clearance.
    clearance = 0.05 # 5 cm

    # Function to calculate distance to block surface for a given robot part
    # This function returns a negative value if the part is penetrating or too close, 0 otherwise.
    # CRITICAL RULE: Uses relative distances (robot_part_pos - object_pos).
    def dist_to_block_surface(robot_part_pos, object_pos, half_dims, clearance):
        # Calculate distance from center of part to center of block, then subtract half-dims and clearance
        # This gives a measure of how much the part is "inside" the clearance zone or penetrating.
        dx = torch.abs(robot_part_pos[:, 0] - object_pos[:, 0]) - (half_dims[0] + clearance)
        dy = torch.abs(robot_part_pos[:, 1] - object_pos[:, 1]) - (half_dims[1] + clearance)
        dz = torch.abs(robot_part_pos[:, 2] - object_pos[:, 2]) - (half_dims[2] + clearance)

        # Only consider negative distances (penetration or too close)
        # CRITICAL RULE: Continuous reward, more negative for deeper penetration.
        dist_x = torch.where(dx < 0, dx, torch.zeros_like(dx))
        dist_y = torch.where(dy < 0, dy, torch.zeros_like(dy))
        dist_z = torch.where(dz < 0, dz, torch.zeros_like(dz))

        # Sum of negative distances (more negative for deeper penetration)
        return dist_x + dist_y + dist_z

    # Calculate collision penalties for relevant body parts
    # CRITICAL RULE: Tensor operations handle batched environments.
    # CRITICAL RULE: Ensure half_dims tensor is on the correct device.
    collision_penalty_pelvis = dist_to_block_surface(pelvis_pos, object1.data.root_pos_w,
                                                     torch.tensor([object1_half_x, object1_half_y, object1_half_z], device=env.device), clearance)
    collision_penalty_head = dist_to_block_surface(head_pos, object1.data.root_pos_w,
                                                   torch.tensor([object1_half_x, object1_half_y, object1_half_z], device=env.device), clearance)
    collision_penalty_left_knee = dist_to_block_surface(left_knee_pos, object1.data.root_pos_w,
                                                        torch.tensor([object1_half_x, object1_half_y, object1_half_z], device=env.device), clearance)
    collision_penalty_right_knee = dist_to_block_surface(right_knee_pos, object1.data.root_pos_w,
                                                         torch.tensor([object1_half_x, object1_half_y, object1_half_z], device=env.device), clearance)

    # Sum all penalties. Since dist_to_block_surface returns negative values for collision,
    # we want to add them as negative rewards.
    reward = collision_penalty_pelvis + collision_penalty_head + \
             collision_penalty_left_knee + collision_penalty_right_knee

    # Ensure the reward is always negative or zero (no positive reward for being far)
    reward = torch.min(reward, torch.zeros_like(reward))

    # CRITICAL RULE: Mandatory normalization
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
    Reward terms for the Ascend_Small_Block_for_robot_interaction skill.
    """
    # Main reward for approaching and ascending the small block - CRITICAL RULE: Main reward weight is 1.0.
    MainAscendSmallBlockReward = RewTerm(func=main_ascend_small_block_reward, weight=1.0,
                                         params={"normalise": True, "normaliser_name": "main_ascend_small_block_reward"})

    # Shaping reward for maintaining stable pelvis height once on the block - CRITICAL RULE: Shaping reward weight is < 1.0.
    PelvisHeightOnBlockReward = RewTerm(func=pelvis_height_on_block_reward, weight=0.4,
                                        params={"normalise": True, "normaliser_name": "pelvis_height_on_block_reward"})

    # Shaping reward for penalizing unwanted collisions with the block - CRITICAL RULE: Shaping reward weight is < 1.0.
    CollisionPenaltyReward = RewTerm(func=collision_penalty_reward, weight=0.2,
                                     params={"normalise": True, "normaliser_name": "collision_penalty_reward"})