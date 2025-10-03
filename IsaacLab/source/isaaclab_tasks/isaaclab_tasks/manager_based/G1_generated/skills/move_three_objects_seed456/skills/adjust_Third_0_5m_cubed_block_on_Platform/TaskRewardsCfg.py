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


def main_adjust_block_on_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for ensuring Object3 (Third 0.5m cubed block) is fully and stably positioned
    within the boundaries of Object4 (Platform).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3'] # Third 0.5m cubed block
    object4 = env.scene['Object4'] # Platform

    # Object dimensions (hardcoded from object configuration, as per requirements)
    block_half_size = 0.5 / 2.0 # 0.25m for a 0.5m cubed block
    platform_half_x = 2.0 / 2.0 # 1.0m for a 2m wide platform
    platform_half_y = 2.0 / 2.0 # 1.0m for a 2m deep platform
    platform_height = 0.001 # Z dimension of platform (from object config)

    # Get positions using approved patterns
    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Calculate distance of block center to platform center (relative distances)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    center_dist_x = torch.abs(object3_pos[:, 0] - object4_pos[:, 0])
    center_dist_y = torch.abs(object3_pos[:, 1] - object4_pos[:, 1])

    # Reward for block being centered on platform (smaller distance is better, continuous reward)
    reward_center_alignment = - (center_dist_x + center_dist_y)

    # Penalize block edges going beyond platform boundaries
    # Requirement: NEVER use hard-coded positions or arbitrary thresholds (except for object dimensions and relative target offsets).
    # A small margin is added to ensure it's "fully" within, this is a relative offset.
    margin = 0.05 # Small margin for stability, relative to the block's edge and platform boundary

    # Calculate the effective bounds for the block's center to be within the platform
    # This accounts for the block's size and the platform's size, ensuring the block's edges are inside.
    x_upper_bound_center = object4_pos[:, 0] + platform_half_x - block_half_size - margin
    x_lower_bound_center = object4_pos[:, 0] - platform_half_x + block_half_size + margin
    y_upper_bound_center = object4_pos[:, 1] + platform_half_y - block_half_size - margin
    y_lower_bound_center = object4_pos[:, 1] - platform_half_y + block_half_size + margin

    # Calculate how much the block's center is outside the desired bounds
    # Max(0, distance_outside) ensures positive penalty only when outside, making it continuous.
    penalty_x_upper = torch.max(torch.tensor(0.0, device=env.device), object3_pos[:, 0] - x_upper_bound_center)
    penalty_x_lower = torch.max(torch.tensor(0.0, device=env.device), x_lower_bound_center - object3_pos[:, 0])
    penalty_y_upper = torch.max(torch.tensor(0.0, device=env.device), object3_pos[:, 1] - y_upper_bound_center)
    penalty_y_lower = torch.max(torch.tensor(0.0, device=env.device), y_lower_bound_center - object3_pos[:, 1])

    reward_boundary_penalty = - (penalty_x_upper + penalty_x_lower + penalty_y_upper + penalty_y_lower) * 5.0 # Increased penalty weight for being off platform

    # Reward for block being at the correct height relative to the platform
    # Block's bottom should be on platform's top surface.
    # Block's center Z should be platform_z + platform_half_height + block_half_size.
    # Requirement: Z is the only absolute position allowed, used here for height alignment.
    target_block_z = object4_pos[:, 2] + (platform_height / 2.0) + block_half_size
    reward_z_alignment = -torch.abs(object3_pos[:, 2] - target_block_z)

    reward = reward_center_alignment + reward_boundary_penalty + reward_z_alignment

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def approach_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    """
    Shaping reward encouraging the robot's active hand to approach Object3 and maintaining stable pelvis height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object3 = env.scene['Object3'] # Third 0.5m cubed block
    robot = env.scene["robot"]

    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx] # Accessing robot part position

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position

    # Object dimensions (hardcoded from object configuration)
    block_half_size = 0.5 / 2.0 # 0.25m

    # Calculate distance vector between right palm and Object3 (relative distances)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    distance_palm_x = object3.data.root_pos_w[:, 0] - right_palm_pos[:, 0]
    distance_palm_y = object3.data.root_pos_w[:, 1] - right_palm_pos[:, 1]
    distance_palm_z = object3.data.root_pos_w[:, 2] - right_palm_pos[:, 2]

    # Reward for hand approaching the block (closer is better, continuous reward)
    # Consider a small offset for pushing, e.g., 0.1m from the block's surface.
    # This offset is a relative target for the hand's position.
    target_palm_offset_z = block_half_size + 0.1 # Encourage hand to be slightly above/behind the block in Z for pushing
    
    # Reward for X and Y distance to block center
    reward_palm_approach_xy = -torch.norm(torch.stack([distance_palm_x, distance_palm_y], dim=1), dim=1) # Combined XY distance

    # Reward for Z distance, encouraging the hand to be at a specific relative height for pushing
    reward_palm_approach_z = -torch.abs(distance_palm_z - target_palm_offset_z)

    # Reward for maintaining stable pelvis height
    # Requirement: Z is the only absolute position allowed, used here for pelvis height.
    target_pelvis_z = 0.7 # Target height for pelvis, a relative offset from ground.
    reward_pelvis_stability = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    reward = (reward_palm_approach_xy + reward_palm_approach_z) * 0.5 + reward_pelvis_stability * 0.2

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_and_feet_placement_penalty(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    """
    Shaping reward encouraging collision avoidance between robot's body parts (excluding active hand)
    and Object3/Object4, and penalizing robot's feet being on the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object3 = env.scene['Object3'] # Third 0.5m cubed block
    object4 = env.scene['Object4'] # Platform
    robot = env.scene["robot"]

    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx]

    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object dimensions (hardcoded from object configuration)
    block_half_size = 0.5 / 2.0 # 0.25m
    platform_half_x = 2.0 / 2.0 # 1.0m
    platform_half_y = 2.0 / 2.0 # 1.0m
    platform_height = 0.001 # Z dimension of platform

    # Penalty for feet being on the platform
    # Feet Z should be below platform Z + small tolerance.
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    # The platform's top Z is relative to its root position.
    platform_top_z = object4.data.root_pos_w[:, 2] + platform_height / 2.0
    # Accumulator per environment
    foot_on_platform_penalty = torch.zeros(env.num_envs, device=env.device)

    # Check if left foot is above platform Z and within platform XY bounds (relative to platform center)
    left_foot_on_platform_condition = (left_ankle_pos[:, 2] > platform_top_z - 0.05) & \
                                      (torch.abs(left_ankle_pos[:, 0] - object4.data.root_pos_w[:, 0]) < platform_half_x) & \
                                      (torch.abs(left_ankle_pos[:, 1] - object4.data.root_pos_w[:, 1]) < platform_half_y)
    foot_on_platform_penalty += torch.where(
        left_foot_on_platform_condition,
        torch.ones_like(left_foot_on_platform_condition, dtype=foot_on_platform_penalty.dtype),
        torch.zeros_like(left_foot_on_platform_condition, dtype=foot_on_platform_penalty.dtype)
    ) * 5.0

    # Check if right foot is above platform Z and within platform XY bounds (relative to platform center)
    right_foot_on_platform_condition = (right_ankle_pos[:, 2] > platform_top_z - 0.05) & \
                                       (torch.abs(right_ankle_pos[:, 0] - object4.data.root_pos_w[:, 0]) < platform_half_x) & \
                                       (torch.abs(right_ankle_pos[:, 1] - object4.data.root_pos_w[:, 1]) < platform_half_y)
    foot_on_platform_penalty += torch.where(
        right_foot_on_platform_condition,
        torch.ones_like(right_foot_on_platform_condition, dtype=foot_on_platform_penalty.dtype),
        torch.zeros_like(right_foot_on_platform_condition, dtype=foot_on_platform_penalty.dtype)
    ) * 5.0

    # Collision avoidance for pelvis with Object3 and Object4
    # Define a safe distance for collision avoidance (a relative threshold)
    safe_dist_pelvis = 0.3 # meters

    # Distance from pelvis to Object3 (relative distances)
    dist_pelvis_obj3_x = torch.abs(pelvis_pos[:, 0] - object3.data.root_pos_w[:, 0])
    dist_pelvis_obj3_y = torch.abs(pelvis_pos[:, 1] - object3.data.root_pos_w[:, 1])
    dist_pelvis_obj3_z = torch.abs(pelvis_pos[:, 2] - object3.data.root_pos_w[:, 2])

    # Penalty if pelvis is too close to Object3 (continuous penalty)
    pelvis_obj3_collision_penalty = torch.max(torch.tensor(0.0, device=env.device), safe_dist_pelvis - dist_pelvis_obj3_x) + \
                                    torch.max(torch.tensor(0.0, device=env.device), safe_dist_pelvis - dist_pelvis_obj3_y) + \
                                    torch.max(torch.tensor(0.0, device=env.device), safe_dist_pelvis - dist_pelvis_obj3_z)
    pelvis_obj3_collision_penalty *= 2.0 # Increased penalty weight

    # Distance from pelvis to Object4 (platform) (relative distances)
    dist_pelvis_obj4_x = torch.abs(pelvis_pos[:, 0] - object4.data.root_pos_w[:, 0])
    dist_pelvis_obj4_y = torch.abs(pelvis_pos[:, 1] - object4.data.root_pos_w[:, 1])
    dist_pelvis_obj4_z = torch.abs(pelvis_pos[:, 2] - object4.data.root_pos_w[:, 2])

    # Penalty if pelvis is too close to Object4 (continuous penalty)
    pelvis_obj4_collision_penalty = torch.max(torch.tensor(0.0, device=env.device), safe_dist_pelvis - dist_pelvis_obj4_x) + \
                                    torch.max(torch.tensor(0.0, device=env.device), safe_dist_pelvis - dist_pelvis_obj4_y) + \
                                    torch.max(torch.tensor(0.0, device=env.device), safe_dist_pelvis - dist_pelvis_obj4_z)
    pelvis_obj4_collision_penalty *= 1.0 # Increased penalty weight

    reward = - (foot_on_platform_penalty + pelvis_obj3_collision_penalty + pelvis_obj4_collision_penalty)

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
    # Primary reward for block placement, weight 1.0 as it's the main objective
    MainAdjustBlockOnPlatformReward = RewTerm(func=main_adjust_block_on_platform_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for hand approach and pelvis stability, lower weight
    ApproachAndStabilityReward = RewTerm(func=approach_and_stability_reward, weight=0.6,
                                         params={"normalise": True, "normaliser_name": "shaping_reward_1"})

    # Shaping reward for collision avoidance and feet placement, lower weight (penalties)
    CollisionAvoidanceAndFeetPlacementPenalty = RewTerm(func=collision_avoidance_and_feet_placement_penalty, weight=0.4,
                                                        params={"normalise": True, "normaliser_name": "shaping_reward_2"})