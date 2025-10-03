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


def ascend_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "ascend_medium_block_reward") -> torch.Tensor:
    """
    Primary reward for the robot to climb onto the Medium Block (Object2).
    Encourages both feet to be on the top surface of the Medium Block, horizontally centered,
    and the robot's pelvis to be at a stable standing height relative to the block's top surface.
    """
    # Get normalizer instance (MANDATORY IMPORT)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (MANDATORY: Use ObjectN names)
    medium_block = env.scene['Object2']

    # Access the required robot part(s) (MANDATORY: Use robot.body_names.index)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Medium Block dimensions (MANDATORY: Hardcode from object configuration, cannot access from object directly)
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # Calculate block's top surface Z position (MANDATORY: Relative distance calculation)
    block_top_z = medium_block.data.root_pos_w[:, 2] + medium_block_height / 2.0

    # Target Z position for feet relative to block's base (MANDATORY: Relative distance calculation)
    # A small offset to be clearly on top, slightly above the block's surface.
    target_foot_z_on_block = block_top_z + 0.05

    # Target Z position for pelvis relative to block's base (MANDATORY: Relative distance calculation)
    # 0.7m above the block's top surface for stable standing.
    target_pelvis_z_on_block = block_top_z + 0.7

    # Horizontal center of the block (MANDATORY: Relative distance calculation)
    block_center_x = medium_block.data.root_pos_w[:, 0]
    block_center_y = medium_block.data.root_pos_w[:, 1]

    # Reward for feet being on top of the block and horizontally centered
    # Use a small margin for being "on" the block (MANDATORY: Relative distance calculation, continuous reward)
    margin_x = medium_block_x_size / 2.0 - 0.1 # 0.1m margin from edge
    margin_y = medium_block_y_size / 2.0 - 0.1 # 0.1m margin from edge

    # Distance of left foot to block center (x, y) and target z (MANDATORY: Relative distance calculation)
    dist_left_foot_x = torch.abs(left_foot_pos[:, 0] - block_center_x)
    dist_left_foot_y = torch.abs(left_foot_pos[:, 1] - block_center_y)
    dist_left_foot_z = torch.abs(left_foot_pos[:, 2] - target_foot_z_on_block)

    # Distance of right foot to block center (x, y) and target z (MANDATORY: Relative distance calculation)
    dist_right_foot_x = torch.abs(right_foot_pos[:, 0] - block_center_x)
    dist_right_foot_y = torch.abs(right_foot_pos[:, 1] - block_center_y)
    dist_right_foot_z = torch.abs(right_foot_pos[:, 2] - target_foot_z_on_block)

    # Combined foot placement reward (MANDATORY: Continuous reward, handles batch processing)
    # Penalize if feet are outside horizontal bounds
    feet_on_block_horizontal_penalty = \
        (torch.max(torch.tensor(0.0, device=env.device), dist_left_foot_x - margin_x) +
         torch.max(torch.tensor(0.0, device=env.device), dist_left_foot_y - margin_y) +
         torch.max(torch.tensor(0.0, device=env.device), dist_right_foot_x - margin_x) +
         torch.max(torch.tensor(0.0, device=env.device), dist_right_foot_y - margin_y)) * 2.0
    feet_on_block_horizontal_reward = -feet_on_block_horizontal_penalty

    # Reward for feet being at the correct Z height (MANDATORY: Continuous reward)
    feet_z_reward = - (dist_left_foot_z + dist_right_foot_z)

    # Pelvis height reward for stability once feet are on the block (MANDATORY: Continuous reward)
    dist_pelvis_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block)
    pelvis_height_reward = - dist_pelvis_z

    # Condition for feet being generally on or above the block's top surface
    # This ensures the pelvis height reward only applies when the robot is actually ascending/on the block.
    feet_above_block_condition = (left_foot_pos[:, 2] > block_top_z - 0.1) & \
                                 (right_foot_pos[:, 2] > block_top_z - 0.1)

    # Combine rewards: prioritize getting feet on, then standing stably (MANDATORY: Continuous reward)
    primary_reward = feet_on_block_horizontal_reward + feet_z_reward + \
                     torch.where(feet_above_block_condition, pelvis_height_reward, torch.tensor(0.0, device=env.device))

    # MANDATORY: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes the robot if its body parts (pelvis, knees) collide with or get too close
    to the sides of the Medium Block during the ascent. Encourages a clean climb.
    """
    # Get normalizer instance (MANDATORY IMPORT)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (MANDATORY: Use ObjectN names)
    medium_block = env.scene['Object2']

    # Access the required robot part(s) (MANDATORY: Use robot.body_names.index)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_knee_idx = robot.body_names.index('left_knee_link')
    right_knee_idx = robot.body_names.index('right_knee_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]

    # Medium Block dimensions (MANDATORY: Hardcode from object configuration)
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # Block boundaries relative to its center (MANDATORY: Relative distance calculation)
    block_min_x = medium_block.data.root_pos_w[:, 0] - medium_block_x_size / 2.0
    block_max_x = medium_block.data.root_pos_w[:, 0] + medium_block_x_size / 2.0
    block_min_y = medium_block.data.root_pos_w[:, 1] - medium_block_y_size / 2.0
    block_max_y = medium_block.data.root_pos_w[:, 1] + medium_block_y_size / 2.0
    block_min_z = medium_block.data.root_pos_w[:, 2] - medium_block_height / 2.0
    block_max_z = medium_block.data.root_pos_w[:, 2] + medium_block_height / 2.0

    # Define a clearance margin (MANDATORY: No hard-coded positions, this is a relative margin)
    clearance_margin = 0.1 # 10 cm clearance

    # Initialize as per-environment tensor to avoid broadcasting issues
    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Check for pelvis collision/proximity (MANDATORY: Relative distance calculation, continuous penalty)
    # Penalize if any part of the body (considering clearance) is outside the block boundaries.
    pelvis_collision_x = torch.max(torch.tensor(0.0, device=env.device), block_min_x - (pelvis_pos[:, 0] + clearance_margin)) + \
                         torch.max(torch.tensor(0.0, device=env.device), (pelvis_pos[:, 0] - clearance_margin) - block_max_x)
    pelvis_collision_y = torch.max(torch.tensor(0.0, device=env.device), block_min_y - (pelvis_pos[:, 1] + clearance_margin)) + \
                         torch.max(torch.tensor(0.0, device=env.device), (pelvis_pos[:, 1] - clearance_margin) - block_max_y)
    pelvis_collision_z = torch.max(torch.tensor(0.0, device=env.device), block_min_z - (pelvis_pos[:, 2] + clearance_margin)) + \
                         torch.max(torch.tensor(0.0, device=env.device), (pelvis_pos[:, 2] - clearance_margin) - block_max_z)
    collision_penalty += (pelvis_collision_x + pelvis_collision_y + pelvis_collision_z) * 0.5

    # Check for left knee collision/proximity (MANDATORY: Relative distance calculation, continuous penalty)
    left_knee_collision_x = torch.max(torch.tensor(0.0, device=env.device), block_min_x - (left_knee_pos[:, 0] + clearance_margin)) + \
                            torch.max(torch.tensor(0.0, device=env.device), (left_knee_pos[:, 0] - clearance_margin) - block_max_x)
    left_knee_collision_y = torch.max(torch.tensor(0.0, device=env.device), block_min_y - (left_knee_pos[:, 1] + clearance_margin)) + \
                            torch.max(torch.tensor(0.0, device=env.device), (left_knee_pos[:, 1] - clearance_margin) - block_max_y)
    left_knee_collision_z = torch.max(torch.tensor(0.0, device=env.device), block_min_z - (left_knee_pos[:, 2] + clearance_margin)) + \
                            torch.max(torch.tensor(0.0, device=env.device), (left_knee_pos[:, 2] - clearance_margin) - block_max_z)
    collision_penalty += (left_knee_collision_x + left_knee_collision_y + left_knee_collision_z) * 0.5

    # Check for right knee collision/proximity (MANDATORY: Relative distance calculation, continuous penalty)
    right_knee_collision_x = torch.max(torch.tensor(0.0, device=env.device), block_min_x - (right_knee_pos[:, 0] + clearance_margin)) + \
                             torch.max(torch.tensor(0.0, device=env.device), (right_knee_pos[:, 0] - clearance_margin) - block_max_x)
    right_knee_collision_y = torch.max(torch.tensor(0.0, device=env.device), block_min_y - (right_knee_pos[:, 1] + clearance_margin)) + \
                             torch.max(torch.tensor(0.0, device=env.device), (right_knee_pos[:, 1] - clearance_margin) - block_max_y)
    right_knee_collision_z = torch.max(torch.tensor(0.0, device=env.device), block_min_z - (right_knee_pos[:, 2] + clearance_margin)) + \
                             torch.max(torch.tensor(0.0, device=env.device), (right_knee_pos[:, 2] - clearance_margin) - block_max_z)
    collision_penalty += (right_knee_collision_x + right_knee_collision_y + right_knee_collision_z) * 0.5

    shaping_reward_1 = -collision_penalty # Convert penalty to negative reward

    # MANDATORY: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_1)
        return scaled_reward
    return shaping_reward_1


def forward_progress_alignment_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_alignment_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to move towards the Medium Block and align itself correctly
    before attempting to climb. It focuses on reducing the horizontal distance between the robot's pelvis
    and the block's center, but only when the robot is still primarily below the block's top surface.
    """
    # Get normalizer instance (MANDATORY IMPORT)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (MANDATORY: Use ObjectN names)
    medium_block = env.scene['Object2']

    # Access the required robot part(s) (MANDATORY: Use robot.body_names.index)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Horizontal center of the block (MANDATORY: Relative distance calculation)
    block_center_x = medium_block.data.root_pos_w[:, 0]
    block_center_y = medium_block.data.root_pos_w[:, 1]

    # Distance from pelvis to block center (x, y) (MANDATORY: Relative distance calculation, continuous reward)
    dist_pelvis_to_block_center_x = torch.abs(pelvis_pos[:, 0] - block_center_x)
    dist_pelvis_to_block_center_y = torch.abs(pelvis_pos[:, 1] - block_center_y)

    # Medium Block dimensions (MANDATORY: Hardcode from object configuration)
    medium_block_height = 0.6

    # Calculate block's top surface Z position (MANDATORY: Relative distance calculation)
    block_top_z = medium_block.data.root_pos_w[:, 2] + medium_block_height / 2.0

    # Condition: Pelvis is still below the top of the block (or slightly above, indicating initial ascent)
    # This reward should primarily guide the approach and initial climb, not stabilization on top.
    # Use a threshold slightly above the block's top surface to allow for initial lifting.
    activation_condition = pelvis_pos[:, 2] < (block_top_z + 0.3) # Pelvis should not be too high above the block yet

    # Reward for reducing horizontal distance (MANDATORY: Continuous reward, handles batch processing)
    horizontal_alignment_reward = - (dist_pelvis_to_block_center_x + dist_pelvis_to_block_center_y)

    shaping_reward_2 = torch.where(activation_condition, horizontal_alignment_reward, torch.tensor(0.0, device=env.device))

    # MANDATORY: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_2)
        return scaled_reward
    return shaping_reward_2


@configclass
class TaskRewardsCfg:
    # Primary reward for ascending the medium block (MANDATORY: Weight ~1.0)
    AscendMediumBlockReward = RewTerm(func=ascend_medium_block_reward, weight=1.0,
                                      params={"normalise": True, "normaliser_name": "ascend_medium_block_reward"})

    # Shaping reward for collision avoidance with the medium block (MANDATORY: Weight <1.0)
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for forward progress and alignment towards the medium block (MANDATORY: Weight <1.0)
    ForwardProgressAlignmentReward = RewTerm(func=forward_progress_alignment_reward, weight=0.3,
                                             params={"normalise": True, "normaliser_name": "forward_progress_alignment_reward"})