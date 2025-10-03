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


def main_jump_onto_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the robot to land stably on top of the Medium Block (Object2).
    This reward encourages the robot's feet to be on the top surface of the block and its pelvis
    to be at a stable standing height relative to the block.
    """
    # Get normalizer instance
    # Requirement: EVERY reward function MUST include normalization
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w

    # Access the required robot part(s)
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Medium Block dimensions (from object configuration, hardcoded as per requirements)
    # Requirement: There is no way to access the SIZE of an object. You must read this from the object config and hard code the value for it.
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # Calculate target z-position for feet (top surface of the block)
    # Block's root_pos_w is its center. Top surface is root_pos_w.z + half_height
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    target_block_top_z = medium_block_pos[:, 2] + (medium_block_height / 2.0)

    # Calculate target z-position for pelvis (stable standing height relative to block top)
    # A typical standing height for the pelvis relative to the ground is around 0.7m.
    # So, relative to the block's top, it should be 0.7m - (medium_block_height / 2.0)
    pelvis_standing_height_relative_to_ground = 0.7
    target_pelvis_z = target_block_top_z + (pelvis_standing_height_relative_to_ground - (medium_block_height / 2.0))

    # Distance of feet to the center of the block's top surface (x, y components)
    # Use the block's root_pos_w for x,y center
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]

    # Reward for feet being on top of the block (z-component)
    # Encourages feet to be at the correct height (top of the block)
    # Requirement: Rewards should be continuous and positive where possible. Using negative absolute distance for minimization.
    reward_feet_z = -torch.abs(left_foot_pos[:, 2] - target_block_top_z) - torch.abs(right_foot_pos[:, 2] - target_block_top_z)

    # Reward for feet being within the x-y boundaries of the block
    # Encourages feet to be horizontally centered on the block.
    # Requirement: Rewards should be continuous.
    reward_feet_x_y = \
        -torch.abs(left_foot_pos[:, 0] - block_center_x) \
        -torch.abs(left_foot_pos[:, 1] - block_center_y) \
        -torch.abs(right_foot_pos[:, 0] - block_center_x) \
        -torch.abs(right_foot_pos[:, 1] - block_center_y)

    # Reward for pelvis being at a stable standing height
    # Encourages the robot to stand upright and stably on the block.
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards. The feet on block reward should be dominant for landing.
    # The pelvis height reward ensures stability after landing.
    reward = reward_feet_z + reward_feet_x_y + reward_pelvis_z

    # Mandatory reward normalization
    # Requirement: EVERY reward function MUST include normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_approach_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_approach_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach the Medium Block (Object2) horizontally
    before the jump. This reward is active when the robot's pelvis is below the block's top surface.
    """
    # Get normalizer instance
    # Requirement: EVERY reward function MUST include normalization
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w

    # Access the required robot part(s)
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Medium Block dimensions (from object configuration, hardcoded)
    # Requirement: There is no way to access the SIZE of an object. You must read this from the object config and hard code the value for it.
    medium_block_height = 0.6

    # Calculate target z-position for feet (top surface of the block)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    target_block_top_z = medium_block_pos[:, 2] + (medium_block_height / 2.0)

    # Distance of pelvis to the center of the block (x, y components)
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]

    # Calculate relative distances
    distance_x = pelvis_pos[:, 0] - block_center_x
    distance_y = pelvis_pos[:, 1] - block_center_y

    # Condition: Pelvis is below the top surface of the block, indicating approach/pre-jump phase
    # Add a small buffer to ensure it's clearly below, not just touching
    # Requirement: All operations must work with batched environments
    activation_condition = (pelvis_pos[:, 2] < target_block_top_z - 0.05)

    # Reward for reducing horizontal distance to the block
    # Negative absolute distance encourages minimization, making it continuous.
    # Requirement: Rewards should be continuous.
    shaping_reward = -torch.abs(distance_x) - torch.abs(distance_y)

    # Apply activation condition
    # Requirement: Rewards should be continuous. Using torch.where for conditional reward.
    reward = torch.where(activation_condition, shaping_reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    # Requirement: EVERY reward function MUST include normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_jump_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_jump_height_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to gain sufficient vertical height during the jump.
    It rewards the pelvis and feet for being above the block's top surface, but only when the robot
    is horizontally aligned with the block, indicating it's in the jump phase.
    """
    # Get normalizer instance
    # Requirement: EVERY reward function MUST include normalization
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    medium_block = env.scene['Object2']
    medium_block_pos = medium_block.data.root_pos_w

    # Access the required robot part(s)
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Medium Block dimensions (from object configuration, hardcoded)
    # Requirement: There is no way to access the SIZE of an object. You must read this from the object config and hard code the value for it.
    medium_block_height = 0.6
    medium_block_x_size = 1.0
    medium_block_y_size = 1.0

    # Calculate target z-position for feet (top surface of the block)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    target_block_top_z = medium_block_pos[:, 2] + (medium_block_height / 2.0)

    # Block's root_pos_w for x,y center
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]

    # Condition: Robot is horizontally aligned with the block (within its x-y bounds)
    # and its pelvis is above the block's top surface (indicating jump or landing)
    # Requirement: All operations must work with batched environments
    horizontal_alignment_x = (pelvis_pos[:, 0] > block_center_x - medium_block_x_size / 2.0) & \
                             (pelvis_pos[:, 0] < block_center_x + medium_block_x_size / 2.0)
    horizontal_alignment_y = (pelvis_pos[:, 1] > block_center_y - medium_block_y_size / 2.0) & \
                             (pelvis_pos[:, 1] < block_center_y + medium_block_y_size / 2.0)

    # Pelvis is above the block's top surface (with a small buffer)
    pelvis_above_block = (pelvis_pos[:, 2] > target_block_top_z + 0.05)

    activation_condition = horizontal_alignment_x & horizontal_alignment_y & pelvis_above_block

    # Reward for increasing height above the block's top surface
    # We want to maximize this height during the jump, so positive reward for higher Z
    # Requirement: Rewards should be continuous and positive where possible.
    shaping_reward = (pelvis_pos[:, 2] - target_block_top_z) + \
                     (left_foot_pos[:, 2] - target_block_top_z) + \
                     (right_foot_pos[:, 2] - target_block_top_z)

    # Ensure reward is only positive when above the block
    shaping_reward = torch.where(shaping_reward > 0, shaping_reward, torch.tensor(0.0, device=env.device))
    # Apply activation condition
    # Requirement: Rewards should be continuous. Using torch.where for conditional reward.
    reward = torch.where(activation_condition, shaping_reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    # Requirement: EVERY reward function MUST include normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for landing on the medium block
    # Requirement: Main reward with weight 1.0
    MainJumpOntoMediumBlockReward = RewTerm(func=main_jump_onto_medium_block_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for horizontal approach
    # Requirement: Supporting rewards with lower weights (0.5 as per plan)
    ShapingApproachMediumBlockReward = RewTerm(func=shaping_approach_medium_block_reward, weight=0.5,
                                               params={"normalise": True, "normaliser_name": "shaping_approach_reward"})

    # Shaping reward for vertical jump height
    # Requirement: Supporting rewards with lower weights (0.5 as per plan)
    ShapingJumpHeightReward = RewTerm(func=shaping_jump_height_reward, weight=0.5,
                                      params={"normalise": True, "normaliser_name": "shaping_jump_height_reward"})