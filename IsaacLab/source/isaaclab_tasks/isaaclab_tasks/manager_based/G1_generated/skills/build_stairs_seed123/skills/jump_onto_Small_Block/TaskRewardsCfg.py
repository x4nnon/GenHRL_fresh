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


def jump_onto_small_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the jump_onto_Small_Block skill.
    Encourages the robot to stand stably on top of the Small Block (Object1).
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (MANDATORY: Approved access pattern)
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w

    # Access the required robot parts (MANDATORY: Approved access pattern)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object1 dimensions (MANDATORY: Hardcoded from skill info, not accessed from object)
    object1_height = 0.3
    object1_half_x = 0.5
    object1_half_y = 0.5

    # Calculate Object1's center and top surface Z-coordinate
    object1_center_x = object1_pos[:, 0]
    object1_center_y = object1_pos[:, 1]
    object1_top_z = object1_pos[:, 2] + object1_height

    # Calculate relative distances (MANDATORY: Use relative distances)
    # Horizontal distance of pelvis to object1 center
    pelvis_dist_x = torch.abs(pelvis_pos[:, 0] - object1_center_x)
    pelvis_dist_y = torch.abs(pelvis_pos[:, 1] - object1_center_y)

    # Vertical distance of feet to object1 top surface
    left_foot_dist_z = torch.abs(left_foot_pos[:, 2] - object1_top_z)
    right_foot_dist_z = torch.abs(right_foot_pos[:, 2] - object1_top_z)

    # Pelvis height relative to object1 top surface for stability
    # Target pelvis height when standing on block (e.g., average human pelvis height ~0.7m above ground)
    # This is a relative height above the block's top surface.
    target_pelvis_z_on_block = 0.7
    pelvis_height_on_block_error = torch.abs(pelvis_pos[:, 2] - (object1_top_z + target_pelvis_z_on_block))

    # Reward for horizontal alignment (pelvis over block)
    # Rewards smaller horizontal distances, encouraging the robot to move over the block.
    horizontal_alignment_reward = - (pelvis_dist_x + pelvis_dist_y)

    # Reward for feet being on top of the block
    # This reward is active when feet are above the block's base and within horizontal bounds.
    # A small positive offset (0.05) is used to ensure feet are clearly above the base.
    # Horizontal bounds are slightly larger than the block to allow for some tolerance.
    feet_on_block_condition = (left_foot_pos[:, 2] > object1_pos[:, 2] + 0.05) & \
                              (right_foot_pos[:, 2] > object1_pos[:, 2] + 0.05) & \
                              (pelvis_dist_x < object1_half_x + 0.2) & \
                              (pelvis_dist_y < object1_half_y + 0.2)

    # Rewards smaller vertical distances of feet to the block's top surface.
    feet_on_block_reward = - (left_foot_dist_z + right_foot_dist_z)

    # Reward for stable standing on the block
    # This reward is active when both feet are very close to the block's top surface (within 0.05m).
    standing_on_block_condition = (left_foot_dist_z < 0.05) & (right_foot_dist_z < 0.05)

    # Rewards smaller error in pelvis height, encouraging stable standing.
    stability_reward = - pelvis_height_on_block_error

    # Combine rewards based on phases for continuous shaping
    # Phase 1: Approach and horizontal alignment is always active.
    primary_reward = horizontal_alignment_reward

    # Phase 2: Feet on block. Once horizontally aligned and feet are above the base,
    # add a stronger reward for feet being close to the top surface.
    primary_reward = torch.where(feet_on_block_condition, primary_reward + feet_on_block_reward * 2.0, primary_reward)

    # Phase 3: Standing stably on block. Once feet are on block,
    # add an even stronger reward for pelvis height stability.
    primary_reward = torch.where(standing_on_block_condition, primary_reward + stability_reward * 3.0, primary_reward)

    reward = primary_reward

    # Normalization (MANDATORY: Complete normalization implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_small_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward 1: Penalizes collisions between robot body parts (pelvis, knees) and the sides of Object1.
    Encourages the robot to clear the block rather than hitting it.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (MANDATORY: Approved access pattern)
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w

    # Access the required robot parts (MANDATORY: Approved access pattern)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_knee_idx = robot.body_names.index('left_knee_link')
    right_knee_idx = robot.body_names.index('right_knee_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]

    # Object1 dimensions (MANDATORY: Hardcoded from skill info)
    object1_half_x = 0.5
    object1_half_y = 0.5
    object1_height = 0.3

    # Object1's base and top Z-coordinates
    object1_center_x = object1_pos[:, 0]
    object1_center_y = object1_pos[:, 1]
    object1_base_z = object1_pos[:, 2]
    object1_top_z = object1_base_z + object1_height

    # Define a buffer zone around the block for collision detection (MANDATORY: Relative distances, not arbitrary thresholds)
    # These buffers define how close a body part can get to the block's sides before a penalty is applied.
    buffer_x = 0.1
    buffer_y = 0.1
    buffer_z = 0.1 # This buffer is for the vertical extent of the collision zone, allowing some clearance above the top.

    # Check for collision with pelvis (MANDATORY: Relative distances for collision detection)
    pelvis_collision_x = torch.abs(pelvis_pos[:, 0] - object1_center_x) < (object1_half_x + buffer_x)
    pelvis_collision_y = torch.abs(pelvis_pos[:, 1] - object1_center_y) < (object1_half_y + buffer_y)
    # Collision in Z is when pelvis is between the block's base and slightly above its top.
    pelvis_collision_z = (pelvis_pos[:, 2] > object1_base_z) & (pelvis_pos[:, 2] < object1_top_z + buffer_z)
    pelvis_colliding = pelvis_collision_x & pelvis_collision_y & pelvis_collision_z

    # Check for collision with left knee (MANDATORY: Relative distances for collision detection)
    left_knee_collision_x = torch.abs(left_knee_pos[:, 0] - object1_center_x) < (object1_half_x + buffer_x)
    left_knee_collision_y = torch.abs(left_knee_pos[:, 1] - object1_center_y) < (object1_half_y + buffer_y)
    left_knee_collision_z = (left_knee_pos[:, 2] > object1_base_z) & (left_knee_pos[:, 2] < object1_top_z + buffer_z)
    left_knee_colliding = left_knee_collision_x & left_knee_collision_y & left_knee_collision_z

    # Check for collision with right knee (MANDATORY: Relative distances for collision detection)
    right_knee_collision_x = torch.abs(right_knee_pos[:, 0] - object1_center_x) < (object1_half_x + buffer_x)
    right_knee_collision_y = torch.abs(right_knee_pos[:, 1] - object1_center_y) < (object1_half_y + buffer_y)
    right_knee_collision_z = (right_knee_pos[:, 2] > object1_base_z) & (right_knee_pos[:, 2] < object1_top_z + buffer_z)
    right_knee_colliding = right_knee_collision_x & right_knee_collision_y & right_knee_collision_z

    # Combine collision conditions (MANDATORY: Tensor operations for batched environments)
    any_body_part_colliding = pelvis_colliding | left_knee_colliding | right_knee_colliding

    # Apply a negative reward for collision (MANDATORY: Continuous rewards where possible, here it's a binary penalty)
    # A fixed penalty of -1.0 is applied when a collision is detected.
    collision_reward = torch.where(any_body_part_colliding, -1.0, 0.0)

    reward = collision_reward

    # Normalization (MANDATORY: Complete normalization implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def jump_height_encouragement_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_height_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourages the robot to gain sufficient vertical height (pelvis z-position)
    when it is horizontally aligned with Object1 and before its feet have landed on the block.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (MANDATORY: Approved access pattern)
    object1 = env.scene['Object1']
    object1_pos = object1.data.root_pos_w

    # Access the required robot parts (MANDATORY: Approved access pattern)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object1 dimensions (MANDATORY: Hardcoded from skill info)
    object1_height = 0.3
    object1_half_x = 0.5
    object1_half_y = 0.5

    # Object1's center and top surface Z-coordinate
    object1_center_x = object1_pos[:, 0]
    object1_center_y = object1_pos[:, 1]
    object1_top_z = object1_pos[:, 2] + object1_height

    # Horizontal alignment condition (pelvis roughly over the block)
    # This condition uses relative distances to check if the pelvis is within a reasonable horizontal range of the block.
    horizontal_aligned_condition = (torch.abs(pelvis_pos[:, 0] - object1_center_x) < object1_half_x + 0.2) & \
                                   (torch.abs(pelvis_pos[:, 1] - object1_center_y) < object1_half_y + 0.2)

    # Condition: Feet are not yet on the block (i.e., still in the air or approaching)
    # This checks if at least one foot is below the block's top surface (with a small buffer).
    feet_not_on_block_condition = (left_foot_pos[:, 2] < object1_top_z - 0.05) | \
                                  (right_foot_pos[:, 2] < object1_top_z - 0.05)

    # Activation condition: Robot is horizontally aligned AND feet are not yet on the block.
    # This ensures the reward is active during the jump phase over the block.
    activation_condition = horizontal_aligned_condition & feet_not_on_block_condition

    # Reward for increasing pelvis height above the block's top surface
    # Target height for pelvis during jump, e.g., 0.4m above block top.
    # This is a relative target height.
    target_jump_pelvis_z = object1_top_z + 0.4
    # Rewards smaller absolute difference between current pelvis Z and target jump Z.
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_jump_pelvis_z)

    # Apply reward only when conditions are met (MANDATORY: Tensor operations for batched environments)
    # The reward is zero if the activation conditions are not met, making it phase-specific.
    reward = torch.where(activation_condition, pelvis_height_reward, 0.0)

    # Normalization (MANDATORY: Complete normalization implementation)
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
    Configuration for the reward terms for the jump_onto_Small_Block skill.
    (MANDATORY: All reward components included with appropriate weights)
    """
    # Main reward for standing stably on the block
    MainJumpOntoSmallBlockReward = RewTerm(
        func=jump_onto_small_block_main_reward,
        weight=1.0,  # Primary reward, typically weight 1.0
        params={"normalise": True, "normaliser_name": "main_reward"}
    )

    # Shaping reward for collision avoidance
    CollisionAvoidanceSmallBlockReward = RewTerm(
        func=collision_avoidance_small_block_reward,
        weight=0.5,  # Shaping reward, typically lower weight
        params={"normalise": True, "normaliser_name": "collision_reward"}
    )

    # Shaping reward for encouraging jump height
    JumpHeightEncouragementReward = RewTerm(
        func=jump_height_encouragement_reward,
        weight=0.3,  # Shaping reward, typically lower weight
        params={"normalise": True, "normaliser_name": "jump_height_reward"}
    )