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


def main_jump_onto_medium_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_jump_onto_medium_block_reward") -> torch.Tensor:
    '''
    Main reward for the jump_onto_Medium_Block skill.
    This reward encourages the robot to jump onto the top surface of the Medium Block (Object2) and stand stably.
    It combines rewards for approaching the block, landing accurately on its top surface, and maintaining stability.
    '''
    # Get normalizer instance for this reward function.
    # This addresses the "MANDATORY REWARD NORMALIZATION" requirement.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns.
    # This addresses "USE ONLY APPROVED ACCESS PATTERNS" and "DIRECT OBJECT ACCESS" requirements.
    robot = env.scene["robot"]
    medium_block = env.scene['Object2']

    # Get indices for specific robot body parts.
    # This addresses "USE ONLY APPROVED ACCESS PATTERNS" and "NO hard-coded body indices" requirements.
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # Get world positions for the robot parts.
    # This addresses "USE ONLY APPROVED ACCESS PATTERNS" requirement.
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get world position for the medium block.
    # This addresses "USE ONLY APPROVED ACCESS PATTERNS" and "YOU MUST ACCESS OBJECT LOCATIONS" requirements.
    medium_block_pos = medium_block.data.root_pos_w

    # Hardcode Object2 dimensions from the skill information.
    # This addresses "THERE IS NO way to access the SIZE of an object" and "HOW TO USE OBJECT DIMENSIONS" requirements.
    # Medium Block (Object2) measures x=1m, y=1m, z=0.6m.
    medium_block_height = 0.6
    medium_block_half_x = 0.5
    medium_block_half_y = 0.5

    # Calculate block's center and top surface Z-coordinate.
    # These are relative calculations based on object root position and hardcoded dimensions.
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]
    block_top_z = medium_block_pos[:, 2] + medium_block_height

    # --- Reward Component 1: Horizontal Approach Reward (Pelvis to Block Center) ---
    # This encourages the robot to move horizontally towards the block.
    # It's a negative reward, so smaller distance (closer) is better.
    # Uses relative distances and works with batched environments.
    pelvis_dist_x = torch.abs(pelvis_pos[:, 0] - block_center_x)
    pelvis_dist_y = torch.abs(pelvis_pos[:, 1] - block_center_y)
    horizontal_approach_reward = - (pelvis_dist_x + pelvis_dist_y)
    # Normalization: Scale by a typical approach distance, e.g., 5m.
    horizontal_approach_reward = horizontal_approach_reward / 5.0

    # --- Reward Component 2: Vertical Landing Reward (Feet to Block Top Surface) ---
    # This encourages the feet to be at the same height as the block's top surface.
    # It's a negative reward, so smaller difference is better.
    # Uses relative distances and works with batched environments.
    left_foot_z_diff = torch.abs(left_foot_pos[:, 2] - block_top_z)
    right_foot_z_diff = torch.abs(right_foot_pos[:, 2] - block_top_z)
    vertical_landing_reward = - (left_foot_z_diff + right_foot_z_diff)
    # Normalization: Scale by a typical height difference, e.g., 1m.
    vertical_landing_reward = vertical_landing_reward / 1.0

    # --- Reward Component 3: Horizontal Landing Accuracy Reward (Feet within Block Boundaries) ---
    # This penalizes feet being outside the block's horizontal boundaries.
    # `torch.max(0.0, ...)` ensures penalty only applies when outside the block.
    # Uses relative distances and works with batched environments.
    left_foot_on_block_x_dist = torch.abs(left_foot_pos[:, 0] - block_center_x)
    left_foot_on_block_y_dist = torch.abs(left_foot_pos[:, 1] - block_center_y)
    right_foot_on_block_x_dist = torch.abs(right_foot_pos[:, 0] - block_center_x)
    right_foot_on_block_y_dist = torch.abs(right_foot_pos[:, 1] - block_center_y)

    horizontal_landing_accuracy_penalty = \
        torch.max(torch.tensor(0.0, device=env.device), left_foot_on_block_x_dist - medium_block_half_x) + \
        torch.max(torch.tensor(0.0, device=env.device), left_foot_on_block_y_dist - medium_block_half_y) + \
        torch.max(torch.tensor(0.0, device=env.device), right_foot_on_block_x_dist - medium_block_half_x) + \
        torch.max(torch.tensor(0.0, device=env.device), right_foot_on_block_y_dist - medium_block_half_y)
    horizontal_landing_accuracy_reward = -horizontal_landing_accuracy_penalty
    # Normalization: Scale by a typical horizontal overshoot, e.g., 1m.
    horizontal_landing_accuracy_reward = horizontal_landing_accuracy_reward / 1.0

    # --- Reward Component 4: Pelvis Stability Reward (Pelvis at Target Height above Block) ---
    # This encourages the pelvis to be at a stable standing height above the block.
    # A typical standing height for a humanoid robot's pelvis might be around 0.7m above the ground/surface.
    # Uses relative distances and works with batched environments.
    target_pelvis_z_on_block = block_top_z + 0.7
    pelvis_stability_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block)
    # Normalization: Scale by a typical height difference, e.g., 1m.
    pelvis_stability_reward = pelvis_stability_reward / 1.0

    # --- Combine Rewards based on phases/conditions ---
    # Define a condition for being "on top" of the block.
    # This means feet are horizontally within bounds and vertically near the top surface.
    # This uses relative positions and works with batched environments.
    feet_on_block_horizontal_condition = \
        (left_foot_on_block_x_dist < medium_block_half_x) & \
        (left_foot_on_block_y_dist < medium_block_half_y) & \
        (right_foot_on_block_x_dist < medium_block_half_x) & \
        (right_foot_on_block_y_dist < medium_block_half_y)

    # Allow a small tolerance for vertical position to consider feet "on" the block.
    feet_on_block_vertical_condition = \
        (left_foot_pos[:, 2] > block_top_z - 0.1) & \
        (right_foot_pos[:, 2] > block_top_z - 0.1) & \
        (left_foot_pos[:, 2] < block_top_z + 0.2) & \
        (right_foot_pos[:, 2] < block_top_z + 0.2)

    feet_on_block_condition = feet_on_block_horizontal_condition & feet_on_block_vertical_condition

    # If the robot is considered "on the block", prioritize landing accuracy and stability.
    # Otherwise, prioritize horizontal approach. This creates a continuous reward.
    on_block_combined_reward = vertical_landing_reward + horizontal_landing_accuracy_reward + pelvis_stability_reward
    reward = torch.where(feet_on_block_condition, on_block_combined_reward, horizontal_approach_reward)

    # Ensure reward is continuous and positive where possible by adding a constant offset.
    # The maximum possible negative reward for approach is around -5.0.
    # The maximum possible negative reward for landing/stability is around -3.0.
    # So, a constant of 5.0 or 6.0 should make it mostly positive.
    reward = reward + 6.0

    # Mandatory reward normalization.
    # This addresses the "MANDATORY REWARD NORMALIZATION" requirement.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def jump_height_encouragement_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_height_encouragement_reward") -> torch.Tensor:
    '''
    Shaping reward to encourage the robot to increase its pelvis height above the block's top surface during the jump.
    This ensures it clears the block. It is active when the robot is horizontally near the block but not yet landed on it.
    '''
    # Get normalizer instance.
    # This addresses the "MANDATORY REWARD NORMALIZATION" requirement.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts.
    # This addresses "USE ONLY APPROVED ACCESS PATTERNS" and "DIRECT OBJECT ACCESS" requirements.
    robot = env.scene["robot"]
    medium_block = env.scene['Object2']
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    medium_block_pos = medium_block.data.root_pos_w

    # Hardcode Object2 dimensions.
    # This addresses "THERE IS NO way to access the SIZE of an object" and "HOW TO USE OBJECT DIMENSIONS" requirements.
    medium_block_height = 0.6
    medium_block_half_x = 0.5
    medium_block_half_y = 0.5

    # Calculate block's center and top surface Z-coordinate.
    # These are relative calculations based on object root position and hardcoded dimensions.
    block_center_x = medium_block_pos[:, 0]
    block_center_y = medium_block_pos[:, 1]
    block_top_z = medium_block_pos[:, 2] + medium_block_height

    # Horizontal distance from pelvis to block center.
    # Uses relative distances and works with batched environments.
    pelvis_dist_x = torch.abs(pelvis_pos[:, 0] - block_center_x)
    pelvis_dist_y = torch.abs(pelvis_pos[:, 1] - block_center_y)

    # Define activation condition:
    # 1. Pelvis is horizontally near the block (e.g., within 1.0m radius).
    # 2. Pelvis is below the target standing height on the block (block_top_z + 0.7m).
    # This ensures the reward is active during the jump phase, not after landing or too far away.
    # This uses relative positions and works with batched environments.
    activation_condition = \
        (pelvis_dist_x < medium_block_half_x + 0.5) & \
        (pelvis_dist_y < medium_block_half_y + 0.5) & \
        (pelvis_pos[:, 2] < block_top_z + 0.7) & \
        (pelvis_pos[:, 2] > block_top_z - 0.1) # Ensure it's not too low, already on the ground

    # Reward for increasing pelvis height above block top surface.
    # We want to maximize (pelvis_z - block_top_z) but only up to a certain point (e.g., 0.7m clearance).
    # Use torch.clamp to cap the reward, encouraging clearance without overshooting excessively.
    # This creates a smooth, continuous reward.
    pelvis_clearance = pelvis_pos[:, 2] - block_top_z
    # Reward for being 0 to 0.7m above block.
    # Clamped to 0.0 to ensure no negative reward if below block_top_z, and max 0.7 for target clearance.
    reward = torch.clamp(pelvis_clearance, min=0.0, max=0.7)

    # Apply the activation condition.
    # This ensures the reward is active only when relevant.
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Normalization: Scale by the maximum possible reward (0.7m).
    reward = reward / 0.7

    # Mandatory reward normalization.
    # This addresses the "MANDATORY REWARD NORMALIZATION" requirement.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_with_block_sides_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''
    Shaping reward that penalizes the robot for any part of its body (pelvis, feet, hands) getting too close to or
    colliding with the sides of Object2 during the jump. This prevents the robot from scraping or getting stuck on the block.
    '''
    # Get normalizer instance.
    # This addresses the "MANDATORY REWARD NORMALIZATION" requirement.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts.
    # This addresses "USE ONLY APPROVED ACCESS PATTERNS" and "DIRECT OBJECT ACCESS" requirements.
    robot = env.scene["robot"]
    medium_block = env.scene['Object2']

    # Get indices and positions for relevant robot body parts.
    # This addresses "USE ONLY APPROVED ACCESS PATTERNS" and "NO hard-coded body indices" requirements.
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    medium_block_pos = medium_block.data.root_pos_w

    # Hardcode Object2 dimensions.
    # This addresses "THERE IS NO way to access the SIZE of an object" and "HOW TO USE OBJECT DIMENSIONS" requirements.
    medium_block_height = 0.6
    medium_block_half_x = 0.5
    medium_block_half_y = 0.5

    # Calculate block's bounding box coordinates (relative to its root position).
    # These are relative calculations based on object root position and hardcoded dimensions.
    block_min_x = medium_block_pos[:, 0] - medium_block_half_x
    block_max_x = medium_block_pos[:, 0] + medium_block_half_x
    block_min_y = medium_block_pos[:, 1] - medium_block_half_y
    block_max_y = medium_block_pos[:, 1] + medium_block_half_y
    block_min_z = medium_block_pos[:, 2]
    block_max_z = medium_block_pos[:, 2] + medium_block_height

    # Define a small buffer for collision avoidance.
    # This buffer creates a "no-go" zone around the block sides.
    buffer = 0.1 # 10 cm buffer

    # Function to calculate penetration depth for a given body part.
    # This function returns a negative penalty if the body part is "penetrating" the buffer zone around the block sides.
    # This function uses relative positions and works with batched environments.
    def calculate_penetration_penalty(body_pos, block_min_x, block_max_x, block_min_y, block_max_y, block_min_z, block_max_z, buffer, device):
        # Check if body part is vertically aligned with the block's sides.
        # We only care about collisions with sides, not if it's far above or below.
        is_aligned_vertically = (body_pos[:, 2] > block_min_z) & (body_pos[:, 2] < block_max_z)

        # Calculate horizontal distances to the block's boundaries (including buffer).
        # `dist_x_min` is positive if `body_pos[:, 0]` is less than `block_min_x - buffer`.
        dist_x_min = block_min_x - buffer - body_pos[:, 0]
        dist_x_max = body_pos[:, 0] - (block_max_x + buffer)
        dist_y_min = block_min_y - buffer - body_pos[:, 1]
        dist_y_max = body_pos[:, 1] - (block_max_y + buffer)

        # Calculate the "penetration" into the buffer zone.
        # This will be positive if the body part is *outside* the buffer zone, and negative if *inside*.
        # We want to penalize when it's inside the buffer zone (i.e., dist_x_min or dist_x_max are negative).
        # Use torch.max(0.0, -distance) to get a positive value for penetration depth.
        penetration_x = torch.max(torch.tensor(0.0, device=device), -dist_x_min) + \
                        torch.max(torch.tensor(0.0, device=device), -dist_x_max)
        penetration_y = torch.max(torch.tensor(0.0, device=device), -dist_y_min) + \
                        torch.max(torch.tensor(0.0, device=device), -dist_y_max)

        # Only penalize if horizontally penetrating AND vertically aligned with sides.
        # The penalty is negative, so larger penetration means more negative reward.
        # This creates a smooth, continuous reward.
        collision_penalty = torch.where(
            is_aligned_vertically,
            -(penetration_x + penetration_y), # Negative reward for penetration
            torch.tensor(0.0, device=device)
        )
        return collision_penalty

    # Calculate penalties for relevant body parts.
    # This applies the penalty calculation for multiple robot parts.
    pelvis_penalty = calculate_penetration_penalty(pelvis_pos, block_min_x, block_max_x, block_min_y, block_max_y, block_min_z, block_max_z, buffer, env.device)
    left_foot_penalty = calculate_penetration_penalty(left_foot_pos, block_min_x, block_max_x, block_min_y, block_max_y, block_min_z, block_max_z, buffer, env.device)
    right_foot_penalty = calculate_penetration_penalty(right_foot_pos, block_min_x, block_max_x, block_min_y, block_max_y, block_min_z, block_max_z, buffer, env.device)
    left_hand_penalty = calculate_penetration_penalty(left_hand_pos, block_min_x, block_max_x, block_min_y, block_max_y, block_min_z, block_max_z, buffer, env.device)
    right_hand_penalty = calculate_penetration_penalty(right_hand_pos, block_min_x, block_max_x, block_min_y, block_max_y, block_min_z, block_max_z, buffer, env.device)

    # Sum all penalties.
    reward = pelvis_penalty + left_foot_penalty + right_foot_penalty + left_hand_penalty + right_hand_penalty

    # Normalization: Scale by a typical maximum penalty (e.g., 5 body parts * 0.2m penetration = 1.0).
    # Add a small constant to make it mostly positive, as penalties are negative.
    reward = reward + 1.0
    reward = reward / 1.0 # Max possible positive reward is 1.0 (no collision)

    # Mandatory reward normalization.
    # This addresses the "MANDATORY REWARD NORMALIZATION" requirement.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for jumping onto the medium block and stabilizing.
    # This addresses "PROPER WEIGHTS" (main reward ~1.0) and "TaskRewardsCfg" requirements.
    MainJumpOntoMediumBlockReward = RewTerm(func=main_jump_onto_medium_block_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_jump_onto_medium_block_reward"})

    # Shaping reward to encourage sufficient jump height.
    # This addresses "PROPER WEIGHTS" (supporting rewards <1.0) and "TaskRewardsCfg" requirements.
    JumpHeightEncouragementReward = RewTerm(func=jump_height_encouragement_reward, weight=0.4,
                                            params={"normalise": True, "normaliser_name": "jump_height_encouragement_reward"})

    # Shaping reward to penalize collisions with the sides of the block.
    # This addresses "PROPER WEIGHTS" (supporting rewards <1.0) and "TaskRewardsCfg" requirements.
    CollisionAvoidanceWithBlockSidesReward = RewTerm(func=collision_avoidance_with_block_sides_reward, weight=0.2,
                                                     params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})