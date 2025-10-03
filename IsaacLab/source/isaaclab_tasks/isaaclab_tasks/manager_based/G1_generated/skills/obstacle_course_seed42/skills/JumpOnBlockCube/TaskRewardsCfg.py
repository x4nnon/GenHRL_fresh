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


def main_jump_on_block_cube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_jump_on_block_cube_reward") -> torch.Tensor:
    """
    Main reward for the JumpOnBlockCube skill.
    Encourages the robot to approach the block, then jump and land on top of it, and finally stabilize.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (MANDATORY: env.scene['ObjectName'])
    object5 = env.scene['Object5'] # Object5 is the block cube

    # Access required robot parts (MANDATORY: robot.data.body_pos_w[:, robot.body_names.index('part_name')])
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object5 dimensions (MANDATORY: Hardcode from object configuration, NO data.size or similar)
    block_height = 0.5
    block_half_height = block_height / 2.0
    block_half_width_x = 0.5 / 2.0 # Block is 0.5m cubed
    block_half_width_y = 0.5 / 2.0

    # Calculate block's top surface Z coordinate (relative to its root position)
    block_top_z = object5.data.root_pos_w[:, 2] + block_half_height

    # Phase 1: Approach the block horizontally
    # Reward for reducing x-distance to the block's center (MANDATORY: Relative distance)
    # Using negative absolute distance for continuous reward that increases as distance decreases
    reward_approach_x = -torch.abs(pelvis_pos[:, 0] - object5.data.root_pos_w[:, 0])

    # Reward for staying aligned with the block in y-axis (MANDATORY: Relative distance)
    reward_align_y = -torch.abs(pelvis_pos[:, 1] - object5.data.root_pos_w[:, 1])

    # Phase 2 & 3: Land and stabilize on top of the block
    # Condition for being "on top" of the block (horizontally)
    # Check if left foot is within block's x-bounds
    left_foot_on_block_x = (left_foot_pos[:, 0] > (object5.data.root_pos_w[:, 0] - block_half_width_x)) & \
                           (left_foot_pos[:, 0] < (object5.data.root_pos_w[:, 0] + block_half_width_x))
    # Check if right foot is within block's x-bounds
    right_foot_on_block_x = (right_foot_pos[:, 0] > (object5.data.root_pos_w[:, 0] - block_half_width_x)) & \
                            (right_foot_pos[:, 0] < (object5.data.root_pos_w[:, 0] + block_half_width_x))

    # Check if left foot is within block's y-bounds
    left_foot_on_block_y = (left_foot_pos[:, 1] > (object5.data.root_pos_w[:, 1] - block_half_width_y)) & \
                           (left_foot_pos[:, 1] < (object5.data.root_pos_w[:, 1] + block_half_width_y))
    # Check if right foot is within block's y-bounds
    right_foot_on_block_y = (right_foot_pos[:, 1] > (object5.data.root_pos_w[:, 1] - block_half_width_y)) & \
                            (right_foot_pos[:, 1] < (object5.data.root_pos_w[:, 1] + block_half_width_y))

    # Combined horizontal condition for both feet being on the block
    feet_horizontally_on_block = left_foot_on_block_x & right_foot_on_block_x & \
                                 left_foot_on_block_y & right_foot_on_block_y

    # Reward for feet being at the correct Z height relative to the block's top surface
    # A small positive offset (e.g., 0.05m) for foot thickness/standing on top
    target_foot_z = block_top_z + 0.05
    reward_foot_z_left = -torch.abs(left_foot_pos[:, 2] - target_foot_z)
    reward_foot_z_right = -torch.abs(right_foot_pos[:, 2] - target_foot_z)
    reward_feet_z = (reward_foot_z_left + reward_foot_z_right) / 2.0 # Average reward for both feet

    # Reward for pelvis stability and height when on the block
    # Target pelvis height above block top for standing (e.g., 0.7m for a typical humanoid)
    target_pelvis_z_on_block = block_top_z + 0.7
    reward_pelvis_z_on_block = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_block)

    # Combine rewards based on phases (MANDATORY: Continuous rewards, torch.where for phase transition)
    # If feet are horizontally on the block, prioritize vertical positioning and stability
    primary_reward = torch.where(
        feet_horizontally_on_block,
        reward_feet_z + reward_pelvis_z_on_block + 5.0, # Add a bonus for being on the block
        reward_approach_x + reward_align_y
    )

    # Normalization (MANDATORY: Complete normalization implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def shaping_jump_preparation_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_jump_preparation_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to lift its pelvis and feet when it is close to the block,
    preparing for the jump. This reward is active only when the robot is within a certain horizontal range
    of the block, but not yet on top.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    object5 = env.scene['Object5']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object5 dimensions (MANDATORY: Hardcode)
    block_height = 0.5
    block_top_z = object5.data.root_pos_w[:, 2] + (block_height / 2.0)

    # Condition: Robot is close to the block (e.g., within 0.5m in x) but not yet on top
    # This condition ensures the robot is approaching the block, not past it or already on it.
    # (MANDATORY: Relative distances for conditions)
    approach_x_threshold_min = object5.data.root_pos_w[:, 0] - 0.5 # 0.5m before block center
    approach_x_threshold_max = object5.data.root_pos_w[:, 0] + 0.2 # 0.2m past block center (to allow for jump initiation)

    is_approaching_block = (pelvis_pos[:, 0] > approach_x_threshold_min) & \
                           (pelvis_pos[:, 0] < approach_x_threshold_max)

    # Ensure robot is not already on top (check feet z-height relative to block top)
    # (MANDATORY: Relative distances for conditions)
    not_on_block_yet = (left_foot_pos[:, 2] < block_top_z + 0.1) | (right_foot_pos[:, 2] < block_top_z + 0.1)

    activation_condition = is_approaching_block & not_on_block_yet

    # Reward for increasing pelvis and feet height above ground level, aiming to clear the block
    # Target height for pelvis to clear the block (e.g., block_top_z + 0.3m)
    # (MANDATORY: Relative distances for rewards)
    target_pelvis_jump_z = block_top_z + 0.3
    reward_pelvis_jump_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_jump_z)

    # Reward for feet lifting off the ground, aiming to clear the block
    # Target height for feet to clear the block (e.g., block_top_z + 0.1m)
    # (MANDATORY: Relative distances for rewards)
    target_foot_jump_z = block_top_z + 0.1
    reward_foot_jump_z_left = -torch.abs(left_foot_pos[:, 2] - target_foot_jump_z)
    reward_foot_jump_z_right = -torch.abs(right_foot_pos[:, 2] - target_foot_jump_z)
    reward_feet_jump_z = (reward_foot_jump_z_left + reward_foot_jump_z_right) / 2.0

    # Apply reward only when activation condition is met (MANDATORY: Continuous rewards)
    shaping_reward = torch.where(activation_condition, reward_pelvis_jump_z + reward_feet_jump_z, torch.tensor(0.0, device=env.device))

    # Normalization (MANDATORY: Complete normalization implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward)
        RewNormalizer.update_stats(normaliser_name, shaping_reward)
        return scaled_reward
    return shaping_reward


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 2: Collision avoidance reward. Penalizes the robot for any part of its body
    getting too close to the block cube (Object5) *before* the jump phase, or for any part of its body
    colliding with the ground.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    object5 = env.scene['Object5']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Object5 dimensions (MANDATORY: Hardcode)
    block_half_width_x = 0.5 / 2.0
    block_half_width_y = 0.5 / 2.0
    block_half_height = 0.5 / 2.0
    block_top_z = object5.data.root_pos_w[:, 2] + block_half_height

    # Calculate distances from pelvis to block surfaces (MANDATORY: Relative distances)
    # These are distances to the *center* of the block's faces, not necessarily the closest point on the surface.
    # We'll use these to check if the pelvis is "inside" the block's horizontal bounds.
    pelvis_relative_x = pelvis_pos[:, 0] - object5.data.root_pos_w[:, 0]
    pelvis_relative_y = pelvis_pos[:, 1] - object5.data.root_pos_w[:, 1]
    pelvis_relative_z = pelvis_pos[:, 2] - object5.data.root_pos_w[:, 2]

    # Condition for "before landing on top" - feet are below block_top_z + 0.1
    # (MANDATORY: Relative distances for conditions)
    not_on_block_yet = (left_foot_pos[:, 2] < block_top_z + 0.1) | (right_foot_pos[:, 2] < block_top_z + 0.1)

    # Penalize if pelvis is inside the block's horizontal bounds and below its top
    # (MANDATORY: Relative distances for conditions)
    pelvis_inside_x = (pelvis_relative_x > -block_half_width_x) & (pelvis_relative_x < block_half_width_x)
    pelvis_inside_y = (pelvis_relative_y > -block_half_width_y) & (pelvis_relative_y < block_half_width_y)
    pelvis_below_top = pelvis_pos[:, 2] < block_top_z # Pelvis Z is below the block's top surface

    collision_with_block_condition = not_on_block_yet & pelvis_inside_x & pelvis_inside_y & pelvis_below_top

    # Reward for avoiding collision with the block (negative reward for collision)
    # Use a small positive value for distance to avoid division by zero or very large negative values
    # (MANDATORY: Continuous rewards)
    # Calculate minimum distance to any horizontal face if inside horizontal bounds
    dist_to_block_surface_x = torch.abs(pelvis_relative_x) - block_half_width_x
    dist_to_block_surface_y = torch.abs(pelvis_relative_y) - block_half_width_y
    
    # Only consider negative distances (penetration) for penalty
    penalty_x = torch.clamp(dist_to_block_surface_x, max=0.0)
    penalty_y = torch.clamp(dist_to_block_surface_y, max=0.0)

    # Combine penalties for horizontal penetration
    # We want to penalize if the pelvis is inside the block's horizontal extent
    # A simple approach is to penalize based on how deep it is inside
    # Using a squared penalty for smoother gradient and stronger penalty for deeper penetration
    reward_collision_block = torch.where(
        collision_with_block_condition,
        (penalty_x**2 + penalty_y**2) * -10.0, # Scale the penalty
        torch.tensor(0.0, device=env.device)
    )

    # General ground collision avoidance for feet and hands
    # (MANDATORY: Relative distances for conditions, continuous rewards)
    ground_z = 0.0 # Assuming ground is at z=0
    foot_ground_clearance = 0.05 # Small clearance to avoid constant penalty when standing
    hand_ground_clearance = 0.05

    # Penalize if feet or hands are below ground_z + clearance
    reward_ground_collision_left_foot = torch.where(left_foot_pos[:, 2] < ground_z + foot_ground_clearance, -1.0, 0.0)
    reward_ground_collision_right_foot = torch.where(right_foot_pos[:, 2] < ground_z + foot_ground_clearance, -1.0, 0.0)
    reward_ground_collision_left_hand = torch.where(left_hand_pos[:, 2] < ground_z + hand_ground_clearance, -1.0, 0.0)
    reward_ground_collision_right_hand = torch.where(right_hand_pos[:, 2] < ground_z + hand_ground_clearance, -1.0, 0.0)

    shaping_reward = reward_collision_block + \
                     reward_ground_collision_left_foot + \
                     reward_ground_collision_right_foot + \
                     reward_ground_collision_left_hand + \
                     reward_ground_collision_right_hand

    # Normalization (MANDATORY: Complete normalization implementation)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward)
        RewNormalizer.update_stats(normaliser_name, shaping_reward)
        return scaled_reward
    return shaping_reward


@configclass
class TaskRewardsCfg:
    """
    Reward terms for the JumpOnBlockCube skill.
    """
    # Main reward for jumping on the block and stabilizing (MANDATORY: Weight ~1.0)
    MainJumpOnBlockCubeReward = RewTerm(func=main_jump_on_block_cube_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "main_jump_on_block_cube_reward"})

    # Shaping reward for jump preparation (MANDATORY: Weight < 1.0)
    ShapingJumpPreparationReward = RewTerm(func=shaping_jump_preparation_reward, weight=0.4,
                                           params={"normalise": True, "normaliser_name": "shaping_jump_preparation_reward"})

    # Shaping reward for collision avoidance (MANDATORY: Weight < 1.0)
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.2,
                                              params={"normalise": True, "normaliser_name": "shaping_collision_avoidance_reward"})