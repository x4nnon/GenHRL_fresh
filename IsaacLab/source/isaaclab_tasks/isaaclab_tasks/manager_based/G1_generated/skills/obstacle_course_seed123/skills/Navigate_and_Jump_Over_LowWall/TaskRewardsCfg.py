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


def primary_navigate_jump_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_navigate_jump_reward") -> torch.Tensor:
    """
    This reward guides the robot to approach the low wall, jump over it, and land in the correct zone
    between the low wall (Object3) and the large sphere (Object1), while maintaining stability.
    It combines progress towards the wall, clearance over the wall, and landing in the target area.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and objects using approved patterns
    robot = env.scene["robot"]
    object_low_wall = env.scene['Object3']
    object_large_sphere = env.scene['Object1']

    # Access robot part positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2]
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2]
    # lowest_foot_z is not directly used in the reward calculation but can be useful for conditions.

    # Access object positions using approved patterns
    low_wall_x = object_low_wall.data.root_pos_w[:, 0]
    low_wall_z = object_low_wall.data.root_pos_w[:, 2]
    large_sphere_x = object_large_sphere.data.root_pos_w[:, 0]

    # Hardcoded object dimensions from the task description (Rule 8)
    low_wall_height = 0.5 # From object config: 0.5m in z
    low_wall_x_dim = 0.3  # From object config: 0.3m in x

    # Phase 1: Approach the wall (pelvis x < low_wall_x)
    # Reward for reducing the x-distance to the wall's center.
    # This is a continuous reward that becomes more positive as the robot gets closer.
    approach_reward = -torch.abs(pelvis_pos_x - low_wall_x)
    # Condition for when the robot is before or at the start of the wall's x-extent.
    approach_condition = pelvis_pos_x < low_wall_x + low_wall_x_dim / 2.0

    # Phase 2: Jump over the wall (pelvis x around low_wall_x, pelvis z > low_wall_z + clearance)
    # Reward for increasing pelvis height over the wall's top surface.
    # The target height is wall_z + half_wall_height + 0.2m clearance.
    # This is a continuous reward, higher pelvis_z means higher reward.
    target_jump_z = low_wall_z + low_wall_height / 2.0 + 0.2 # Target z for pelvis to clear wall
    jump_height_reward = (pelvis_pos_z - target_jump_z) * 2.0 # Scale to make it more impactful
    # Condition for when the robot is within the x-extent of the wall, plus a small buffer for jump initiation/landing.
    jump_height_condition = (pelvis_pos_x >= low_wall_x - low_wall_x_dim / 2.0 - 0.2) & \
                            (pelvis_pos_x <= low_wall_x + low_wall_x_dim / 2.0 + 0.2)

    # Phase 3: Land in target zone (pelvis x > low_wall_x, pelvis x < large_sphere_x)
    # Reward for being in the target landing zone (midpoint between low wall and large sphere).
    # This is a continuous reward, becoming more positive as robot approaches the midpoint.
    # The target landing x is calculated relative to the low wall and large sphere positions.
    target_landing_x = low_wall_x + low_wall_x_dim / 2.0 + (large_sphere_x - (low_wall_x + low_wall_x_dim / 2.0)) / 2.0
    target_zone_x_reward = -torch.abs(pelvis_pos_x - target_landing_x)
    # Reward for maintaining a stable standing height (0.7m) after landing.
    # This is a continuous reward, penalizing deviation from 0.7m.
    target_zone_z_reward = -torch.abs(pelvis_pos_z - 0.7)
    # Condition for when the robot is past the low wall's x-extent.
    target_zone_condition = pelvis_pos_x > low_wall_x + low_wall_x_dim / 2.0

    # Combine rewards based on phases using torch.where for smooth transitions.
    # Prioritize jump height when at the wall, then landing zone, otherwise approach.
    reward = torch.zeros_like(pelvis_pos_x) # Initialize reward tensor

    # If robot is past the wall, prioritize landing zone reward.
    # This condition is checked first to ensure landing reward takes precedence once past the wall.
    reward = torch.where(target_zone_condition, target_zone_x_reward + target_zone_z_reward, reward)
    # If robot is at the wall, prioritize jump height.
    # This condition is checked second, applying if not already in the target zone.
    reward = torch.where(jump_height_condition, jump_height_reward, reward)
    # If robot is before the wall, prioritize approach.
    # This condition is checked last, applying if neither jump nor target zone conditions are met.
    reward = torch.where(approach_condition, approach_reward, reward)

    # Ensure all rewards are continuous and positive where possible, or negative for penalties.
    # The current structure uses negative distances for approach/landing and positive for height gain.

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_collision_avoidance_reward") -> torch.Tensor:
    """
    This reward penalizes the robot if any part of its body (specifically pelvis or feet) collides with the low wall
    during the jump. It also encourages clearing the wall by a small margin.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object using approved patterns
    robot = env.scene["robot"]
    object_low_wall = env.scene['Object3']

    # Access robot part positions
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Access object position
    low_wall_pos = object_low_wall.data.root_pos_w

    # Hardcoded object dimensions from the task description (Rule 8)
    low_wall_x_dim = 0.3 # From object config
    low_wall_y_dim = 5.0 # From object config
    low_wall_z_dim = 0.5 # From object config

    # Define wall boundaries for collision check based on relative positions (Rule 1)
    # These are relative to the wall's center position.
    wall_x_min = low_wall_pos[:, 0] - low_wall_x_dim / 2.0
    wall_x_max = low_wall_pos[:, 0] + low_wall_x_dim / 2.0
    wall_y_min = low_wall_pos[:, 1] - low_wall_y_dim / 2.0
    wall_y_max = low_wall_pos[:, 1] + low_wall_y_dim / 2.0
    wall_z_min = low_wall_pos[:, 2] - low_wall_z_dim / 2.0
    wall_z_max = low_wall_pos[:, 2] + low_wall_z_dim / 2.0

    # Check for pelvis collision (Rule 6: separate x, y, z components)
    pelvis_collision_x = (pelvis_pos[:, 0] > wall_x_min) & (pelvis_pos[:, 0] < wall_x_max)
    pelvis_collision_y = (pelvis_pos[:, 1] > wall_y_min) & (pelvis_pos[:, 1] < wall_y_max)
    pelvis_collision_z = (pelvis_pos[:, 2] > wall_z_min) & (pelvis_pos[:, 2] < wall_z_max)
    pelvis_colliding = pelvis_collision_x & pelvis_collision_y & pelvis_collision_z

    # Check for left foot collision
    left_foot_collision_x = (left_foot_pos[:, 0] > wall_x_min) & (left_foot_pos[:, 0] < wall_x_max)
    left_foot_collision_y = (left_foot_pos[:, 1] > wall_y_min) & (left_foot_pos[:, 1] < wall_y_max)
    left_foot_collision_z = (left_foot_pos[:, 2] > wall_z_min) & (left_foot_pos[:, 2] < wall_z_max)
    left_foot_colliding = left_foot_collision_x & left_foot_collision_y & left_foot_collision_z

    # Check for right foot collision
    right_foot_collision_x = (right_foot_pos[:, 0] > wall_x_min) & (right_foot_pos[:, 0] < wall_x_max)
    right_foot_collision_y = (right_foot_pos[:, 1] > wall_y_min) & (right_foot_pos[:, 1] < wall_y_max)
    right_foot_collision_z = (right_foot_pos[:, 2] > wall_z_min) & (right_foot_pos[:, 2] < wall_z_max)
    right_foot_colliding = right_foot_collision_x & right_foot_collision_y & right_foot_collision_z

    # Any part colliding results in a penalty
    any_part_colliding = pelvis_colliding | left_foot_colliding | right_foot_colliding
    # Continuous penalty: -5.0 if colliding, 0.0 otherwise.
    collision_reward = torch.where(any_part_colliding, torch.tensor(-5.0, device=env.device), torch.tensor(0.0, device=env.device))

    # Encourage clearance over the wall (positive reward for being above wall top when x is over wall)
    # This is a continuous reward that encourages being above the wall's top surface when the robot's x-position
    # is within the wall's x-extent.
    clearance_condition = (pelvis_pos[:, 0] > wall_x_min) & (pelvis_pos[:, 0] < wall_x_max)
    # Reward is (pelvis_z - (wall_top_z + buffer)). Positive if above buffer, negative if below.
    clearance_reward = torch.where(clearance_condition, (pelvis_pos[:, 2] - (wall_z_max + 0.1)), torch.tensor(0.0, device=env.device)) # 0.1m buffer

    reward = collision_reward + clearance_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_posture_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_posture_stability_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain a stable, upright posture (pelvis z-height around 0.7m) when not actively jumping,
    and to ensure both feet are off the ground during the jump phase.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object using approved patterns
    robot = env.scene["robot"]
    object_low_wall = env.scene['Object3']

    # Access robot part positions
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2]
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2]

    # Access object position
    low_wall_x = object_low_wall.data.root_pos_w[:, 0]

    # Hardcoded object dimensions from the task description (Rule 8)
    low_wall_x_dim = 0.3 # From object config

    # Reward for maintaining standing height when not jumping (before and after wall)
    # Condition: robot is far from the wall's x-extent (0.5m buffer).
    standing_height_condition = (pelvis_pos_x < low_wall_x - low_wall_x_dim / 2.0 - 0.5) | \
                                (pelvis_pos_x > low_wall_x + low_wall_x_dim / 2.0 + 0.5)
    # Continuous reward: penalizes deviation from target standing height (0.7m).
    standing_height_reward = -torch.abs(pelvis_pos_z - 0.7)
    reward_standing = torch.where(standing_height_condition, standing_height_reward, torch.tensor(0.0, device=env.device))

    # Reward for both feet being off the ground during the jump phase
    # Condition: robot is in the vicinity of the wall (jump phase, 0.5m buffer around wall's x-extent).
    jump_phase_condition = (pelvis_pos_x >= low_wall_x - low_wall_x_dim / 2.0 - 0.5) & \
                           (pelvis_pos_x <= low_wall_x + low_wall_x_dim / 2.0 + 0.5)
    # Continuous reward: positive if both feet are above a small threshold (0.1m), negative otherwise.
    feet_off_ground_reward = torch.where(
        (left_foot_pos_z > 0.1) & (right_foot_pos_z > 0.1), # Both feet above a small threshold
        torch.tensor(0.5, device=env.device), # Positive reward for being airborne
        torch.tensor(-0.5, device=env.device) # Penalty if not airborne during jump
    )
    reward_feet_off_ground = torch.where(jump_phase_condition, feet_off_ground_reward, torch.tensor(0.0, device=env.device))

    reward = reward_standing + reward_feet_off_ground

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
    # Primary reward for navigating, jumping, and landing. Weight 1.0 as it's the main objective.
    PrimaryNavigateJumpReward = RewTerm(func=primary_navigate_jump_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "primary_navigate_jump_reward"})

    # Shaping reward for avoiding collisions with the low wall and encouraging clearance. Weight 0.4.
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.4,
                                              params={"normalise": True, "normaliser_name": "shaping_collision_avoidance_reward"})

    # Shaping reward for maintaining posture stability and ensuring feet are off the ground during jump. Weight 0.3.
    ShapingPostureStabilityReward = RewTerm(func=shaping_posture_stability_reward, weight=0.3,
                                            params={"normalise": True, "normaliser_name": "shaping_posture_stability_reward"})