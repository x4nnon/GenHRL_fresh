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


def primary_overcome_low_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_overcome_low_wall_reward") -> torch.Tensor:
    """
    Primary reward for the OvercomeLowWall skill.
    This reward guides the robot through three phases: approaching the wall, jumping over it, and landing stably past it.
    It uses relative distances between robot parts and the low wall.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    robot = env.scene["robot"] # Accessing robot using approved pattern
    low_wall = env.scene['Object3'] # Accessing Object3 (Low wall) using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    low_wall_pos = low_wall.data.root_pos_w # Accessing object position using approved pattern

    # Hardcoded dimensions from task description (CRITICAL: Dimensions are hardcoded from object config, not accessed from object)
    low_wall_height = 0.5
    low_wall_depth = 0.3
    # large_sphere_radius = 1.0 # Not directly used for target_x, but for context of next object

    # Calculate relative distances
    # All distances are relative to the low wall's position.
    dist_pelvis_to_wall_x = pelvis_pos_x - low_wall_pos[:, 0]
    dist_pelvis_to_wall_y = pelvis_pos_y - low_wall_pos[:, 1]
    dist_pelvis_to_wall_z = pelvis_pos_z - low_wall_pos[:, 2]

    # Define target x-position after jumping: just past the wall, before the large sphere.
    # The large sphere (Object1) is 3m beyond the low wall (Object3).
    # Target landing zone is between wall and sphere. Let's aim for 1.5m past the wall's center in x.
    target_x_pos_after_wall = low_wall_pos[:, 0] + (low_wall_depth / 2.0) + 1.5 # Relative target x-position

    # Initialize phase rewards
    reward_approach = torch.zeros_like(pelvis_pos_x)
    reward_jump = torch.zeros_like(pelvis_pos_x)
    reward_land = torch.zeros_like(pelvis_pos_x)

    # Phase 1: Approach the wall (pelvis_x < wall_x - wall_depth/2 - clearance)
    # Reward for reducing x-distance to the wall's front face.
    # Aim for 0.1m before the wall's front face.
    approach_target_x = low_wall_pos[:, 0] - low_wall_depth / 2.0 - 0.1
    approach_condition = pelvis_pos_x < (low_wall_pos[:, 0] - low_wall_depth / 2.0) # Active when robot is clearly before the wall
    # Reward is higher when pelvis_x is closer to the approach_target_x
    distance_to_approach = torch.abs(pelvis_pos_x - approach_target_x)
    # Clamp the exponent to prevent overflow/underflow
    clamped_exp_arg = torch.clamp(distance_to_approach * 5.0, min=0.0, max=20.0)
    reward_approach_x = torch.exp(-clamped_exp_arg) # Exponential decay for closeness
    reward_approach = torch.where(approach_condition, reward_approach_x, 0.0)

    # Phase 2: Jump over the wall (pelvis_x around wall_x)
    # Reward for increasing pelvis and feet height above wall, and moving across wall in x.
    # Active when robot is over or very close to the wall in x, and pelvis is above ground.
    jump_x_min = low_wall_pos[:, 0] - low_wall_depth / 2.0 - 0.2 # Start considering jump slightly before wall
    jump_x_max = low_wall_pos[:, 0] + low_wall_depth / 2.0 + 0.2 # End considering jump slightly after wall
    jump_condition = (pelvis_pos_x >= jump_x_min) & \
                     (pelvis_pos_x <= jump_x_max) & \
                     (pelvis_pos_z > 0.2) # Pelvis must be off the ground to be jumping

    # Reward for pelvis height above wall top (0.1m clearance)
    pelvis_clearance_target_z = low_wall_pos[:, 2] + low_wall_height / 2.0 + 0.1
    reward_pelvis_height = torch.clamp(pelvis_pos_z - pelvis_clearance_target_z, min=0.0) * 5.0 # Positive reward for clearing

    # Reward for feet height above wall top (0.1m clearance)
    feet_clearance_target_z = low_wall_pos[:, 2] + low_wall_height / 2.0 + 0.1
    min_foot_z = torch.min(left_foot_pos_z, right_foot_pos_z)
    reward_feet_height = torch.clamp(min_foot_z - feet_clearance_target_z, min=0.0) * 5.0 # Positive reward for clearing

    # Reward for being centered on wall in x during jump (encourages crossing)
    distance_to_wall_center = torch.abs(dist_pelvis_to_wall_x)
    # Clamp the exponent to prevent overflow/underflow
    clamped_exp_arg = torch.clamp(distance_to_wall_center * 2.0, min=0.0, max=20.0)
    reward_jump_x_progress = torch.exp(-clamped_exp_arg) # Max when pelvis is at wall center

    reward_jump = torch.where(jump_condition, reward_pelvis_height + reward_feet_height + reward_jump_x_progress, 0.0)

    # Phase 3: Land and position for next skill (pelvis_x > wall_x + wall_depth/2 + clearance)
    # Reward for landing stably and being at the target x-position.
    land_condition = pelvis_pos_x > (low_wall_pos[:, 0] + low_wall_depth / 2.0 + 0.1) # Active when robot is past the wall
    # Reward for being at the target x-position after the wall
    distance_to_target_x = torch.abs(pelvis_pos_x - target_x_pos_after_wall)
    clamped_exp_arg = torch.clamp(distance_to_target_x * 5.0, min=0.0, max=20.0)
    reward_land_x = torch.exp(-clamped_exp_arg) # Exponential decay for closeness to target x

    # Reward for y-alignment with the wall's center (encourages straight path)
    distance_to_wall_y = torch.abs(dist_pelvis_to_wall_y)
    clamped_exp_arg = torch.clamp(distance_to_wall_y * 5.0, min=0.0, max=20.0)
    reward_land_y_alignment = torch.exp(-clamped_exp_arg) # Max when pelvis y is aligned with wall y

    # Reward for stable standing height (e.g., 0.7m for pelvis)
    distance_to_target_height = torch.abs(pelvis_pos_z - 0.7)
    clamped_exp_arg = torch.clamp(distance_to_target_height * 5.0, min=0.0, max=20.0)
    reward_land_pelvis_height = torch.exp(-clamped_exp_arg) # Max when pelvis z is 0.7m

    reward_land = torch.where(land_condition, reward_land_x + reward_land_y_alignment + reward_land_pelvis_height, 0.0)

    # Combine rewards from all phases
    # The phases are mutually exclusive by design of their conditions, so summing them works.
    reward = reward_approach + reward_jump + reward_land

    # Safety check: Replace any NaN/Inf values with zeros
    reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)

    # Mandatory normalization (CRITICAL: Normalization must be included in every reward function)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_low_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_low_wall_reward") -> torch.Tensor:
    """
    Penalizes the robot for colliding with the low wall (Object3) with its body parts.
    This encourages clearing the wall cleanly.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    robot = env.scene["robot"] # Accessing robot using approved pattern
    low_wall = env.scene['Object3'] # Accessing Object3 (Low wall) using approved pattern

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    low_wall_pos = low_wall.data.root_pos_w # Accessing object position using approved pattern

    # Hardcoded dimensions from task description (CRITICAL: Dimensions are hardcoded from object config)
    low_wall_height = 0.5
    low_wall_depth = 0.3
    low_wall_width = 5.0 # From task description

    # Define wall's bounding box edges relative to its center
    wall_x_min = low_wall_pos[:, 0] - low_wall_depth / 2.0
    wall_x_max = low_wall_pos[:, 0] + low_wall_depth / 2.0
    wall_y_min = low_wall_pos[:, 1] - low_wall_width / 2.0
    wall_y_max = low_wall_pos[:, 1] + low_wall_width / 2.0
    wall_z_min = low_wall_pos[:, 2] - low_wall_height / 2.0
    wall_z_max = low_wall_pos[:, 2] + low_wall_height / 2.0

    # Collision check for pelvis (CRITICAL: Using relative distances for collision detection)
    pelvis_collision_x = (pelvis_pos[:, 0] > wall_x_min) & (pelvis_pos[:, 0] < wall_x_max)
    pelvis_collision_y = (pelvis_pos[:, 1] > wall_y_min) & (pelvis_pos[:, 1] < wall_y_max)
    pelvis_collision_z = (pelvis_pos[:, 2] > wall_z_min) & (pelvis_pos[:, 2] < wall_z_max)
    pelvis_colliding = pelvis_collision_x & pelvis_collision_y & pelvis_collision_z

    # Collision check for left foot
    left_foot_collision_x = (left_foot_pos[:, 0] > wall_x_min) & (left_foot_pos[:, 0] < wall_x_max)
    left_foot_collision_y = (left_foot_pos[:, 1] > wall_y_min) & (left_foot_pos[:, 1] < wall_y_max)
    left_foot_collision_z = (left_foot_pos[:, 2] > wall_z_min) & (left_foot_pos[:, 2] < wall_z_max)
    left_foot_colliding = left_foot_collision_x & left_foot_collision_y & left_foot_collision_z

    # Collision check for right foot
    right_foot_collision_x = (right_foot_pos[:, 0] > wall_x_min) & (right_foot_pos[:, 0] < wall_x_max)
    right_foot_collision_y = (right_foot_pos[:, 1] > wall_y_min) & (right_foot_pos[:, 1] < wall_y_max)
    right_foot_collision_z = (right_foot_pos[:, 2] > wall_z_min) & (right_foot_pos[:, 2] < wall_z_max)
    right_foot_colliding = right_foot_collision_x & right_foot_collision_y & right_foot_collision_z

    # Negative reward for collision (CRITICAL: Continuous reward, penalty for being inside the wall)
    # A larger penalty for collision to strongly discourage it.
    collision_penalty = -10.0
    reward = torch.where(pelvis_colliding | left_foot_colliding | right_foot_colliding, collision_penalty, 0.0)

    # Safety check: Replace any NaN/Inf values with zeros
    reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "maintain_upright_posture_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain an upright and stable posture throughout the skill, especially during landing.
    This is crucial for stability and preparing for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    pelvis_pos_z = pelvis_pos[:, 2]

    # Encourage pelvis to be at a stable height (e.g., 0.7m)
    # This is a continuous reward, always active. Reward is higher when pelvis_z is closer to 0.7m.
    # Using an exponential decay for smoothness and to reward closeness to target height.
    target_pelvis_height = 0.7
    distance_to_target = torch.abs(pelvis_pos_z - target_pelvis_height)
    # Clamp the exponent to prevent overflow/underflow
    clamped_exp_arg = torch.clamp(distance_to_target * 5.0, min=0.0, max=20.0)
    reward = torch.exp(-clamped_exp_arg) # CRITICAL: Continuous and positive reward

    # Safety check: Replace any NaN/Inf values with zeros
    reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)

    # Mandatory normalization
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
    Reward terms for the OvercomeLowWall skill.
    """
    # Primary reward for guiding the robot through the approach, jump, and land phases.
    # Weight 1.0 as it's the main objective.
    PrimaryOvercomeLowWallReward = RewTerm(func=primary_overcome_low_wall_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "primary_overcome_low_wall_reward"})

    # Shaping reward for avoiding collisions with the low wall.
    # Weight 0.6 to provide a significant penalty without overshadowing the primary goal.
    CollisionAvoidanceLowWallReward = RewTerm(func=collision_avoidance_low_wall_reward, weight=0.4,
                                              params={"normalise": True, "normaliser_name": "collision_avoidance_low_wall_reward"})

    # Shaping reward for maintaining an upright posture.
    # Weight 0.2 to encourage stability without being too restrictive.
    MaintainUprightPostureReward = RewTerm(func=maintain_upright_posture_reward, weight=0.3,
                                           params={"normalise": True, "normaliser_name": "maintain_upright_posture_reward"})