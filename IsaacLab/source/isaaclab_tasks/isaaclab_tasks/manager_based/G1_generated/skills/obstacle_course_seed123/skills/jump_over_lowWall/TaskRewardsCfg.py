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


def jump_over_lowWall_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_over_lowWall_primary_reward") -> torch.Tensor:
    """
    Primary reward for the jump_over_lowWall skill.
    This reward guides the robot through approaching the wall, clearing it, and landing in the correct post-jump position.
    It combines three phases: approach, jump/clearance, and landing/positioning.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions using approved patterns
    robot = env.scene["robot"]
    object_low_wall = env.scene['Object3'] # Object3 is the low wall
    object_large_sphere = env.scene['Object1'] # Object1 is the large sphere

    # Get robot part positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Get object positions using approved patterns
    low_wall_x = object_low_wall.data.root_pos_w[:, 0]
    low_wall_z = object_low_wall.data.root_pos_w[:, 2]
    large_sphere_x = object_large_sphere.data.root_pos_w[:, 0]

    # Hardcode object dimensions from the configuration (as per requirements)
    low_wall_height = 0.5 # From object config: 0.5m in z
    low_wall_depth = 0.3  # From object config: 0.3m in x
    large_sphere_radius = 1.0 # From object config: 1m radius

    # --- Phase 1: Approach & Pre-Jump ---
    # Reward for getting closer to the wall in x-direction when the robot is before the wall.
    # The reward is negative absolute distance, so it increases as distance decreases.
    # Uses relative distance: torch.abs(low_wall_x - pelvis_pos_x)
    approach_x_reward = -torch.abs(low_wall_x - pelvis_pos_x)
    # Condition: pelvis is before the front edge of the low wall.
    approach_condition = (pelvis_pos_x < low_wall_x - low_wall_depth / 2)
    reward_phase1 = torch.where(approach_condition, approach_x_reward, torch.tensor(0.0, device=env.device))

    # --- Phase 2: Jump & Clearance ---
    # Reward for pelvis and feet being above wall height plus a small clearance.
    # Uses relative height: pelvis_pos_z - clearance_height_target
    clearance_height_target = low_wall_z + low_wall_height + 0.1 # Target Z-height for clearance (wall top + 0.1m)

    # Reward for pelvis clearing the wall
    # Continuous reward: positive if above target, 0 otherwise.
    pelvis_clearance_reward = torch.where(pelvis_pos_z > clearance_height_target, pelvis_pos_z - clearance_height_target, torch.tensor(0.0, device=env.device))
    # Reward for left foot clearing the wall
    left_foot_clearance_reward = torch.where(left_foot_pos_z > clearance_height_target, left_foot_pos_z - clearance_height_target, torch.tensor(0.0, device=env.device))
    # Reward for right foot clearing the wall
    right_foot_clearance_reward = torch.where(right_foot_pos_z > clearance_height_target, right_foot_pos_z - clearance_height_target, torch.tensor(0.0, device=env.device))

    # Average clearance reward for the three parts
    jump_clearance_reward = (pelvis_clearance_reward + left_foot_clearance_reward + right_foot_clearance_reward) / 3.0

    # Condition for being "over" the wall (pelvis x-position within wall's depth).
    # This ensures the clearance reward is only active when the robot is actually attempting to jump over.
    over_wall_condition = (pelvis_pos_x >= low_wall_x - low_wall_depth / 2) & (pelvis_pos_x <= low_wall_x + low_wall_depth / 2)
    reward_phase2 = torch.where(over_wall_condition, jump_clearance_reward, torch.tensor(0.0, device=env.device))

    # --- Phase 3: Landing & Post-Jump Positioning ---
    # Reward for landing (pelvis_z returning to a stable height, e.g., 0.7m)
    # And for being in the target x-range (between low wall and large sphere).
    target_pelvis_z = 0.7 # A typical stable standing height for the pelvis.

    # Reward for pelvis z-position being close to the target stable height.
    # Negative absolute difference, so reward is higher when closer to target_pelvis_z.
    # Uses relative height: -torch.abs(pelvis_pos_z - target_pelvis_z)
    landing_z_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Define the target x-range for landing: just past the low wall to just before the large sphere.
    # This ensures the robot doesn't overshoot the next skill's starting position.
    # Uses relative positions: low_wall_x + low_wall_depth / 2 + 0.1 and large_sphere_x - large_sphere_radius - 0.1
    target_x_min = low_wall_x + low_wall_depth / 2 + 0.1 # Slightly past the wall's back edge
    target_x_max = large_sphere_x - large_sphere_radius - 0.1 # Slightly before the large sphere's front edge

    # Reward for being within the target x-range, peaking at the center of the range.
    # Using a Gaussian-like reward for smooth transition and peak at the ideal spot.
    # Uses relative position: pelvis_pos_x - x_range_center
    x_range_center = (target_x_min + target_x_max) / 2.0
    x_range_width = target_x_max - target_x_min
    # Avoid division by zero if width is zero (unlikely but good practice)
    x_position_reward = torch.exp(-((pelvis_pos_x - x_range_center) / (x_range_width / 2.0 + 1e-6))**2)

    # Condition: Active when robot is past the low wall.
    post_jump_condition = (pelvis_pos_x > low_wall_x + low_wall_depth / 2)
    # Combine landing Z reward and X positioning reward for this phase.
    reward_phase3 = torch.where(post_jump_condition, landing_z_reward + x_position_reward, torch.tensor(0.0, device=env.device))

    # Combine rewards from all phases.
    reward = reward_phase1 + reward_phase2 + reward_phase3

    # Normalization (MANDATORY)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def jump_over_lowWall_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_over_lowWall_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to penalize robot parts for colliding with or getting too close to the low wall (Object3).
    This encourages a clean jump over the obstacle.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access object position using approved patterns
    object_low_wall = env.scene['Object3'] # Object3 is the low wall
    low_wall_pos = object_low_wall.data.root_pos_w
    low_wall_x = low_wall_pos[:, 0]
    low_wall_y = low_wall_pos[:, 1]
    low_wall_z = low_wall_pos[:, 2]

    # Hardcode object dimensions from the configuration (as per requirements)
    low_wall_depth = 0.3  # From object config: 0.3m in x
    low_wall_width = 5.0  # From object config: 5m in y
    low_wall_height = 0.5 # From object config: 0.5m in z

    # Define robot parts to check for collision
    robot_parts_to_check = ['pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link',
                            'left_palm_link', 'right_palm_link', 'left_knee_link', 'right_knee_link']
    collision_reward = torch.zeros_like(low_wall_x, device=env.device) # Initialize reward tensor

    # Define collision threshold (distance below which penalty applies)
    collision_threshold = 0.1 # 0.1 meters proximity

    for part_name in robot_parts_to_check:
        # Access robot part index and position using approved patterns
        part_idx = env.scene["robot"].body_names.index(part_name)
        part_pos = env.scene["robot"].data.body_pos_w[:, part_idx]
        part_pos_x = part_pos[:, 0]
        part_pos_y = part_pos[:, 1]
        part_pos_z = part_pos[:, 2]

        # Calculate distance to the closest point on the wall's bounding box.
        # This creates a continuous distance metric to the wall's surface.
        # Uses relative distances: torch.abs(part_pos_x - low_wall_x), etc.
        dx = torch.max(torch.tensor(0.0, device=env.device), torch.abs(part_pos_x - low_wall_x) - low_wall_depth / 2)
        dy = torch.max(torch.tensor(0.0, device=env.device), torch.abs(part_pos_y - low_wall_y) - low_wall_width / 2)
        dz = torch.max(torch.tensor(0.0, device=env.device), torch.abs(part_pos_z - low_wall_z) - low_wall_height / 2)

        distance_to_wall = torch.sqrt(dx**2 + dy**2 + dz**2)

        # Apply a negative reward (penalty) if the distance is below the threshold.
        # The penalty is inverse proportional to distance, making it stronger when closer.
        # Add a small epsilon to avoid division by zero.
        penalty = torch.where(distance_to_wall < collision_threshold, -1.0 / (distance_to_wall + 1e-6), torch.tensor(0.0, device=env.device))
        collision_reward += penalty

    reward = collision_reward

    # Normalization (MANDATORY)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def jump_over_lowWall_pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_over_lowWall_pelvis_stability_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a relatively stable pelvis height (around 0.7m)
    when not actively jumping over the wall. This helps prevent unnecessary crouching or falling.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions using approved patterns
    robot = env.scene["robot"]
    object_low_wall = env.scene['Object3'] # Object3 is the low wall

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    low_wall_x = object_low_wall.data.root_pos_w[:, 0]

    # Hardcode object dimensions from the configuration (as per requirements)
    low_wall_depth = 0.3 # From object config: 0.3m in x

    target_pelvis_z = 0.7 # Desired stable pelvis height

    # Condition for when this reward is active:
    # It's active when the robot's pelvis is before the wall's front edge OR after the wall's back edge.
    # This means it's inactive when the robot is directly over the wall, allowing for the jump.
    # Uses relative position: pelvis_pos_x < low_wall_x - low_wall_depth / 2, etc.
    condition_before_wall = (pelvis_pos_x < low_wall_x - low_wall_depth / 2)
    condition_after_wall = (pelvis_pos_x > low_wall_x + low_wall_depth / 2)
    stability_condition = condition_before_wall | condition_after_wall

    # Reward for being close to the target pelvis height.
    # Negative absolute difference, so reward is higher when closer to target_pelvis_z.
    # Uses relative height: -torch.abs(pelvis_pos_z - target_pelvis_z)
    stability_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Apply the reward only when the stability condition is met.
    reward = torch.where(stability_condition, stability_reward, torch.tensor(0.0, device=env.device))

    # Normalization (MANDATORY)
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
    Reward terms for the jump_over_lowWall skill.
    """
    # Primary reward for guiding the robot through the jump process
    primary_jump_reward = RewTerm(func=jump_over_lowWall_primary_reward, weight=1.0,
                                  params={"normalise": True, "normaliser_name": "jump_over_lowWall_primary_reward"})

    # Shaping reward for collision avoidance with the low wall
    collision_avoidance_reward = RewTerm(func=jump_over_lowWall_collision_avoidance_reward, weight=0.6,
                                         params={"normalise": True, "normaliser_name": "jump_over_lowWall_collision_avoidance_reward"})

    # Shaping reward for maintaining pelvis stability when not actively jumping
    pelvis_stability_reward = RewTerm(func=jump_over_lowWall_pelvis_stability_reward, weight=0.3,
                                      params={"normalise": True, "normaliser_name": "jump_over_lowWall_pelvis_stability_reward"})