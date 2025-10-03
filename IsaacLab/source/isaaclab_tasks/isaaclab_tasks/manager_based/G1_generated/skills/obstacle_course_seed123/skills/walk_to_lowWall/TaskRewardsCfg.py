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

def walk_to_lowWall_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_lowWall_primary_reward") -> torch.Tensor:
    """
    Primary reward for the walk_to_lowWall skill.
    Guides the robot to walk towards and position itself optimally in front of the low wall (Object3) for a jump.
    Combines progress in the x-axis, alignment in the y-axis, and maintaining a stable pelvis height.
    Penalizes overshooting the wall.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # Requirement 2: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object3 = env.scene['Object3'] # Low wall
    object3_pos = object3.data.root_pos_w

    # Access the required robot part(s)
    # Requirement 3: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object3 dimensions (from task description: 0.3m x-axis, 5m y-axis, 0.5m z-axis)
    # Requirement 8: There is no way to access the SIZE of an object. Hardcode from config.
    object3_x_dim = 0.3
    object3_y_dim = 5.0
    object3_z_dim = 0.5

    # Target position relative to the low wall
    # Optimal jumping distance before the wall's front face
    # Requirement 4: NEVER use hard-coded positions or arbitrary thresholds.
    # These are relative distances/heights, not absolute world coordinates.
    optimal_jump_dist_x = 0.6 # meters, relative to wall's front face
    target_pelvis_x = object3_pos[:, 0] - (object3_x_dim / 2) - optimal_jump_dist_x
    target_pelvis_y = object3_pos[:, 1] # Aligned with wall's center in Y
    target_pelvis_z = 0.7 # Stable pelvis height, relative to ground (z=0)

    # Reward for X-axis positioning: closer to target_pelvis_x
    # Requirement 1: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Requirement 10: Reward space should be as linear as possible, avoid local minima.
    # Using negative absolute difference for continuous reward.
    x_dist_reward = -torch.abs(pelvis_pos[:, 0] - target_pelvis_x)

    # Add a strong penalty if the robot's pelvis goes past the front face of the wall
    # This ensures the robot stops before the wall and doesn't overshoot.
    wall_front_x = object3_pos[:, 0] - (object3_x_dim / 2)
    overshoot_penalty = torch.where(pelvis_pos[:, 0] > wall_front_x, -10.0 * (pelvis_pos[:, 0] - wall_front_x), 0.0)
    x_dist_reward = x_dist_reward + overshoot_penalty

    # Reward for Y-axis alignment: closer to target_pelvis_y
    y_dist_reward = -torch.abs(pelvis_pos[:, 1] - target_pelvis_y)

    # Reward for Z-axis stability: closer to target_pelvis_z
    z_dist_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards with weights
    reward = (x_dist_reward * 0.5) + (y_dist_reward * 0.3) + (z_dist_reward * 0.2)

    # Requirement 6: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_lowWall_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_lowWall_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the low wall (Object3).
    Penalizes any robot part colliding with the low wall.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object3 = env.scene['Object3'] # Low wall
    object3_pos = object3.data.root_pos_w

    # Access all robot body parts for collision checking
    robot = env.scene["robot"]
    robot_body_parts = ['pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link',
                        'left_knee_link', 'right_knee_link', 'left_palm_link', 'right_palm_link',
                        'head_link']

    # Object3 dimensions (from task description: 0.3m x-axis, 5m y-axis, 0.5m z-axis)
    object3_x_dim = 0.3
    object3_y_dim = 5.0
    object3_z_dim = 0.5

    collision_penalty = torch.zeros_like(env.episode_length_buf, dtype=torch.float32)

    for part_name in robot_body_parts:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx]

        # Calculate relative distance from robot part to the center of Object3
        # Requirement 1: ALL rewards MUST ONLY use relative distances
        dist_x = torch.abs(part_pos[:, 0] - object3_pos[:, 0])
        dist_y = torch.abs(part_pos[:, 1] - object3_pos[:, 1])
        dist_z = torch.abs(part_pos[:, 2] - object3_pos[:, 2])

        # Define collision thresholds based on half-dimensions of the wall
        # A small buffer is added to make the penalty activate slightly before actual contact
        # Requirement 4: NEVER use hard-coded positions or arbitrary thresholds.
        # These thresholds are derived from object dimensions and a small buffer.
        buffer = 0.05 # meters
        threshold_x = object3_x_dim / 2.0 + buffer
        threshold_y = object3_y_dim / 2.0 + buffer
        threshold_z = object3_z_dim / 2.0 + buffer

        # Check for collision in each dimension
        # If colliding in all dimensions, apply penalty
        is_colliding = (dist_x < threshold_x) & (dist_y < threshold_y) & (dist_z < threshold_z)
        collision_penalty += torch.where(is_colliding, -5.0, 0.0) # Apply a negative reward for collision

    reward = collision_penalty

    # Requirement 6: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_lowWall_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_lowWall_stability_reward") -> torch.Tensor:
    """
    Shaping reward for feet on ground and overall stability.
    Encourages the robot to keep its feet on the ground and maintain a stable pelvis height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Target ground height for feet (assuming ground is at z=0)
    ground_z = 0.0
    # Target pelvis height for stability
    target_pelvis_z = 0.7

    # Penalty for feet being too high off the ground (e.g., > 0.1m)
    # Requirement 1: ALL rewards MUST ONLY use relative distances (z-height relative to ground)
    # Requirement 4: NEVER use hard-coded positions or arbitrary thresholds.
    # Thresholds are relative to ground_z.
    foot_lift_threshold = 0.1 # meters
    foot_off_ground_penalty_left = torch.where(left_foot_pos[:, 2] > (ground_z + foot_lift_threshold),
                                               -2.0 * (left_foot_pos[:, 2] - (ground_z + foot_lift_threshold)), 0.0)
    foot_off_ground_penalty_right = torch.where(right_foot_pos[:, 2] > (ground_z + foot_lift_threshold),
                                                -2.0 * (right_foot_pos[:, 2] - (ground_z + foot_lift_threshold)), 0.0)

    # Reward for pelvis being close to the target stable height
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards
    reward = foot_off_ground_penalty_left + foot_off_ground_penalty_right + (pelvis_height_reward * 0.5)

    # Requirement 6: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for reaching the target position in front of the low wall
    # Weight: 1.0 as it's the main objective.
    WalkToLowWallPrimaryReward = RewTerm(func=walk_to_lowWall_primary_reward, weight=1.0,
                                         params={"normalise": True, "normaliser_name": "walk_to_lowWall_primary_reward"})

    # Shaping reward for avoiding collisions with the low wall
    # Weight: 0.6 to strongly discourage collisions.
    WalkToLowWallCollisionAvoidance = RewTerm(func=walk_to_lowWall_collision_avoidance_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "walk_to_lowWall_collision_avoidance_reward"})

    # Shaping reward for maintaining stability and keeping feet on the ground
    # Weight: 0.4 to encourage stable walking.
    WalkToLowWallStability = RewTerm(func=walk_to_lowWall_stability_reward, weight=0.4,
                                     params={"normalise": True, "normaliser_name": "walk_to_lowWall_stability_reward"})