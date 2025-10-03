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


def walk_to_smallSphere_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_smallSphere_main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_smallSphere skill.
    Encourages the robot's pelvis to be positioned near the small sphere (Object2),
    ready to interact with it, specifically for kicking.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # CRITICAL RULE: ALWAYS access objects using env.scene['ObjectName']
    small_sphere = env.scene['Object2'] # Object2 is the small sphere for robot to kick

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Object dimensions from task description (hardcoded as per rules)
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from task description.
    small_sphere_radius = 0.2 # From task description: "A small sphere 0.2m radius."
    kicking_clearance = 0.15 # Desired clearance for kicking, a small buffer
    # Calculate the target x-offset from the sphere's center for the robot's pelvis
    # The robot should be behind the sphere in the x-direction to kick it forward.
    target_x_offset = small_sphere_radius + kicking_clearance

    # Calculate target positions relative to the small sphere's position
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Target X: Robot pelvis should be target_x_offset behind the sphere in x-axis
    target_pelvis_x = small_sphere.data.root_pos_w[:, 0] - target_x_offset
    # Target Y: Robot pelvis should be aligned with the sphere in y-axis
    target_pelvis_y = small_sphere.data.root_pos_w[:, 1]
    # Target Z: Stable pelvis height for the robot
    # CRITICAL RULE: Z-height can be an absolute value if it represents a stable posture.
    target_pelvis_z = 0.7 # A reasonable stable height for the robot's pelvis

    # Calculate the distance vector components between current pelvis position and target position
    # CRITICAL RULE: Use torch.abs for distance components
    distance_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    distance_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    distance_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward is negative absolute distance to encourage getting closer to the target
    # CRITICAL RULE: Rewards should be continuous and positive (or negative for penalties)
    # Penalize deviation from the target position in each dimension
    reward_x = -distance_x
    reward_y = -distance_y
    reward_z = -distance_z

    # Combine rewards for overall positioning
    reward = reward_x + reward_y + reward_z

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_smallSphere_collision_penalty(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_smallSphere_collision_penalty") -> torch.Tensor:
    """
    Shaping reward that penalizes the robot for colliding or being too close to the small sphere (Object2).
    This encourages the robot to approach carefully and maintain a safe distance until the final desired position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # CRITICAL RULE: ALWAYS access objects using env.scene['ObjectName']
    small_sphere = env.scene['Object2'] # Object2 is the small sphere

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object dimensions from task description (hardcoded as per rules)
    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from task description.
    small_sphere_radius = 0.2 # From task description
    # Define a collision threshold: sphere radius + a small buffer
    collision_threshold = small_sphere_radius + 0.05

    # Calculate Euclidean distances from robot parts to the small sphere's center
    # CRITICAL RULE: Use torch.norm for Euclidean distance
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_pelvis_sphere = torch.norm(pelvis_pos - small_sphere.data.root_pos_w, dim=-1)
    dist_left_foot_sphere = torch.norm(left_foot_pos - small_sphere.data.root_pos_w, dim=-1)
    dist_right_foot_sphere = torch.norm(right_foot_pos - small_sphere.data.root_pos_w, dim=-1)

    # Penalize if any part is too close to the sphere
    # Use a continuous penalty that increases as distance decreases below the threshold
    # CRITICAL RULE: Rewards should be continuous
    penalty_pelvis = torch.where(dist_pelvis_sphere < collision_threshold, -(collision_threshold - dist_pelvis_sphere), 0.0)
    penalty_left_foot = torch.where(dist_left_foot_sphere < collision_threshold, -(collision_threshold - dist_left_foot_sphere), 0.0)
    penalty_right_foot = torch.where(dist_right_foot_sphere < collision_threshold, -(collision_threshold - dist_right_foot_sphere), 0.0)

    # Sum the penalties
    reward = penalty_pelvis + penalty_left_foot + penalty_right_foot

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_smallSphere_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_smallSphere_posture_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain an upright posture and avoid falling.
    This is crucial for stability before the next skill (kicking).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part
    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-component of pelvis position

    # Reward for maintaining pelvis height close to the stable default (0.7m)
    # Penalize deviation from 0.7m
    # CRITICAL RULE: Rewards should be continuous
    # CRITICAL RULE: Z-height can be an absolute value if it represents a stable posture.
    reward = -torch.abs(pelvis_pos_z - 0.7)

    # Additional penalty if pelvis falls significantly below a threshold (e.g., 0.5m), indicating a fall
    fall_threshold = 0.5
    # Apply a stronger, continuous penalty if the pelvis height drops below the fall threshold
    fall_penalty = torch.where(pelvis_pos_z < fall_threshold, -10.0 * (fall_threshold - pelvis_pos_z), 0.0)
    reward = reward + fall_penalty

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
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
    Configuration for the reward terms used in the walk_to_smallSphere skill.
    """
    # Main reward for positioning the robot near the small sphere
    # CRITICAL RULE: Main reward weight ~1.0
    walk_to_smallSphere_main_reward = RewTerm(
        func=walk_to_smallSphere_main_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "walk_to_smallSphere_main_reward"}
    )

    # Shaping reward for penalizing collision/proximity to the small sphere
    # CRITICAL RULE: Supporting rewards typically <1.0
    walk_to_smallSphere_collision_penalty = RewTerm(
        func=walk_to_smallSphere_collision_penalty,
        weight=0.4, # Weight as per reward design plan
        params={"normalise": True, "normaliser_name": "walk_to_smallSphere_collision_penalty"}
    )

    # Shaping reward for maintaining upright posture and preventing falls
    # CRITICAL RULE: Supporting rewards typically <1.0
    walk_to_smallSphere_posture_reward = RewTerm(
        func=walk_to_smallSphere_posture_reward,
        weight=0.3, # A reasonable weight for posture stability
        params={"normalise": True, "normaliser_name": "walk_to_smallSphere_posture_reward"}
    )