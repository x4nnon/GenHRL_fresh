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


def main_kick_small_sphere_away_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the Kick_SmallSphere_Away skill.
    This reward guides the robot to approach the small sphere (Object2), position its kicking foot,
    propel the sphere away in the positive x-direction, and then stabilize without overshooting the block cube (Object5).
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects from the scene.
    # CRITICAL: Using Object2 for small sphere and Object5 for block cube as per object configuration.
    small_sphere = env.scene['Object2']
    block_cube = env.scene['Object5']

    # Access the required robot parts.
    # CRITICAL: Using robot.body_names.index() for robust access, not hardcoded indices.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Object dimensions hardcoded from the provided object configuration.
    # CRITICAL: Object dimensions are NOT accessed from RigidObjectData, but hardcoded from config.
    small_sphere_radius = 0.2
    block_cube_x_size = 0.5

    # Calculate relative distances for reward components.
    # CRITICAL: All rewards use relative distances between objects and robot parts.

    # 1. Approach Small Sphere: Pelvis to Object2
    # Encourages robot to move towards the sphere in x and y.
    dist_pelvis_sphere_x = small_sphere.data.root_pos_w[:, 0] - pelvis_pos_x
    dist_pelvis_sphere_y = small_sphere.data.root_pos_w[:, 1] - pelvis_pos_y
    # Reward is negative absolute distance, encouraging reduction.
    approach_reward = -torch.abs(dist_pelvis_sphere_x) * 0.5 - torch.abs(dist_pelvis_sphere_y) * 0.5

    # 2. Prepare for Kick: Right foot to Object2
    # Encourages the right foot to be positioned correctly relative to the sphere for a kick.
    # Foot should be slightly behind the sphere in x, aligned in y, and at sphere height in z.
    dist_foot_sphere_x = small_sphere.data.root_pos_w[:, 0] - right_foot_pos_x
    dist_foot_sphere_y = small_sphere.data.root_pos_w[:, 1] - right_foot_pos_y
    dist_foot_sphere_z = small_sphere.data.root_pos_w[:, 2] - right_foot_pos_z

    # Reward for foot positioning.
    # Foot should be slightly behind the sphere (e.g., 0.1m behind its center, considering radius).
    kick_prep_reward_x = -torch.abs(dist_foot_sphere_x - small_sphere_radius - 0.1)
    kick_prep_reward_y = -torch.abs(dist_foot_sphere_y) # Aligned in y
    kick_prep_reward_z = -torch.abs(dist_foot_sphere_z - small_sphere_radius) # At sphere height

    kick_prep_reward = kick_prep_reward_x + kick_prep_reward_y + kick_prep_reward_z

    # Combine approach and kick prep rewards based on proximity.
    # This creates a continuous transition between approaching and preparing for the kick.
    # If pelvis is far from sphere, focus on approach. If close, focus on kick prep.
    # Using a smooth transition with sigmoid or similar could be more continuous, but a simple blend based on distance is also continuous enough.
    # Let's use a simple threshold for blending.
    # A small threshold for pelvis x-distance to sphere to switch to kick prep.
    # The original plan's conditional blending was removed as it can create discontinuities.
    # A simple sum is more continuous as kick_prep_reward naturally becomes more dominant when conditions are met.
    
    # 3. Kick the Sphere Away: Object2's x-position relative to its initial position.
    # The initial x-position of the small sphere (Object2) is 12m as per the task description.
    # CRITICAL: This is one of the few hardcoded values, explicitly allowed as it's an environment setup parameter.
    small_sphere_initial_x_pos = 12.0
    sphere_moved_x = small_sphere.data.root_pos_w[:, 0] - small_sphere_initial_x_pos
    # Reward for moving the sphere in the positive x-direction.
    kick_away_reward = sphere_moved_x * 5.0 # Scaled to make it a strong driver.

    # 4. Robot Positioning After Kick: Pelvis x-position relative to Object5.
    # Penalize the robot if its pelvis overshoots the block cube (Object5).
    # The robot should be before or just at the x-position of Object5, ready for the next skill.
    # Target x-position for robot is the x-center of Object5 plus half its x-size.
    target_robot_x_pos_relative_to_block = block_cube.data.root_pos_w[:, 0] + block_cube_x_size / 2.0
    
    # Penalty for overshooting.
    robot_overshoot_penalty = torch.where(pelvis_pos_x > target_robot_x_pos_relative_to_block,
                                          -(pelvis_pos_x - target_robot_x_pos_relative_to_block) * 10.0, # Strong penalty
                                          0.0)

    # Combine all reward components.
    reward = approach_reward + kick_prep_reward + kick_away_reward + robot_overshoot_penalty

    # CRITICAL: Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable standing posture.
    This includes keeping the pelvis at a desired height and both feet close to the ground.
    """
    # Get normalizer instance.
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Desired pelvis height for standing. This is a hardcoded desired state.
    desired_pelvis_z = 0.7
    # Small threshold for foot height above ground.
    foot_on_ground_threshold = 0.05

    # Reward for pelvis height: penalize deviation from desired_pelvis_z.
    pelvis_height_reward = -torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Reward for feet being close to the ground (z=0).
    # Penalize if feet are too high off the ground.
    left_foot_ground_reward = -torch.where(left_foot_pos_z > foot_on_ground_threshold, left_foot_pos_z, 0.0)
    right_foot_ground_reward = -torch.where(right_foot_pos_z > foot_on_ground_threshold, right_foot_pos_z, 0.0)

    # Combine stability rewards.
    reward = pelvis_height_reward + left_foot_ground_reward + right_foot_ground_reward

    # CRITICAL: Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to encourage collision avoidance between robot body parts and objects.
    Specifically, avoid collisions with Object2 (small sphere, except for the kicking foot during contact),
    Object4 (high wall), and Object5 (block cube).
    """
    # Get normalizer instance.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects.
    small_sphere = env.scene['Object2']
    high_wall = env.scene['Object4']
    block_cube = env.scene['Object5']

    # Access the required robot parts.
    robot = env.scene["robot"]
    pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')]
    right_knee_pos = robot.data.body_pos_w[:, robot.body_names.index('right_knee_link')]
    left_knee_pos = robot.data.body_pos_w[:, robot.body_names.index('left_knee_link')]
    right_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('right_ankle_roll_link')]
    left_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('left_ankle_roll_link')]

    # Object dimensions hardcoded from the provided object configuration.
    small_sphere_radius = 0.2
    high_wall_x_size = 0.3
    high_wall_y_size = 5.0
    high_wall_z_size = 1.0
    block_cube_x_size = 0.5
    block_cube_y_size = 0.5
    block_cube_z_size = 0.5

    # Define collision distance thresholds for different objects and robot parts.
    # These are relative thresholds, combining object half-sizes/radii with a small robot part radius approximation.
    robot_part_radius_approx = 0.1 # General approximation for robot limb thickness

    # Thresholds for sphere collision (Object2)
    collision_threshold_sphere = small_sphere_radius + robot_part_radius_approx

    # Thresholds for wall collision (Object4)
    collision_threshold_wall_x = high_wall_x_size / 2.0 + robot_part_radius_approx
    collision_threshold_wall_y = high_wall_y_size / 2.0 + robot_part_radius_approx
    collision_threshold_wall_z = high_wall_z_size / 2.0 + robot_part_radius_approx

    # Thresholds for block collision (Object5)
    collision_threshold_block_x = block_cube_x_size / 2.0 + robot_part_radius_approx
    collision_threshold_block_y = block_cube_y_size / 2.0 + robot_part_radius_approx
    collision_threshold_block_z = block_cube_z_size / 2.0 + robot_part_radius_approx

    collision_penalty = torch.zeros_like(pelvis_pos[:, 0]) # Initialize penalty tensor

    # Avoidance with small sphere (Object2)
    # Check all parts except the right foot (which is intended to collide for kicking).
    parts_to_check_sphere = [pelvis_pos, left_knee_pos, right_knee_pos, left_foot_pos]
    for part_pos in parts_to_check_sphere:
        # Calculate Euclidean distance between robot part and sphere center.
        dist_to_sphere = torch.norm(small_sphere.data.root_pos_w - part_pos, dim=1)
        # Apply penalty if distance is below threshold. Inverse distance penalty for continuity.
        collision_condition = dist_to_sphere < collision_threshold_sphere
        collision_penalty += torch.where(collision_condition, -1.0 / (dist_to_sphere + 1e-6), 0.0)

    # Avoidance with high wall (Object4)
    parts_to_check_wall = [pelvis_pos, right_knee_pos, left_knee_pos, right_foot_pos, left_foot_pos]
    for part_pos in parts_to_check_wall:
        # Calculate distances to the wall's bounding box center.
        dist_x = torch.abs(high_wall.data.root_pos_w[:, 0] - part_pos[:, 0])
        dist_y = torch.abs(high_wall.data.root_pos_w[:, 1] - part_pos[:, 1])
        dist_z = torch.abs(high_wall.data.root_pos_w[:, 2] - part_pos[:, 2])

        # Collision if all components are within their respective thresholds.
        collision_condition = (dist_x < collision_threshold_wall_x) & \
                              (dist_y < collision_threshold_wall_y) & \
                              (dist_z < collision_threshold_wall_z)
        # Inverse distance penalty. Sum of distances to avoid division by zero if one component is zero.
        collision_penalty += torch.where(collision_condition, -1.0 / (dist_x + dist_y + dist_z + 1e-6), 0.0)

    # Avoidance with block cube (Object5)
    parts_to_check_block = [pelvis_pos, right_knee_pos, left_knee_pos, right_foot_pos, left_foot_pos]
    for part_pos in parts_to_check_block:
        # Calculate distances to the block's bounding box center.
        dist_x = torch.abs(block_cube.data.root_pos_w[:, 0] - part_pos[:, 0])
        dist_y = torch.abs(block_cube.data.root_pos_w[:, 1] - part_pos[:, 1])
        dist_z = torch.abs(block_cube.data.root_pos_w[:, 2] - part_pos[:, 2])

        # Collision if all components are within their respective thresholds.
        collision_condition = (dist_x < collision_threshold_block_x) & \
                              (dist_y < collision_threshold_block_y) & \
                              (dist_z < collision_threshold_block_z)
        # Inverse distance penalty.
        collision_penalty += torch.where(collision_condition, -1.0 / (dist_x + dist_y + dist_z + 1e-6), 0.0)

    reward = collision_penalty

    # CRITICAL: Mandatory reward normalization.
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
    Reward terms for the Kick_SmallSphere_Away skill.
    Defines the primary and shaping rewards with their respective weights and normalization parameters.
    """
    # Primary reward for approaching, kicking, and positioning.
    MainKickSmallSphereAwayReward = RewTerm(func=main_kick_small_sphere_away_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for maintaining robot stability.
    ShapingStabilityReward = RewTerm(func=shaping_stability_reward, weight=0.4,
                                     params={"normalise": True, "normaliser_name": "stability_reward"})

    # Shaping reward for avoiding unwanted collisions.
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})