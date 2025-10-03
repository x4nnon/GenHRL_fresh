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


def stabilize_after_jump_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stabilize_after_jump_primary_reward") -> torch.Tensor:
    """
    Primary reward for the 'stabilize_after_jump' skill.
    Encourages the robot to achieve a stable, upright standing position on the far side of Object3 (low wall)
    and position itself correctly for the next skill (pushing Object1).
    Combines rewards for pelvis height, feet on the ground, and appropriate x-position relative to Object3 and Object1.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3']  # Low wall
    object1 = env.scene['Object1']  # Large sphere

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
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

    # Object dimensions (hardcoded from description as per requirements)
    # Object3: wide low wall, 5m in the y-axis and 0.5m in the z and 0.3m in x axis.
    low_wall_x_dim = 0.3
    # Object1: large sphere 1m radius.
    large_sphere_radius = 1.0

    # Target values (standard stable values, not hard-coded positions)
    target_pelvis_z = 0.7
    target_foot_z = 0.05  # Small offset for feet on ground, allowing for slight ground penetration or foot thickness

    # 1. Pelvis height reward: Encourage pelvis to be at target_pelvis_z
    # Uses relative distance to target height. Continuous and negative absolute difference.
    pelvis_height_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # 2. Feet on ground reward: Encourage feet to be close to target_foot_z
    # Uses relative distance to target height. Continuous and negative absolute difference.
    feet_on_ground_reward = -torch.abs(left_foot_pos_z - target_foot_z) - torch.abs(right_foot_pos_z - target_foot_z)

    # 3. Positional alignment reward: Pelvis x-position relative to low wall and large sphere
    # Robot should be past the low wall (Object3) and before the large sphere (Object1)
    # A buffer is added to ensure clearance from the wall and readiness for the sphere.
    buffer = 0.2 # Buffer for clearance, not an arbitrary threshold for position but for relative distance.

    # Calculate the x-coordinate of the far side of the low wall (Object3)
    # Uses relative distance from object's root position and its dimension.
    low_wall_far_x = object3.data.root_pos_w[:, 0] + low_wall_x_dim / 2 + buffer

    # Calculate the x-coordinate of the near side of the large sphere (Object1)
    # Uses relative distance from object's root position and its radius.
    large_sphere_near_x = object1.data.root_pos_w[:, 0] - large_sphere_radius - buffer

    # Reward for being past the low wall: Penalize if pelvis_pos_x is less than low_wall_far_x
    # Uses torch.clamp to create a continuous penalty only when the condition is not met.
    past_low_wall_reward = -torch.abs(torch.clamp(pelvis_pos_x - low_wall_far_x, max=0.0))

    # Reward for not overshooting the large sphere: Penalize if pelvis_pos_x is greater than large_sphere_near_x
    # Uses torch.clamp to create a continuous penalty only when the condition is not met.
    before_large_sphere_reward = -torch.abs(torch.clamp(large_sphere_near_x - pelvis_pos_x, max=0.0))

    # Combine positional rewards
    position_reward = past_low_wall_reward + before_large_sphere_reward

    # Total primary reward
    reward = pelvis_height_reward + feet_on_ground_reward + position_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def stabilize_after_jump_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stabilize_after_jump_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 1: Collision avoidance with Object3 (low wall) and Object1 (large sphere).
    Penalizes the robot if any part of its body (pelvis, feet, hands) gets too close or collides with the low wall or the large sphere after landing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object3 = env.scene['Object3']  # Low wall
    object1 = env.scene['Object1']  # Large sphere

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')]
    left_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('left_ankle_roll_link')]
    right_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('right_ankle_roll_link')]
    left_hand_pos = robot.data.body_pos_w[:, robot.body_names.index('left_palm_link')]
    right_hand_pos = robot.data.body_pos_w[:, robot.body_names.index('right_palm_link')]

    # Object dimensions (hardcoded from description as per requirements)
    # Object3: wide low wall, 5m in the y-axis and 0.5m in the z and 0.3m in x axis.
    low_wall_x_dim = 0.3
    low_wall_y_dim = 5.0
    low_wall_z_dim = 0.5
    # Object1: large sphere 1m radius.
    large_sphere_radius = 1.0

    # Collision threshold (a small buffer for collision detection, not an arbitrary threshold for position)
    collision_threshold = 0.15 # Represents a small margin around robot parts for collision detection

    # Initialize collision penalty
    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # --- Collision with Object3 (low wall) ---
    # Low wall's center position
    obj3_center = object3.data.root_pos_w

    # List of robot parts to check
    robot_parts_pos = [pelvis_pos, left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos]

    for part_pos in robot_parts_pos:
        # Calculate relative distances to the center of the low wall for each axis
        dist_x = torch.abs(part_pos[:, 0] - obj3_center[:, 0])
        dist_y = torch.abs(part_pos[:, 1] - obj3_center[:, 1])
        dist_z = torch.abs(part_pos[:, 2] - obj3_center[:, 2])

        # Check for collision with Object3 (low wall) using AABB-like proximity
        # Collision occurs if any dimension's distance is less than half the object's dimension plus threshold
        is_colliding_x = dist_x < (low_wall_x_dim / 2 + collision_threshold)
        is_colliding_y = dist_y < (low_wall_y_dim / 2 + collision_threshold)
        is_colliding_z = dist_z < (low_wall_z_dim / 2 + collision_threshold)

        # A part is colliding if it's within the bounds in all three dimensions
        part_collision_obj3 = is_colliding_x & is_colliding_y & is_colliding_z
        collision_penalty = torch.where(part_collision_obj3, collision_penalty - 1.0, collision_penalty) # Penalize -1.0 per colliding part

    # --- Collision with Object1 (large sphere) ---
    # Large sphere's center position
    obj1_center = object1.data.root_pos_w

    for part_pos in robot_parts_pos:
        # Calculate Euclidean distance from robot part to sphere center
        # Uses relative distance between robot part and object center.
        distance_to_sphere = torch.norm(part_pos - obj1_center, dim=1)

        # Check for collision with Object1 (large sphere)
        # Collision occurs if distance is less than sphere radius plus threshold
        part_collision_obj1 = distance_to_sphere < (large_sphere_radius + collision_threshold)
        collision_penalty = torch.where(part_collision_obj1, collision_penalty - 1.0, collision_penalty) # Penalize -1.0 per colliding part

    # The total reward is the accumulated penalty (will be negative or zero)
    reward = collision_penalty

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def stabilize_after_jump_posture_and_velocity_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stabilize_after_jump_posture_and_velocity_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourage upright posture and minimal horizontal movement.
    Penalizes large deviations in the pelvis's y-position from the center (0.0) and rewards minimal horizontal
    movement (x,y velocity) of the pelvis once it's past the low wall.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_vel = robot.data.body_vel_w[:, pelvis_idx]
    pelvis_vel_x = pelvis_vel[:, 0]
    pelvis_vel_y = pelvis_vel[:, 1]

    # Object dimensions (hardcoded from description for activation condition)
    # Object3: wide low wall, 5m in the y-axis and 0.5m in the z and 0.3m in x axis.
    low_wall_x_dim = 0.3

    # Target values
    target_pelvis_y = 0.0  # Robot should stay centered on the y-axis

    # 1. Pelvis y-position reward: Penalize deviation from center y-axis
    # Uses relative distance to target y-position. Continuous and negative absolute difference.
    pelvis_y_reward = -torch.abs(pelvis_pos_y - target_pelvis_y)

    # 2. Pelvis horizontal velocity reward: Penalize large horizontal velocities once past the low wall
    # Condition: Robot's pelvis is past the low wall (Object3)
    object3 = env.scene['Object3']
    # Calculate the x-coordinate of the far side of the low wall (Object3)
    # Uses relative distance from object's root position and its dimension.
    low_wall_far_x = object3.data.root_pos_w[:, 0] + low_wall_x_dim / 2
    activation_condition = (pelvis_pos_x > low_wall_far_x)

    # Calculate the magnitude of horizontal velocity
    pelvis_horizontal_vel_magnitude = torch.sqrt(pelvis_vel_x**2 + pelvis_vel_y**2)
    # Penalize higher velocities (continuous negative reward)
    velocity_reward = -pelvis_horizontal_vel_magnitude

    # Apply velocity reward only when the activation condition is met
    # Uses torch.where for conditional application of the reward.
    reward = pelvis_y_reward + torch.where(activation_condition, velocity_reward, torch.tensor(0.0, device=env.device))

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
    # Primary reward for achieving the stable, standing position and correct alignment
    StabilizeAfterJumpPrimaryReward = RewTerm(func=stabilize_after_jump_primary_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "stabilize_after_jump_primary_reward"})

    # Shaping reward for avoiding collisions with the low wall and large sphere
    StabilizeAfterJumpCollisionAvoidanceReward = RewTerm(func=stabilize_after_jump_collision_avoidance_reward, weight=0.6,
                                                         params={"normalise": True, "normaliser_name": "stabilize_after_jump_collision_avoidance_reward"})

    # Shaping reward for maintaining upright posture (y-axis) and minimizing horizontal velocity after landing
    StabilizeAfterJumpPostureAndVelocityReward = RewTerm(func=stabilize_after_jump_posture_and_velocity_reward, weight=0.4,
                                                         params={"normalise": True, "normaliser_name": "stabilize_after_jump_posture_and_velocity_reward"})