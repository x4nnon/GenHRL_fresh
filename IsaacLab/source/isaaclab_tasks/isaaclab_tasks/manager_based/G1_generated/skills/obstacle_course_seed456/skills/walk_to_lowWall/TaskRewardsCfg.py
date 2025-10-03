from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch

from isaaclab.envs import mdp
from isaaclab.utils.math import quat_apply
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

def walk_to_lowWall_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_lowWall_primary_reward") -> torch.Tensor:
    '''
    Primary reward for the walk_to_lowWall skill.
    Encourages the robot to walk until its pelvis is within a close proximity (e.g., 0.5m) to the low wall, facing it.
    The goal state is the robot standing stably in front of the low wall, ready to initiate a jump.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    low_wall = env.scene['Object3'] # Object3 is the low wall for robot to jump over

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    low_wall_pos = low_wall.data.root_pos_w
    low_wall_pos_x = low_wall_pos[:, 0]
    low_wall_pos_y = low_wall_pos[:, 1]

    # Hardcoded low wall dimensions from the task description (x-axis dimension 0.3m, y-axis dimension 5.0m)
    low_wall_x_dim = 0.3
    low_wall_y_dim = 5.0

    # Calculate target x-position relative to the low wall's front face.
    # The robot should be in front of the wall, e.g., 0.5m away from its front face.
    # Wall's root_pos_w is its center. Front face is at low_wall_pos_x + low_wall_x_dim / 2
    # Target pelvis x-position should be (low_wall_pos_x + low_wall_x_dim / 2) - 0.5
    target_x_pos = low_wall_pos_x + (low_wall_x_dim / 2) - 0.5

    # Reward for approaching the target x-position.
    # Use a negative absolute difference to reward closer proximity. This is a continuous reward.
    reward_approach_x = -torch.abs(pelvis_pos_x - target_x_pos)

    # Penalty for overshooting the wall (pelvis goes past the wall's front face).
    # The wall's front face is at low_wall_pos_x + low_wall_x_dim / 2.
    # This is a continuous penalty that activates when the robot passes the wall.
    overshoot_penalty = torch.where(pelvis_pos_x > (low_wall_pos_x + low_wall_x_dim / 2),
                                    -10.0 * (pelvis_pos_x - (low_wall_pos_x + low_wall_x_dim / 2)),
                                    0.0)

    # Reward for being within the y-bounds of the wall.
    # The robot should be aligned with the wall along the y-axis.
    # The wall extends from low_wall_pos_y - low_wall_y_dim / 2 to low_wall_pos_y + low_wall_y_dim / 2.
    # We want the pelvis to be within this range.
    # Use a smooth penalty that increases as the robot moves further away from the wall's y-center.
    # Reward is higher when y_distance_to_wall_center is small, up to half the wall's y-dimension.
    # Beyond half the wall's y-dimension, it becomes a penalty.
    y_distance_to_wall_center = torch.abs(pelvis_pos_y - low_wall_pos_y)
    reward_y_alignment = torch.exp(-2.0 * (y_distance_to_wall_center - (low_wall_y_dim / 2)).relu())

    # Reward for facing the wall.
    # Get robot's forward direction (x-axis of its base frame in world coordinates).
    # Apply the root orientation to the local x-axis unit vector to obtain forward direction in world frame.
    num_envs = robot.data.root_quat_w.shape[0]
    x_axis_b = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=robot.data.root_quat_w.dtype).repeat(num_envs, 1)
    robot_forward_vec = quat_apply(robot.data.root_quat_w, x_axis_b)
    # The wall is generally in the positive X direction from the robot's starting point.
    # So, the robot's forward vector's X component should be positive (facing towards the wall).
    # And the robot's forward vector's Y component should be close to zero (not facing sideways).
    reward_facing_x = robot_forward_vec[:, 0] # Max 1.0 when facing positive X.
    reward_facing_y = -torch.abs(robot_forward_vec[:, 1]) # Max 0.0 when not facing Y.

    # Combine rewards.
    # The primary reward is for x-position and overshooting.
    # Add y-alignment and facing rewards as shaping components.
    reward = reward_approach_x + overshoot_penalty + reward_y_alignment + reward_facing_x + reward_facing_y

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    '''
    Shaping reward 1: Encourages the robot to maintain an upright and stable posture by keeping its pelvis at a desired z-height.
    This helps prevent the robot from falling or crouching excessively.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Desired pelvis z-height for standing stably. This value is a common stable height for humanoid robots.
    desired_pelvis_z = 0.7

    # Reward for maintaining desired pelvis z-height.
    # Negative absolute difference to reward being close to 0.7m. This is a continuous reward.
    reward = -torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_low_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_low_wall_reward") -> torch.Tensor:
    '''
    Shaping reward 2: Encourages collision avoidance between the robot's body parts (pelvis, feet) and the low wall (Object3).
    It provides a negative reward that increases sharply as the robot gets too close to or collides with the wall.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    low_wall = env.scene['Object3'] # Object3 is the low wall for robot to jump over
    robot = env.scene["robot"]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcoded low wall dimensions from the task description (x-axis 0.3m, y-axis 5.0m, z-axis 0.5m)
    low_wall_x_dim = 0.3
    low_wall_y_dim = 5.0
    low_wall_z_dim = 0.5

    # Wall's center position
    wall_center_x = low_wall.data.root_pos_w[:, 0]
    wall_center_y = low_wall.data.root_pos_w[:, 1]
    wall_center_z = low_wall.data.root_pos_w[:, 2]

    # Define a small buffer/radius for robot parts to avoid collision.
    # This is an approximate radius for collision detection, ensuring continuous penalty.
    robot_body_buffer = 0.15 # meters, approximate size of robot's "skin"

    # Initialize total collision penalty
    total_collision_penalty = torch.zeros_like(pelvis_pos[:, 0])

    # List of robot parts to check for collision
    robot_parts = [pelvis_pos, left_foot_pos, right_foot_pos]

    for part_pos in robot_parts:
        part_x = part_pos[:, 0]
        part_y = part_pos[:, 1]
        part_z = part_pos[:, 2]

        # Calculate relative distances to wall surfaces.
        # Distance to front face (positive if part is in front of wall's front face).
        dist_x_front = part_x - (wall_center_x + low_wall_x_dim / 2)
        # Distance to back face (positive if part is behind wall's back face).
        dist_x_back = part_x - (wall_center_x - low_wall_x_dim / 2)
        # Distance to side faces (absolute distance from wall center, minus half wall width).
        dist_y_side = torch.abs(part_y - wall_center_y) - (low_wall_y_dim / 2)
        # Distance to top face (positive if part is above wall's top face).
        dist_z_top = part_z - (wall_center_z + low_wall_z_dim / 2)

        # Collision penalty for x-axis (robot entering wall from front or back).
        # Penalize if part is within the wall's x-extent (or slightly beyond by buffer)
        # and within wall's y-extent and below wall's top.
        # Use a continuous penalty that increases as penetration increases.
        collision_x_condition = (dist_x_front < robot_body_buffer) & (dist_x_back > -robot_body_buffer) & \
                                (dist_y_side < robot_body_buffer) & (dist_z_top < robot_body_buffer)
        # Calculate penetration depth for x-axis.
        # The penetration is the amount by which the part has crossed the wall boundary, considering the buffer.
        # We take the minimum of penetration from front and back to get the closest penetration.
        penetration_x = robot_body_buffer - torch.min(dist_x_front.abs(), dist_x_back.abs())
        collision_x_penalty = torch.where(collision_x_condition, -20.0 * penetration_x.relu(), 0.0)
        total_collision_penalty += collision_x_penalty

        # Collision penalty for y-axis (robot entering wall from sides).
        collision_y_condition = (dist_y_side < robot_body_buffer) & \
                                (dist_x_front < robot_body_buffer) & (dist_x_back > -robot_body_buffer) & \
                                (dist_z_top < robot_body_buffer)
        penetration_y = robot_body_buffer - dist_y_side
        collision_y_penalty = torch.where(collision_y_condition, -20.0 * penetration_y.relu(), 0.0)
        total_collision_penalty += collision_y_penalty

        # Collision penalty for z-axis (robot going through top of wall).
        collision_z_condition = (dist_z_top < robot_body_buffer) & \
                                (dist_x_front < robot_body_buffer) & (dist_x_back > -robot_body_buffer) & \
                                (dist_y_side < robot_body_buffer)
        penetration_z = robot_body_buffer - dist_z_top
        collision_z_penalty = torch.where(collision_z_condition, -20.0 * penetration_z.relu(), 0.0)
        total_collision_penalty += collision_z_penalty

    reward = total_collision_penalty

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for walking to the low wall and facing it
    walk_to_lowWall_primary_reward = RewTerm(func=walk_to_lowWall_primary_reward, weight=1.0,
                                             params={"normalise": True, "normaliser_name": "walk_to_lowWall_primary_reward"})

    # Shaping reward for maintaining desired pelvis height
    pelvis_height_reward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                   params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # Shaping reward for avoiding collisions with the low wall
    collision_avoidance_low_wall_reward = RewTerm(func=collision_avoidance_low_wall_reward, weight=0.05, # Reduced weight as it can be very strong
                                                  params={"normalise": True, "normaliser_name": "collision_avoidance_low_wall_reward"})