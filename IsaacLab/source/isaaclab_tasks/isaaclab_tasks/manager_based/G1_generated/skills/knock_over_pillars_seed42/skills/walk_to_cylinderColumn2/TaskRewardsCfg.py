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


def walk_to_cylinderColumn2_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_cylinderColumn2_main_reward") -> torch.Tensor:
    '''Main reward for walk_to_cylinderColumn2 skill.

    This reward encourages the robot to walk to Cylinder Column 2 and stop in front of it,
    ready to interact. It penalizes the distance of the robot's pelvis from a target
    position 0.5m in front of the cylinder's center.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot and the target object (Object2)
    robot = env.scene["robot"] # CRITICAL: Accessing robot using approved pattern
    object_name = env.scene['Object2'] # CRITICAL: Accessing Object2 directly as per requirements

    # Access the robot's pelvis position
    robot_pelvis_idx = robot.body_names.index('pelvis') # CRITICAL: Accessing robot part index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CRITICAL: Accessing robot part position using approved pattern

    # Access Object2's root position
    object_pos = object_name.data.root_pos_w # CRITICAL: Accessing object position using approved pattern

    # Define target x-position relative to cylinder center (0.5m in front)
    # This is a relative distance calculation, ensuring no hard-coded absolute positions.
    target_x_offset = 0.5
    target_x_pos = object_pos[:, 0] - target_x_offset

    # Calculate relative distances in x and y dimensions
    # CRITICAL: Using relative distances between robot part and object
    distance_x = torch.abs(robot_pelvis_pos[:, 0] - target_x_pos)
    distance_y = torch.abs(robot_pelvis_pos[:, 1] - object_pos[:, 1])

    # Reward is negative of the absolute distances, making it continuous and higher for smaller distances
    reward_x = -distance_x
    reward_y = -distance_y

    # Penalty if column has already fallen (z below radius + tolerance)
    cylinder_fallen_z_threshold = 0.3 + 0.1
    column_fallen = (object_pos[:, 2] <= cylinder_fallen_z_threshold)
    fallen_penalty = torch.where(column_fallen,
                                 torch.tensor(-1.0, device=env.device),
                                 torch.tensor(0.0, device=env.device))

    reward = reward_x + reward_y + fallen_penalty

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining desired pelvis height.

    This reward encourages the robot to maintain a stable and upright posture by keeping its
    pelvis at a desired height (0.7m). This is crucial for general stability and preparing
    for the next skill.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot and its pelvis position
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis') # CRITICAL: Accessing robot part index using approved pattern
    robot_pelvis_pos_z = robot.data.body_pos_w[:, robot_pelvis_idx][:, 2] # CRITICAL: Accessing robot part position using approved pattern

    # Desired pelvis height for stability (hardcoded as per reward design plan)
    desired_pelvis_z = 0.7

    # Reward for maintaining desired pelvis height
    # CRITICAL: Using relative distance (difference from desired height)
    reward = -torch.abs(robot_pelvis_pos_z - desired_pelvis_z)

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for discouraging collisions with Cylinder Column 2.

    This reward discourages collisions between the robot's body parts (pelvis, left foot, right foot)
    and Cylinder Column 2 (Object2). It provides a negative reward if any part of the robot
    gets too close to or penetrates the cylinder.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the target object (Object2)
    object_name = env.scene['Object2'] # CRITICAL: Accessing Object2 directly as per requirements
    object_pos = object_name.data.root_pos_w # CRITICAL: Accessing object position using approved pattern

    # Access the required robot parts
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    robot_left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    robot_left_foot_pos = robot.data.body_pos_w[:, robot_left_foot_idx]

    robot_right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    robot_right_foot_pos = robot.data.body_pos_w[:, robot_right_foot_idx]

    # Object2 dimensions (hardcoded from object configuration)
    cylinder_radius = 0.3 # m
    cylinder_height = 2.0 # m

    # Collision threshold (cylinder radius + small clearance)
    # CRITICAL: Using relative distance based on object dimensions
    collision_threshold = cylinder_radius + 0.05 # 0.05m clearance

    # Calculate horizontal distance from cylinder center for each part
    # CRITICAL: Using relative distances between robot parts and object
    pelvis_dist_xy = torch.sqrt(
        (robot_pelvis_pos[:, 0] - object_pos[:, 0])**2 +
        (robot_pelvis_pos[:, 1] - object_pos[:, 1])**2
    )
    left_foot_dist_xy = torch.sqrt(
        (robot_left_foot_pos[:, 0] - object_pos[:, 0])**2 +
        (robot_left_foot_pos[:, 1] - object_pos[:, 1])**2
    )
    right_foot_dist_xy = torch.sqrt(
        (robot_right_foot_pos[:, 0] - object_pos[:, 0])**2 +
        (robot_right_foot_pos[:, 1] - object_pos[:, 1])**2
    )

    # Check vertical overlap with cylinder for pelvis
    # CRITICAL: Using relative Z-position to check overlap with cylinder height
    pelvis_z_overlap = (robot_pelvis_pos[:, 2] > object_pos[:, 2] - cylinder_height / 2.0) & \
                       (robot_pelvis_pos[:, 2] < object_pos[:, 2] + cylinder_height / 2.0)

    # Check vertical overlap with cylinder for feet (assuming feet are near ground)
    # CRITICAL: Using relative Z-position to check overlap with cylinder height
    foot_z_overlap_lower = (robot_left_foot_pos[:, 2] > object_pos[:, 2] - cylinder_height / 2.0) & \
                           (robot_right_foot_pos[:, 2] > object_pos[:, 2] - cylinder_height / 2.0)
    foot_z_overlap_upper = (robot_left_foot_pos[:, 2] < object_pos[:, 2] + cylinder_height / 2.0) & \
                           (robot_right_foot_pos[:, 2] < object_pos[:, 2] + cylinder_height / 2.0)
    foot_z_overlap = foot_z_overlap_lower & foot_z_overlap_upper


    # Collision condition for each part
    pelvis_collision_condition = (pelvis_dist_xy < collision_threshold) & pelvis_z_overlap
    left_foot_collision_condition = (left_foot_dist_xy < collision_threshold) & foot_z_overlap
    right_foot_collision_condition = (right_foot_dist_xy < collision_threshold) & foot_z_overlap

    # Reward is negative when collision occurs, 0 otherwise. This is a continuous-like penalty.
    reward_pelvis_collision = torch.where(pelvis_collision_condition, -1.0, 0.0)
    reward_left_foot_collision = torch.where(left_foot_collision_condition, -1.0, 0.0)
    reward_right_foot_collision = torch.where(right_foot_collision_condition, -1.0, 0.0)

    reward = reward_pelvis_collision + reward_left_foot_collision + reward_right_foot_collision

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for walking to Cylinder Column 2
    WalkToCylinderColumn2MainReward = RewTerm(func=walk_to_cylinderColumn2_main_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "walk_to_cylinderColumn2_main_reward"})

    # Shaping reward for maintaining pelvis height
    PelvisHeightReward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # Shaping reward for avoiding collisions with Cylinder Column 2
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.5, # Increased weight to strongly discourage collisions
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})