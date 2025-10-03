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


def walk_to_object5_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_object5_reward") -> torch.Tensor:
    """
    Primary reward for the robot to walk to Cylinder Column 5 and stop in front of it.
    This reward penalizes the distance between the robot's pelvis and a target position
    in front of Object5, and also penalizes deviation from a desired pelvis height.
    """
    # CRITICAL RULE: MANDATORY IMPORTS and RewNormalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object5 = env.scene['Object5']
    object5_pos = object5.data.root_pos_w # Shape: [num_envs, 3]

    robot = env.scene["robot"]
    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from object config.
    cylinder_radius = 0.3 # From object configuration for Cylinder Column
    target_offset_y = cylinder_radius + 0.3 # 0.3m clearance from cylinder surface, total 0.6m from center

    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    # The target position is relative to Object5's position.
    target_x = object5_pos[:, 0] # Align x with object
    target_y = object5_pos[:, 1] - target_offset_y # Move to -Y direction relative to object for "in front"
    # CRITICAL RULE: Z-position is the only absolute position allowed when height is important.
    target_z = 0.7 # Desired pelvis height for standing, this is an absolute height, which is allowed.

    # Calculate distance components.
    # CRITICAL RULE: Use relative distances.
    distance_x = pelvis_pos[:, 0] - target_x
    distance_y = pelvis_pos[:, 1] - target_y
    distance_z_pelvis = pelvis_pos[:, 2] - target_z

    # CRITICAL RULE: Rewards should be continuous and negative for distance.
    # Add penalty if Object5 has already fallen (z below radius + tolerance)
    cylinder_fallen_z_threshold = 0.3 + 0.1
    column_fallen = (object5_pos[:, 2] <= cylinder_fallen_z_threshold)
    fallen_penalty = torch.where(column_fallen,
                                 torch.tensor(-1.0, device=env.device),
                                 torch.tensor(0.0, device=env.device))

    reward = -torch.abs(distance_x) - torch.abs(distance_y) - torch.abs(distance_z_pelvis) + fallen_penalty

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    This reward penalizes the robot for colliding with or getting too close to any of the cylinder columns
    (Object1, Object2, Object3, Object4, Object5) with its pelvis or feet. It encourages safe navigation.
    """
    # CRITICAL RULE: MANDATORY IMPORTS and RewNormalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CRITICAL RULE: Access objects directly - objects should always exist in the scene.
    objects = [env.scene[f'Object{i}'] for i in range(1, 6)]

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # CRITICAL RULE: THERE IS NO way to access the SIZE of an object. Hardcode from object config.
    cylinder_radius = 0.3 # From object configuration for Cylinder Column
    collision_threshold = cylinder_radius + 0.1 # Add a small buffer for clearance (0.4m total)

    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    def get_horizontal_distance(robot_part_pos, obj_pos):
        dx = robot_part_pos[:, 0] - obj_pos[:, 0]
        dy = robot_part_pos[:, 1] - obj_pos[:, 1]
        return torch.sqrt(dx**2 + dy**2)

    # Initialize collision reward
    reward_collision = torch.zeros_like(pelvis_pos[:, 0])

    # Check collision for each object with pelvis and feet
    for obj in objects:
        obj_pos = obj.data.root_pos_w

        dist_pelvis_obj = get_horizontal_distance(pelvis_pos, obj_pos)
        dist_left_foot_obj = get_horizontal_distance(left_foot_pos, obj_pos)
        dist_right_foot_obj = get_horizontal_distance(right_foot_pos, obj_pos)

        # CRITICAL RULE: Rewards should be continuous. Penalty increases as distance decreases below threshold.
        # The penalty is scaled by 0.5 to make it less dominant than the primary reward.
        reward_collision -= torch.where(dist_pelvis_obj < collision_threshold, 1.0 - (dist_pelvis_obj / collision_threshold), 0.0) * 0.5
        reward_collision -= torch.where(dist_left_foot_obj < collision_threshold, 1.0 - (dist_left_foot_obj / collision_threshold), 0.0) * 0.5
        reward_collision -= torch.where(dist_right_foot_obj < collision_threshold, 1.0 - (dist_right_foot_obj / collision_threshold), 0.0) * 0.5

    reward = reward_collision

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to keep its feet on the ground and avoid lifting them too high,
    which is crucial for stable walking and standing. It also penalizes the pelvis dropping too low,
    indicating a fall.
    """
    # CRITICAL RULE: MANDATORY IMPORTS and RewNormalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    # CRITICAL RULE: Z-position is the only absolute position allowed when height is important.
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is an absolute height, allowed for stability checks.

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Desired foot height (on ground)
    # CRITICAL RULE: Hardcoded thresholds are allowed for stability checks.
    ground_z = 0.05 # Small offset above zero to account for foot thickness

    # CRITICAL RULE: Rewards should be continuous. Penalty increases linearly beyond max_foot_lift_height.
    max_foot_lift_height = 0.3 # Max reasonable height for a foot during walking
    reward_feet_height = torch.where(left_foot_pos_z > max_foot_lift_height, -(left_foot_pos_z - max_foot_lift_height), 0.0)
    reward_feet_height += torch.where(right_foot_pos_z > max_foot_lift_height, -(right_foot_pos_z - max_foot_lift_height), 0.0)

    # CRITICAL RULE: Rewards should be continuous. Penalty increases linearly below min_pelvis_height.
    min_pelvis_height = 0.4 # Minimum acceptable pelvis height before considering it a fall
    reward_pelvis_fall = torch.where(pelvis_pos_z < min_pelvis_height, -(min_pelvis_height - pelvis_pos_z) * 2.0, 0.0) # Stronger penalty

    reward = reward_feet_height + reward_pelvis_fall

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
    Reward terms for the walk_to_cylinderColumn5 skill.
    """
    # Primary reward for reaching the target position in front of Object5
    # CRITICAL RULE: Main reward weight ~1.0
    walk_to_object5_reward = RewTerm(func=walk_to_object5_reward, weight=1.0,
                                     params={"normalise": True, "normaliser_name": "walk_to_object5_reward"})

    # Shaping reward for avoiding collisions with any cylinder column
    # CRITICAL RULE: Supporting reward weights <1.0
    collision_avoidance_reward = RewTerm(func=collision_avoidance_reward, weight=0.6,
                                         params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining stability (feet on ground, pelvis not too low)
    # CRITICAL RULE: Supporting reward weights <1.0
    stability_reward = RewTerm(func=stability_reward, weight=0.4,
                               params={"normalise": True, "normaliser_name": "stability_reward"})