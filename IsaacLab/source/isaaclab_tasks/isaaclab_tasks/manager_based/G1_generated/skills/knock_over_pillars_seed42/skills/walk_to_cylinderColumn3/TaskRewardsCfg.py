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


def walk_to_cylinderColumn3_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_cylinderColumn3_main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_cylinderColumn3 skill.
    Encourages the robot's pelvis to reach a target position in front of Cylinder Column 3.
    The target position is 0.2m in front of the cylinder's surface (0.5m from its center) in the x-direction,
    aligned with its y-position, and at a stable standing height (0.7m) in z.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot and the target object (Object3) using approved patterns
    robot = env.scene["robot"]
    object3 = env.scene['Object3'] # Object3 is Cylinder Column 3

    # Access robot's pelvis position using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Access Object3's position using approved patterns
    object3_pos = object3.data.root_pos_w # Shape: [num_envs, 3]

    # Define cylinder dimensions (hardcoded from object configuration: radius 0.3m)
    # This adheres to the rule of hardcoding dimensions from the object config.
    cylinder_radius = 0.3
    # Desired distance from the cylinder's surface (e.g., 0.2m clearance)
    target_clearance_from_surface = 0.2
    # Calculate target distance from the cylinder's center
    target_distance_from_cylinder_center = cylinder_radius + target_clearance_from_surface # 0.3 + 0.2 = 0.5m

    # Define the desired stable pelvis height (hardcoded as per reward plan)
    target_pelvis_z = 0.7

    # Calculate the target x-position relative to Object3.
    # Assuming the robot approaches from the negative x-direction relative to the cylinder.
    # This uses relative positioning to the object's root position.
    target_x_pos = object3_pos[:, 0] - target_distance_from_cylinder_center
    # The target y-position is aligned with the cylinder's y-position.
    target_y_pos = object3_pos[:, 1]

    # Calculate the absolute differences for each component
    # This ensures the reward is continuous and penalizes deviation in any direction.
    # Using torch.abs for continuous, relative distance penalties.
    distance_x = torch.abs(pelvis_pos[:, 0] - target_x_pos)
    distance_y = torch.abs(pelvis_pos[:, 1] - target_y_pos)
    distance_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Penalty if column has already fallen (z below radius + tolerance)
    cylinder_fallen_z_threshold = 0.3 + 0.1
    column_fallen = (object3_pos[:, 2] <= cylinder_fallen_z_threshold)
    fallen_penalty = torch.where(column_fallen,
                                 torch.tensor(-1.0, device=env.device),
                                 torch.tensor(0.0, device=env.device))

    # The reward is the negative sum of these absolute distances plus fallen penalty.
    reward = -distance_x - distance_y - distance_z + fallen_penalty

    # Normalization implementation (mandatory)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_shaping_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain an upright posture.
    Penalizes deviation of the robot's pelvis z-position from a stable standing height (0.7m).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot's pelvis position using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2] # Z-component of pelvis position

    # Desired stable pelvis height (hardcoded as per reward plan)
    target_pelvis_z = 0.7

    # Reward is negative absolute difference from the target height.
    # This continuously penalizes the robot for being too high or too low, making it continuous.
    reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Normalization implementation (mandatory)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_shaping_reward") -> torch.Tensor:
    """
    Shaping reward to penalize robot body parts getting too close to any of the cylinder columns.
    Encourages collision avoidance by applying a penalty when the distance between a robot part
    and a cylinder's center is below a safe threshold (cylinder radius + 0.2m clearance).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access all cylinder objects using approved patterns (Object1 to Object5)
    all_cylinders = [
        env.scene['Object1'],
        env.scene['Object2'],
        env.scene['Object3'],
        env.scene['Object4'],
        env.scene['Object5']
    ]

    # Access required robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_palm_idx = robot.body_names.index('left_palm_link')
    right_palm_idx = robot.body_names.index('right_palm_link')
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')

    # Collect positions of all tracked robot parts into a single tensor for batched operations
    # Unsqueeze(1) adds a dimension for concatenation, resulting in [num_envs, num_parts, 3]
    robot_parts_pos = torch.cat([
        robot.data.body_pos_w[:, pelvis_idx].unsqueeze(1),
        robot.data.body_pos_w[:, left_palm_idx].unsqueeze(1),
        robot.data.body_pos_w[:, right_palm_idx].unsqueeze(1),
        robot.data.body_pos_w[:, left_ankle_idx].unsqueeze(1),
        robot.data.body_pos_w[:, right_ankle_idx].unsqueeze(1)
    ], dim=1)

    # Define cylinder radius (hardcoded from object configuration, as per rules)
    cylinder_radius = 0.3
    # Define safe distance from cylinder center (radius + clearance)
    safe_distance = cylinder_radius + 0.2 # 0.5m from center, 0.2m clearance from surface

    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    for cylinder in all_cylinders:
        cylinder_pos = cylinder.data.root_pos_w # Shape: [num_envs, 3]

        # Calculate Euclidean distance from each robot part to the current cylinder's center
        # cylinder_pos.unsqueeze(1) reshapes to [num_envs, 1, 3] for broadcasting
        # Using torch.norm for 3D distance, which is an approved pattern.
        distances = torch.norm(robot_parts_pos - cylinder_pos.unsqueeze(1), dim=2) # Shape: [num_envs, num_parts]

        # Find the minimum distance from any tracked robot part to the current cylinder
        min_distance_to_cylinder = torch.min(distances, dim=1).values # Shape: [num_envs]

        # Apply a linear penalty if the minimum distance is less than the safe_distance.
        # The penalty increases as the distance decreases from safe_distance to 0.
        # torch.clamp ensures penalty_factor is non-negative, making the reward continuous.
        penalty_factor = torch.clamp(safe_distance - min_distance_to_cylinder, min=0.0)
        # Scale the penalty to make it more impactful
        collision_penalty += -penalty_factor * 2.0

    reward = collision_penalty

    # Normalization implementation (mandatory)
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
    Configuration for the rewards used in the walk_to_cylinderColumn3 skill.
    """
    # Main reward for reaching the target position in front of Cylinder Column 3
    walk_to_cylinderColumn3_main_reward = RewTerm(
        func=walk_to_cylinderColumn3_main_reward,
        weight=1.0, # Primary reward, typically weighted 1.0
        params={"normalise": True, "normaliser_name": "walk_to_cylinderColumn3_main_reward"}
    )

    # Shaping reward for maintaining a stable upright posture
    pelvis_height_shaping_reward = RewTerm(
        func=pelvis_height_shaping_reward,
        weight=0.4, # Shaping reward, typically weighted less than the main reward
        params={"normalise": True, "normaliser_name": "pelvis_height_shaping_reward"}
    )

    # Shaping reward for avoiding collisions with any of the cylinder columns
    collision_avoidance_shaping_reward = RewTerm(
        func=collision_avoidance_shaping_reward,
        weight=0.5, # Shaping reward, weighted to encourage avoidance but not overly restrict movement
        params={"normalise": True, "normaliser_name": "collision_avoidance_shaping_reward"}
    )