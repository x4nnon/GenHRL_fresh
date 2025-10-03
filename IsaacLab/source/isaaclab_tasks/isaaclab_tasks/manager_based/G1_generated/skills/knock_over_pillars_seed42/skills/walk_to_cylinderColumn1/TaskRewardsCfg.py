from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats # this automatically sets up the RewNormalizer instance.
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv # Corrected: Added ManagerBasedRLEnv import
import torch

from isaaclab.envs import mdp
# Import custom MDP functions from genhrl
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums


def walk_to_cylinderColumn1_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_cylinderColumn1_main_reward") -> torch.Tensor:
    '''Main reward for walk_to_cylinderColumn1.

    This reward encourages the robot to walk to Cylinder Column 1 and stop in front of it,
    ready to interact. It penalizes the absolute distance between the robot's pelvis
    and a target position relative to Object1 (Cylinder Column 1).
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object1 = env.scene['Object1'] # Accessing Object1 (Cylinder Column 1) using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Check if column has fallen over
    cylinder_fallen_z_threshold = 0.3 + 0.1  # Radius + tolerance = 0.4m
    column_fallen = (object1.data.root_pos_w[:, 2] <= cylinder_fallen_z_threshold)
    
    # Apply large penalty if column has fallen
    fallen_penalty = torch.where(
        column_fallen,
        torch.tensor(-1.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )

    # Object1 dimensions (from description: radius 0.3m) - Hardcoding object dimension from configuration, as per rules
    cylinder_radius = 0.3
    target_distance_x = cylinder_radius + 0.2 # Target 0.5m in front of cylinder center, as per reward plan

    # Calculate the distance vector between the object and the robot part using relative distances
    # The robot should be at object1.x - target_distance_x (assuming approach from positive X)
    # and aligned with object1.y.
    distance_x = (object1.data.root_pos_w[:, 0] - target_distance_x) - pelvis_pos[:, 0] # Relative distance in x-direction
    distance_y = object1.data.root_pos_w[:, 1] - pelvis_pos[:, 1] # Relative distance in y-direction

    # Reward for reducing distance to target x and y. Using negative absolute distance for continuous reward.
    reward = -torch.abs(distance_x) - torch.abs(distance_y) + fallen_penalty # Continuous reward based on relative distances

    # Normalization implementation - Mandatory for all reward functions
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining desired pelvis height.

    This reward encourages the robot to maintain an upright and stable posture by keeping its
    pelvis at a desired height (e.g., 0.7m) and penalizing large deviations.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part using approved patterns
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2] # Accessing robot pelvis z-position

    # Desired pelvis height - Hardcoded desired height, as per reward plan
    desired_pelvis_z = 0.7

    # Reward for maintaining desired pelvis height. Penalizes deviation from desired height.
    # Using negative absolute distance for continuous reward.
    reward = -torch.abs(pelvis_pos_z - desired_pelvis_z) # Continuous reward based on absolute deviation

    # Normalization implementation - Mandatory for all reward functions
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with Object1.

    This reward encourages collision avoidance between the robot's body parts and Object1.
    It applies a negative reward if any part of the robot gets too close to or penetrates Object1,
    preventing the robot from running into or overshooting the cylinder.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required object and robot using approved patterns
    object1 = env.scene['Object1'] # Accessing Object1 (Cylinder Column 1) using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern

    # Object1 dimensions (from description: radius 0.3m, height 2m) - Hardcoding object dimensions from configuration
    cylinder_radius = 0.3
    cylinder_height = 2.0

    # Access relevant robot parts for collision avoidance, as per reward plan
    robot_parts_to_check = ['pelvis', 'left_knee_link', 'right_knee_link', 'left_ankle_roll_link', 'right_ankle_roll_link']
    collision_reward = torch.zeros(env.num_envs, device=env.device) # Initialize reward tensor for batch processing

    for part_name in robot_parts_to_check:
        part_idx = robot.body_names.index(part_name) # Accessing robot part index using approved pattern
        part_pos = robot.data.body_pos_w[:, part_idx] # Accessing robot part position using approved pattern

        # Calculate horizontal distance to cylinder center using relative distances
        dist_xy = torch.sqrt(
            (part_pos[:, 0] - object1.data.root_pos_w[:, 0])**2 + # Relative distance in x
            (part_pos[:, 1] - object1.data.root_pos_w[:, 1])**2   # Relative distance in y
        )

        # Calculate vertical distance to cylinder's base and top using relative positions
        # Assuming cylinder base is at object1.z - cylinder_height/2 and top at object1.z + cylinder_height/2
        object_base_z = object1.data.root_pos_w[:, 2] - (cylinder_height / 2.0)
        object_top_z = object1.data.root_pos_w[:, 2] + (cylinder_height / 2.0)

        # Condition for horizontal proximity (within cylinder radius + small buffer) - Continuous condition
        horizontal_proximity_condition = (dist_xy < (cylinder_radius + 0.1)) # 0.1m buffer

        # Condition for vertical overlap (part_pos_z is between cylinder base and top) - Continuous condition
        vertical_overlap_condition = (part_pos[:, 2] > object_base_z) & (part_pos[:, 2] < object_top_z)

        # Combined collision condition
        collision_condition = horizontal_proximity_condition & vertical_overlap_condition

        # Apply negative reward if collision condition is met
        # The closer the part is to the center of the cylinder, the higher the penalty
        # Use a continuous exponential penalty that increases as distance decreases below threshold
        penalty = torch.where(
            collision_condition,
            -torch.exp(-10.0 * (dist_xy - cylinder_radius)), # Exponential penalty for penetration, continuous
            torch.tensor(0.0, device=env.device)
        )
        collision_reward += penalty # Summing penalties for all checked parts

    reward = collision_reward # Final reward for collision avoidance

    # Normalization implementation - Mandatory for all reward functions
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Main reward for walking to Cylinder Column 1 with weight 1.0
    WalkToCylinderColumn1MainReward = RewTerm(func=walk_to_cylinderColumn1_main_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "walk_to_cylinderColumn1_main_reward"})

    # Shaping reward for maintaining desired pelvis height with weight 0.4
    PelvisHeightReward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # Shaping reward for collision avoidance with Cylinder Column 1 with weight 0.2 (lower as it's a penalty)
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.2,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})