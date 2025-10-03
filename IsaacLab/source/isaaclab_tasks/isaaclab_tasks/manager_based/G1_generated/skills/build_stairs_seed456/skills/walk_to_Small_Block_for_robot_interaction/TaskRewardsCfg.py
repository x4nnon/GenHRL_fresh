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

def walk_to_small_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the robot to walk to a position near the Small Block (Object1) for interaction.
    Encourages the robot's pelvis to be within a specific pushing distance of the Small Block's center
    in the horizontal plane, and to maintain a stable pelvis height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL OBJECT NAMING: Object1 is "Small Block for robot interaction"
    small_block = env.scene['Object1'] # Accessing object using approved pattern
    # Accessing robot object
    robot = env.scene["robot"]

    # Access the required robot part(s)
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Object1 dimensions (hardcoded from object configuration)
    # "Small Block for robot interaction" measures x=1m y=1m and z=0.3m
    small_block_size_x = 1.0
    small_block_size_y = 1.0
    small_block_size_z = 0.3

    # Desired pushing distance from the block's center in the horizontal (x,y) plane.
    # This value is chosen to place the robot's pelvis at a suitable distance for interaction.
    desired_push_dist_xy = 0.75 # meters from block center

    # Calculate horizontal distance components between pelvis and small block
    # Using relative distances as required
    dist_x = small_block.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    dist_y = small_block.data.root_pos_w[:, 1] - pelvis_pos[:, 1]

    # Calculate the Euclidean horizontal distance
    horizontal_distance_xy = torch.sqrt(dist_x**2 + dist_y**2)

    # Reward for approaching the desired horizontal distance
    # This is a continuous negative reward that is maximized (closest to 0) when horizontal_distance_xy
    # is exactly equal to desired_push_dist_xy.
    reward_approach_xy = -torch.abs(horizontal_distance_xy - desired_push_dist_xy)

    # Desired pelvis height for stable standing/interaction
    desired_pelvis_z = 0.7 # meters

    # Reward for maintaining stable pelvis height
    # This is a continuous negative reward that is maximized (closest to 0) when pelvis_pos_z
    # is exactly equal to desired_pelvis_z.
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - desired_pelvis_z)

    # Combine rewards
    # The approach reward is weighted higher as it's the primary goal.
    reward = reward_approach_xy + 0.5 * reward_pelvis_z

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    This reward encourages the robot to avoid collisions with all three blocks (Object1, Object2, Object3).
    It applies a continuous negative reward if any critical robot part (feet, pelvis) gets too close to any block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL OBJECT NAMING: Object1, Object2, Object3 are the blocks
    small_block = env.scene['Object1']
    medium_block = env.scene['Object2']
    large_block = env.scene['Object3']
    # Accessing robot object
    robot = env.scene["robot"]

    # Access the required robot part(s)
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define a minimum safe distance for collision avoidance
    # This value is chosen to provide a buffer around the robot parts.
    safe_distance = 0.2 # meters, slightly larger than robot's foot/shin radius

    # Function to calculate collision reward for a single robot part and object
    # This helper function ensures calculations are batched and use relative distances.
    def calculate_single_collision_penalty(robot_part_pos, obj_pos, safe_dist):
        # Calculate Euclidean distance between robot part and object center
        # Using relative distances for all components (x, y, z)
        dist_x = robot_part_pos[:, 0] - obj_pos[:, 0]
        dist_y = robot_part_pos[:, 1] - obj_pos[:, 1]
        dist_z = robot_part_pos[:, 2] - obj_pos[:, 2]
        distance_xyz = torch.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        
        # Reward is negative and increases sharply as distance falls below safe_dist.
        # Using a negative squared difference for a continuous and increasing penalty.
        # Reward is 0 if distance is greater than or equal to safe_dist.
        collision_penalty = torch.where(distance_xyz < safe_dist, -(safe_dist - distance_xyz)**2, 0.0)
        return collision_penalty

    # Calculate total collision rewards for each relevant robot part with each block
    reward_collision = torch.zeros_like(pelvis_pos[:, 0]) # Initialize with zeros for batching

    # Iterate through all blocks and sum up penalties for each robot part
    for obj in [small_block, medium_block, large_block]:
        obj_pos = obj.data.root_pos_w # Accessing object position using approved pattern
        reward_collision += calculate_single_collision_penalty(left_foot_pos, obj_pos, safe_distance)
        reward_collision += calculate_single_collision_penalty(right_foot_pos, obj_pos, safe_distance)
        reward_collision += calculate_single_collision_penalty(pelvis_pos, obj_pos, safe_distance)

    reward = reward_collision

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain an upright and stable posture by penalizing large
    deviations in the pelvis's horizontal (x and y) velocity. This helps prevent the robot from falling
    or becoming unstable during walking.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Accessing robot object
    robot = env.scene["robot"]

    # Access the required robot part(s)
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_vel = robot.data.body_vel_w[:, pelvis_idx] # Accessing robot part linear velocity using approved pattern

    # Extract horizontal velocity components
    pelvis_vel_x = pelvis_vel[:, 0]
    pelvis_vel_y = pelvis_vel[:, 1]

    # Define a maximum stable horizontal velocity threshold
    # Velocities above this threshold indicate instability or uncontrolled movement.
    max_stable_vel_xy = 0.5 # m/s, threshold for "too fast" horizontal movement

    # Calculate the magnitude of the horizontal pelvis velocity
    pelvis_horizontal_vel_magnitude = torch.sqrt(pelvis_vel_x**2 + pelvis_vel_y**2)

    # Reward is negative, increasing with velocity above a threshold.
    # Using a negative squared difference for a continuous and increasing penalty.
    # Reward is 0 if velocity is below or equal to the threshold.
    reward = torch.where(pelvis_horizontal_vel_magnitude > max_stable_vel_xy,
                         -0.5 * (pelvis_horizontal_vel_magnitude - max_stable_vel_xy)**2,
                         0.0)

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
    # Main reward for approaching the small block and maintaining height
    walk_to_Small_Block_MainReward = RewTerm(func=walk_to_small_block_main_reward, weight=1.0,
                                             params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for avoiding collisions with all blocks
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "collision_reward"})

    # Shaping reward for maintaining pelvis stability
    PelvisStabilityReward = RewTerm(func=pelvis_stability_reward, weight=0.3,
                                    params={"normalise": True, "normaliser_name": "stability_reward"})