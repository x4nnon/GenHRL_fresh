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


def walk_to_small_block_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_small_block_primary_reward") -> torch.Tensor:
    """
    Primary reward for guiding the robot to walk towards and position itself correctly
    in front of the Small Block (Object1) for the subsequent climb.
    It rewards reducing the horizontal distance (x and y) to a target position relative to Object1
    and then maintaining a stable standing height (z) for the pelvis.
    The target x-position is slightly in front of Object1, and the target y-position is aligned with Object1's center.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # Object1: Small Block (x=1m, y=1m, z=0.3m)
    object1 = env.scene['Object1'] # Accessing object using approved pattern
    object1_pos = object1.data.root_pos_w # Accessing object position using approved pattern

    # Access the required robot part(s)
    robot = env.scene["robot"] # Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Object1 dimensions (hardcoded from object configuration as per rules)
    object1_x_dim = 1.0 # Small Block x dimension
    object1_y_dim = 1.0 # Small Block y dimension
    # object1_z_dim = 0.3 # Small Block z dimension (not directly used for positioning reward)

    # Target position relative to Object1 for the robot's pelvis
    # Target x: slightly in front of Object1.
    # Buffer of 0.25m for the robot to stand in front of the 1m wide block.
    # So, target_x_offset = (Object1_x_dim / 2) + buffer = 0.5 + 0.25 = 0.75m
    # The robot should be at Object1's x-coordinate minus this offset.
    target_x_pos = object1_pos[:, 0] - 0.75 # Relative distance calculation
    target_y_pos = object1_pos[:, 1] # Align with Object1's y-center, relative distance calculation
    target_pelvis_z = 0.7 # Stable standing height (absolute z, allowed for height)

    # Calculate distances to target position
    distance_x = pelvis_pos[:, 0] - target_x_pos # Relative distance
    distance_y = pelvis_pos[:, 1] - target_y_pos # Relative distance
    distance_z = pelvis_pos[:, 2] - target_pelvis_z # Relative distance to target Z

    # Reward for approaching the target x,y position
    # Use negative Euclidean distance for continuous positive reward (smaller distance = higher reward)
    reward_xy = -torch.sqrt(torch.square(distance_x) + torch.square(distance_y)) # Continuous reward

    # Reward for maintaining target pelvis z height
    # Penalize deviation from target_pelvis_z
    reward_z = -torch.abs(distance_z) # Continuous reward

    # Combine rewards
    reward = reward_xy + reward_z

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes the robot for collisions or close proximity
    between its key body parts and any of the blocks (Object1, Object2, Object3).
    Encourages the robot to maintain a safe distance from all objects.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')

    # Get positions of key robot parts
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Stack robot part positions for easier iteration (works with batched environments)
    robot_parts_pos = torch.stack([pelvis_pos, left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos], dim=1)

    # Stack object positions
    objects_pos = torch.stack([object1.data.root_pos_w, object2.data.root_pos_w, object3.data.root_pos_w], dim=1)

    # Define a collision threshold. This is a general distance from the object center
    # where penalty starts. It accounts for object size and robot part size.
    # For 1m x 1m x 0.3-0.9m blocks, a threshold of 0.7-1.0m from center is reasonable
    # to start penalizing for proximity, considering robot's own dimensions.
    collision_threshold = 0.8 # meters, tuned to be slightly larger than half block diagonal + robot part radius

    # Initialize reward to zero for all environments
    reward = torch.zeros_like(robot.data.root_pos_w[:, 0])

    # Iterate through robot parts and objects to calculate pairwise distances
    for i in range(robot_parts_pos.shape[1]): # Iterate through robot parts
        for j in range(objects_pos.shape[1]): # Iterate through objects
            # Calculate Euclidean distance between robot part and object center
            dist_to_object_center = torch.norm(robot_parts_pos[:, i] - objects_pos[:, j], dim=1) # Relative distance

            # Apply a negative reward if distance is less than the collision_threshold
            # The penalty is continuous and increases as the distance decreases below the threshold.
            penalty = torch.where(dist_to_object_center < collision_threshold,
                                  - (collision_threshold - dist_to_object_center) * 5.0, # Scale penalty for stronger effect
                                  0.0)
            reward += penalty # Accumulate penalties

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def stability_and_ground_contact_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_and_ground_contact_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain a stable standing posture
    by keeping its feet on the ground and its pelvis at a reasonable height.
    It penalizes feet being too high off the ground and rewards the pelvis being near the target standing height.
    This also implicitly helps with stability for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    # Get z-positions of feet and pelvis
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2] # Z-position (absolute, allowed for height)
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2] # Z-position (absolute, allowed for height)
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2] # Z-position (absolute, allowed for height)

    # Target ground height for feet (assuming ground is z=0).
    # A small offset accounts for foot thickness/model, ensuring feet are "on" the ground.
    target_foot_z = 0.05 # meters

    # Target pelvis height for stable standing
    target_pelvis_z = 0.7 # meters

    # Reward for feet being close to the ground
    # Penalize if feet are too high off the ground. Use negative absolute difference for continuous penalty.
    foot_height_penalty_left = -torch.abs(left_foot_pos_z - target_foot_z) # Continuous reward
    foot_height_penalty_right = -torch.abs(right_foot_pos_z - target_foot_z) # Continuous reward

    # Reward for pelvis being at the target standing height
    # Penalize deviation from target_pelvis_z.
    pelvis_height_reward = -torch.abs(pelvis_pos_z - target_pelvis_z) # Continuous reward

    # Combine rewards
    reward = foot_height_penalty_left + foot_height_penalty_right + pelvis_height_reward

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
    # Primary reward for positioning the robot in front of the Small Block
    walk_to_Small_Block_Primary_Reward = RewTerm(func=walk_to_small_block_primary_reward, weight=1.0,
                                                 params={"normalise": True, "normaliser_name": "walk_to_small_block_primary_reward"})

    # Shaping reward for avoiding collisions with all blocks
    Collision_Avoidance_Reward = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                         params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining stability and ground contact
    Stability_and_Ground_Contact_Reward = RewTerm(func=stability_and_ground_contact_reward, weight=0.3,
                                                  params={"normalise": True, "normaliser_name": "stability_and_ground_contact_reward"})