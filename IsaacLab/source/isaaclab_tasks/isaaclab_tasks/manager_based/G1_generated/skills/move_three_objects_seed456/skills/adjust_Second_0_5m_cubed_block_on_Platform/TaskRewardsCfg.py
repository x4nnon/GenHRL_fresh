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


def adjust_Second_0_5m_cubed_block_on_Platform_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'adjust_Second_0_5m_cubed_block_on_Platform' skill.
    This reward encourages the Second 0.5m cubed block (Object2) to be fully and stably positioned
    within the boundaries of the Platform (Object4).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object2 = env.scene['Object2'] # Second 0.5m cubed block
    object4 = env.scene['Object4'] # Platform

    # Hardcode object dimensions from the object configuration (as per rules)
    platform_width_x = 2.0
    platform_depth_y = 2.0
    platform_height_z = 0.001
    block_size = 0.5

    # Calculate the target position for Object2's center relative to Object4's center and top surface.
    # Object4's root_pos_w is its center.
    # The block's center should align with the platform's center in x, y.
    # The block's base should be on the platform's top surface in z.
    # Object2's center in z will be at object4.data.root_pos_w[:, 2] + platform_height_z / 2 + block_size / 2
    target_object2_x = object4.data.root_pos_w[:, 0]
    target_object2_y = object4.data.root_pos_w[:, 1]
    target_object2_z = object4.data.root_pos_w[:, 2] + platform_height_z / 2 + block_size / 2

    # Calculate relative distances for Object2 from its target position on Object4
    # These are continuous rewards based on absolute differences.
    distance_x = object2.data.root_pos_w[:, 0] - target_object2_x
    distance_y = object2.data.root_pos_w[:, 1] - target_object2_y
    distance_z = object2.data.root_pos_w[:, 2] - target_object2_z

    # Reward for being centered on the platform (x, y) and at the correct height (z)
    # The block's center must be within (platform_dim - block_dim)/2 from platform center for it to be fully on.
    max_x_offset = (platform_width_x - block_size) / 2
    max_y_offset = (platform_depth_y - block_size) / 2

    # Reward for x-alignment: negative absolute distance, with a large penalty if outside the x boundary.
    # This ensures the block is within the platform's x-bounds.
    reward_x = -torch.abs(distance_x)
    reward_x = torch.where(torch.abs(distance_x) <= max_x_offset, reward_x, -10.0)

    # Reward for y-alignment: negative absolute distance, with a large penalty if outside the y boundary.
    # This ensures the block is within the platform's y-bounds.
    reward_y = -torch.abs(distance_y)
    reward_y = torch.where(torch.abs(distance_y) <= max_y_offset, reward_y, -10.0)

    # Reward for z-alignment: negative absolute distance from the target height (on top of the platform).
    reward_z = -torch.abs(distance_z)

    # Combine rewards for overall positioning.
    reward = reward_x + reward_y + reward_z

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hand_proximity_to_object2_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot's right hand to be close to Object2, facilitating pushing.
    This reward is active only when Object2 is not yet perfectly positioned on Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot part using approved patterns
    object2 = env.scene['Object2'] # Second 0.5m cubed block
    object4 = env.scene['Object4'] # Platform
    robot = env.scene["robot"]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Hardcode object dimensions from the object configuration
    platform_width_x = 2.0
    platform_depth_y = 2.0
    platform_height_z = 0.001
    block_size = 0.5

    # Calculate target position for Object2 relative to Object4 to determine activation condition
    target_object2_x = object4.data.root_pos_w[:, 0]
    target_object2_y = object4.data.root_pos_w[:, 1]
    target_object2_z = object4.data.root_pos_w[:, 2] + platform_height_z / 2 + block_size / 2

    # Calculate distances for Object2 relative to its target position on Object4
    dist_obj2_target_x = object2.data.root_pos_w[:, 0] - target_object2_x
    dist_obj2_target_y = object2.data.root_pos_w[:, 1] - target_object2_y
    dist_obj2_target_z = object2.data.root_pos_w[:, 2] - target_object2_z

    # Condition: Object2 is not yet perfectly positioned on Object4.
    # This condition checks if Object2 is outside a small tolerance zone around the target.
    tolerance = 0.05 # 5cm tolerance for "perfectly positioned"
    condition_not_on_platform = (torch.abs(dist_obj2_target_x) > tolerance) | \
                               (torch.abs(dist_obj2_target_y) > tolerance) | \
                               (torch.abs(dist_obj2_target_z) > tolerance)

    # Calculate relative distance vector between right hand and Object2's center
    distance_hand_obj2_x = object2.data.root_pos_w[:, 0] - right_hand_pos[:, 0]
    distance_hand_obj2_y = object2.data.root_pos_w[:, 1] - right_hand_pos[:, 1]
    distance_hand_obj2_z = object2.data.root_pos_w[:, 2] - right_hand_pos[:, 2]

    # Reward for hand proximity to Object2: negative sum of absolute differences in x, y, z.
    # This encourages the hand to get close to the block.
    reward = -torch.abs(distance_hand_obj2_x) - torch.abs(distance_hand_obj2_y) - torch.abs(distance_hand_obj2_z)

    # Apply activation condition: reward is 0.0 if Object2 is already on the platform.
    reward = torch.where(condition_not_on_platform, reward, 0.0)

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def stability_and_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_collision_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain a stable, upright posture by keeping its pelvis
    at a desired height and avoiding collisions with Object1, Object3, and Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot part using approved patterns
    object1 = env.scene['Object1'] # First 0.5m cubed block
    object3 = env.scene['Object3'] # Third 0.5m cubed block
    object4 = env.scene['Object4'] # Platform
    robot = env.scene["robot"]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position of pelvis (absolute height, allowed for height-related rewards)

    # Desired pelvis height for stability (hardcoded as per design plan)
    desired_pelvis_z = 0.7

    # Reward for maintaining pelvis height: negative absolute difference from desired height.
    # This encourages the robot to stay upright and stable.
    reward_pelvis_height = -torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Collision avoidance for robot's body parts with other objects
    collision_reward = torch.zeros_like(pelvis_pos_z) # Initialize collision reward tensor
    collision_threshold = 0.1 # Minimum safe distance to objects (hardcoded clearance)
    block_half_size = 0.5 / 2 # Half size of the 0.5m cubed blocks

    # Define critical robot parts for collision avoidance
    robot_parts_for_collision = ['pelvis', 'left_palm_link', 'right_palm_link', 'left_ankle_roll_link', 'right_ankle_roll_link']

    # Hardcode platform dimensions for collision checks
    platform_half_width_x = 2.0 / 2
    platform_half_depth_y = 2.0 / 2
    platform_half_height_z = 0.001 / 2

    for part_name in robot_parts_for_collision:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx]

        # Collision with Object1 (First 0.5m cubed block)
        # Calculate Euclidean distance between robot part and Object1's center.
        # Penalty applied if distance is less than (block_half_size + collision_threshold).
        dist_obj1_vec = object1.data.root_pos_w - part_pos
        dist_obj1_total = torch.norm(dist_obj1_vec, dim=1)
        # Reward is positive when distance is greater than threshold, negative when less.
        # The factor 5.0 scales the penalty/reward.
        collision_reward += torch.where(dist_obj1_total < (block_half_size + collision_threshold),
                                        (dist_obj1_total - (block_half_size + collision_threshold)) * 5.0, 0.0)

        # Collision with Object3 (Third 0.5m cubed block)
        # Similar logic as for Object1.
        dist_obj3_vec = object3.data.root_pos_w - part_pos
        dist_obj3_total = torch.norm(dist_obj3_vec, dim=1)
        collision_reward += torch.where(dist_obj3_total < (block_half_size + collision_threshold),
                                        (dist_obj3_total - (block_half_size + collision_threshold)) * 5.0, 0.0)

        # Collision with Object4 (Platform)
        # For platform, consider horizontal and vertical collision separately for general body parts.
        # Avoid hitting the sides or bottom of the platform.
        dist_obj4_x = object4.data.root_pos_w[:, 0] - part_pos[:, 0]
        dist_obj4_y = object4.data.root_pos_w[:, 1] - part_pos[:, 1]
        dist_obj4_z = object4.data.root_pos_w[:, 2] - part_pos[:, 2]

        # Check if part is horizontally within platform bounds + buffer
        is_within_platform_x = torch.abs(dist_obj4_x) < (platform_half_width_x + collision_threshold)
        is_within_platform_y = torch.abs(dist_obj4_y) < (platform_half_depth_y + collision_threshold)

        # If part is too low (below platform top surface + buffer) and within horizontal bounds, it's a collision.
        # This is for general body parts, not specifically feet for standing.
        collision_with_platform_body = (part_pos[:, 2] < object4.data.root_pos_w[:, 2] + platform_half_height_z + collision_threshold) & \
                                       is_within_platform_x & is_within_platform_y

        # Apply a penalty for general body parts colliding with the platform.
        collision_reward += torch.where(collision_with_platform_body, -5.0, 0.0)

    # Combine pelvis height reward and collision avoidance reward
    reward = reward_pelvis_height + collision_reward

    # Mandatory normalization implementation
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
    Configuration for the reward terms used in the 'adjust_Second_0_5m_cubed_block_on_Platform' skill.
    """
    # Main reward for positioning Object2 on Object4
    main_reward = RewTerm(func=adjust_Second_0_5m_cubed_block_on_Platform_main_reward, weight=1.0,
                          params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for right hand proximity to Object2
    hand_proximity_reward = RewTerm(func=hand_proximity_to_object2_reward, weight=0.4,
                                    params={"normalise": True, "normaliser_name": "hand_proximity_reward"})

    # Shaping reward for robot stability and collision avoidance
    stability_collision_reward = RewTerm(func=stability_and_collision_avoidance_reward, weight=0.2, # Reduced weight as it's a general stability/safety reward
                                         params={"normalise": True, "normaliser_name": "stability_collision_reward"})