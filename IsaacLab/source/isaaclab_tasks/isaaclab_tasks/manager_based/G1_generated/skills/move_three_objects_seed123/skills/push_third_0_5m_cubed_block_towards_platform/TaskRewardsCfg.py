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


def main_push_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Primary reward for pushing Object3 (third 0.5m cubed block) towards Object4 (platform),
    penalizing if it overshoots and lands on the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']

    # Calculate the distance vector between Object3 and Object4
    # All rewards MUST ONLY use relative distances between objects and robot parts.
    # Access object positions using: env.scene['ObjectName'].data.root_pos_w
    distance_x_obj3_obj4 = object3.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    distance_y_obj3_obj4 = object3.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    distance_z_obj3_obj4 = object3.data.root_pos_w[:, 2] - object4.data.root_pos_w[:, 2]

    # Euclidean distance
    current_distance_obj3_obj4 = torch.sqrt(distance_x_obj3_obj4**2 + distance_y_obj3_obj4**2 + distance_z_obj3_obj4**2)

    # Initial distance (hypothetical, for scaling). NEVER use hard-coded positions or arbitrary thresholds.
    # A large constant for initial distance can be used if not dynamically calculated.
    # Based on task description, blocks are 2m from robot, platform is 4m from triangle.
    # So initial distance could be around 4m to 6m. Using 5.0 as a reasonable max.
    initial_max_distance = 5.0

    # Reward for reducing distance, scaled to be positive and continuous.
    # Reward normalization is handled at the end of the function.
    reward_distance_reduction = (initial_max_distance - current_distance_obj3_obj4) / initial_max_distance

    # Penalty for pushing Object3 too far (onto the platform).
    # Object3 is a 0.5m cube, so its half-size is 0.25m.
    # Platform z is 0.001m.
    # If Object3's center is less than 0.25m from Object4's center, it's likely on the platform.
    # If Object3's z-center is too low, it's also likely on the platform.
    # Object3's z-center should be around 0.25m (half its height) when on the ground.
    # If it's on the platform (z=0.001m), its z-center would be 0.25m + 0.001m = 0.251m.
    # A small buffer (e.g., 0.05m) is added to account for slight variations.
    # NEVER use previous_object_positions or any non-approved attributes.
    object3_half_height = 0.25 # Hardcoded from object configuration (0.5m cube)
    platform_z_pos = 0.001 # Hardcoded from task description
    z_threshold_on_platform = object3_half_height + platform_z_pos - 0.05 # If z is below this, it's too low

    # Condition for overshooting: distance is too small OR z-position indicates it's on the platform.
    overshoot_penalty_condition = (current_distance_obj3_obj4 < 0.25) | (object3.data.root_pos_w[:, 2] < z_threshold_on_platform)

    # Apply penalty if overshooting
    reward = torch.where(overshoot_penalty_condition, -10.0, reward_distance_reduction)

    # Implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_approach_and_hand_proximity_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_reward") -> torch.Tensor:
    """
    Shaping reward encouraging the robot's pelvis to move closer to Object3 and its hands to be near Object3,
    while maintaining a stable, upright posture. Active during the approach phase.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object3 = env.scene['Object3']
    object4 = env.scene['Object4'] # Needed for activation condition
    robot = env.scene["robot"]

    # Access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Distance from pelvis to Object3 (relative distance)
    current_dist_pelvis_obj3 = torch.norm(object3.data.root_pos_w - pelvis_pos, dim=1)

    # Distance from hands to Object3 (relative distance)
    current_dist_left_hand_obj3 = torch.norm(object3.data.root_pos_w - left_hand_pos, dim=1)
    current_dist_right_hand_obj3 = torch.norm(object3.data.root_pos_w - right_hand_pos, dim=1)

    # Reward for pelvis getting closer to Object3 (negative distance for positive reward)
    reward_pelvis_approach = -current_dist_pelvis_obj3

    # Reward for hands getting closer to Object3 (take the minimum distance of either hand)
    reward_hands_proximity = -torch.min(current_dist_left_hand_obj3, current_dist_right_hand_obj3)

    # Reward for maintaining stable pelvis height. NEVER use hard-coded positions or arbitrary thresholds.
    # Target pelvis z=0.7m is a standard stable height for the robot.
    target_pelvis_z = 0.7
    reward_pelvis_stability = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine approach and hand proximity rewards
    reward_combined = (reward_pelvis_approach * 0.5) + (reward_hands_proximity * 0.5) + (reward_pelvis_stability * 0.2)

    # Activation condition: Active when Object3 is still far from Object4 (initial approach phase).
    # Use the same distance calculation as in primary reward for consistency.
    current_distance_obj3_obj4 = torch.norm(object3.data.root_pos_w - object4.data.root_pos_w, dim=1)

    # If Object3 is more than 1.0m away from Object4, this reward is active.
    # This threshold should be larger than the "overshoot_threshold" in primary reward.
    approach_activation_threshold = 1.0 # meters
    activation_condition = (current_distance_obj3_obj4 > approach_activation_threshold)

    # Apply reward only when active, otherwise 0.0
    reward = torch.where(activation_condition, reward_combined, 0.0)

    # Implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward penalizing collisions or close proximity between the robot's body parts
    and non-target objects (Object1, Object2, Object4).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    # Object3 is the target, so collisions with it are expected/desired for pushing, not penalized here.
    object4 = env.scene['Object4']
    robot = env.scene["robot"]

    # Access robot parts
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Define a small buffer for collision detection. NEVER use hard-coded positions or arbitrary thresholds.
    # This buffer is based on the object size (0.5m cubes) and general robot part size.
    # A 0.3m buffer means if the center of a robot part is within 0.3m of an object's center, it's considered close.
    # This is relative to the object's half-size (0.25m) plus some robot part radius.
    collision_buffer = 0.3
    threshold_sq = collision_buffer**2 # Use squared distance for efficiency

    # Function to calculate squared Euclidean distance
    def squared_distance(pos1, pos2):
        # All rewards MUST ONLY use relative distances between objects and robot parts
        return torch.sum((pos1 - pos2)**2, dim=1)

    reward_collision = torch.zeros_like(pelvis_pos[:, 0]) # Initialize reward tensor

    # Penalize proximity to Object1 (first 0.5m cubed block)
    # All distances are relative.
    dist_sq_pelvis_obj1 = squared_distance(pelvis_pos, object1.data.root_pos_w)
    dist_sq_left_hand_obj1 = squared_distance(left_hand_pos, object1.data.root_pos_w)
    dist_sq_right_hand_obj1 = squared_distance(right_hand_pos, object1.data.root_pos_w)
    dist_sq_left_foot_obj1 = squared_distance(left_foot_pos, object1.data.root_pos_w)
    dist_sq_right_foot_obj1 = squared_distance(right_foot_pos, object1.data.root_pos_w)

    # Penalize proximity to Object2 (second 0.5m cubed block)
    dist_sq_pelvis_obj2 = squared_distance(pelvis_pos, object2.data.root_pos_w)
    dist_sq_left_hand_obj2 = squared_distance(left_hand_pos, object2.data.root_pos_w)
    dist_sq_right_hand_obj2 = squared_distance(right_hand_pos, object2.data.root_pos_w)
    dist_sq_left_foot_obj2 = squared_distance(left_foot_pos, object2.data.root_pos_w)
    dist_sq_right_foot_obj2 = squared_distance(right_foot_pos, object2.data.root_pos_w)

    # Penalize proximity to Object4 (platform) - except for feet on ground.
    # Focus on pelvis and hands for platform collision.
    dist_sq_pelvis_obj4 = squared_distance(pelvis_pos, object4.data.root_pos_w)
    dist_sq_left_hand_obj4 = squared_distance(left_hand_pos, object4.data.root_pos_w)
    dist_sq_right_hand_obj4 = squared_distance(right_hand_pos, object4.data.root_pos_w)

    # Apply penalty if distance is below a threshold (i.e., actual collision or very close).
    # Use inverse squared distance for a continuous, stronger penalty when closer.
    epsilon = 1e-3 # Small constant to avoid division by zero and smooth the curve

    # Penalty for Object1
    reward_collision -= torch.where(dist_sq_pelvis_obj1 < threshold_sq, 1.0 / (dist_sq_pelvis_obj1 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_left_hand_obj1 < threshold_sq, 1.0 / (dist_sq_left_hand_obj1 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_right_hand_obj1 < threshold_sq, 1.0 / (dist_sq_right_hand_obj1 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_left_foot_obj1 < threshold_sq, 1.0 / (dist_sq_left_foot_obj1 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_right_foot_obj1 < threshold_sq, 1.0 / (dist_sq_right_foot_obj1 + epsilon), 0.0)

    # Penalty for Object2
    reward_collision -= torch.where(dist_sq_pelvis_obj2 < threshold_sq, 1.0 / (dist_sq_pelvis_obj2 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_left_hand_obj2 < threshold_sq, 1.0 / (dist_sq_left_hand_obj2 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_right_hand_obj2 < threshold_sq, 1.0 / (dist_sq_right_hand_obj2 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_left_foot_obj2 < threshold_sq, 1.0 / (dist_sq_left_foot_obj2 + epsilon), 0.0)
    reward_collision -= torch.where(dist_sq_right_foot_obj2 < threshold_sq, 1.0 / (dist_sq_right_foot_obj2 + epsilon), 0.0)

    # For platform (Object4), only penalize pelvis and hands if they are too low (colliding with top surface).
    # Feet are expected to be on the platform, so their collision is handled by stability, not penalized here.
    # Platform z is 0.001m. A small buffer (0.05m) above the platform's surface.
    platform_z_pos = 0.001 # Hardcoded from task description
    pelvis_z_too_low = (pelvis_pos[:, 2] < (platform_z_pos + 0.05))
    left_hand_z_too_low = (left_hand_pos[:, 2] < (platform_z_pos + 0.05))
    right_hand_z_too_low = (right_hand_pos[:, 2] < (platform_z_pos + 0.05))

    reward_collision -= torch.where(pelvis_z_too_low & (dist_sq_pelvis_obj4 < threshold_sq), 1.0 / (dist_sq_pelvis_obj4 + epsilon), 0.0)
    reward_collision -= torch.where(left_hand_z_too_low & (dist_sq_left_hand_obj4 < threshold_sq), 1.0 / (dist_sq_left_hand_obj4 + epsilon), 0.0)
    reward_collision -= torch.where(right_hand_z_too_low & (dist_sq_right_hand_obj4 < threshold_sq), 1.0 / (dist_sq_right_hand_obj4 + epsilon), 0.0)

    reward = reward_collision

    # Implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for moving Object3 towards Object4, with overshoot penalty.
    # Weight 1.0 as it's the main objective.
    MainPushBlockReward = RewTerm(func=main_push_block_reward, weight=1.0,
                                  params={"normalise": True, "normaliser_name": "main_push_block_reward"})

    # Shaping reward for robot approach, hand proximity, and pelvis stability.
    # Weight 0.6 to guide the robot during the initial phase.
    RobotApproachAndHandProximityReward = RewTerm(func=robot_approach_and_hand_proximity_reward, weight=0.6,
                                                  params={"normalise": True, "normaliser_name": "robot_approach_reward"})

    # Shaping reward for collision avoidance with non-target objects.
    # Weight 0.4 to encourage safe navigation.
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})