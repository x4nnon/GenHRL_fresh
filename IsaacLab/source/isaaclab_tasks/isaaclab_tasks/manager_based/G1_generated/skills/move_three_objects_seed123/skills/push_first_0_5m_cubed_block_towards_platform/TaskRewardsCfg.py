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

def main_push_block_towards_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for pushing the 'first 0.5m cubed block' (Object1) towards the 'platform' (Object4).
    The goal is to move Object1 significantly closer to Object4, but not yet fully onto it.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    # Object dimensions (hardcoded from object configuration as per requirements)
    # From object config: "Object1": "first 0.5m cubed block" (size 0.5m cubed)
    object1_size = 0.5
    # From object config: "Object4": "platform" (x=2m, y=2m)
    platform_x_dim = 2.0
    platform_y_dim = 2.0

    # Calculate the 2D distance between Object1 and Object4 (relative distance)
    # This uses only relative distances between object root positions.
    distance_x_obj1_obj4 = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    distance_y_obj1_obj4 = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    distance_2d_obj1_obj4 = torch.sqrt(distance_x_obj1_obj4**2 + distance_y_obj1_obj4**2)

    # Define target distance range for Object1 from Object4 (center to center)
    # Initial distance is roughly 4m. A good "significantly closer" but "not on it" could be 1.0m to 2.0m.
    target_min_distance = 1.0
    target_max_distance = 2.0
    target_mid_distance = (target_min_distance + target_max_distance) / 2.0

    # Reward for reducing distance, with a plateau/slight penalty if too close or too far.
    # This creates a continuous reward that peaks at the target_mid_distance.
    reward = 1.0 / (1.0 + torch.abs(distance_2d_obj1_obj4 - target_mid_distance))

    # Add a small bonus for being within the target range to encourage staying in the desired zone.
    in_range_condition = (distance_2d_obj1_obj4 >= target_min_distance) & (distance_2d_obj1_obj4 <= target_max_distance)
    reward = reward + torch.where(in_range_condition, 0.5, 0.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_hand_proximity_and_alignment_to_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    """
    Shaping reward to encourage the robot's hands to be close to Object1 and aligned to push it towards Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    # Access robot body part positions using approved patterns
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object dimensions (hardcoded from object configuration)
    # From object config: "Object1": "first 0.5m cubed block" (size 0.5m cubed)
    object1_size = 0.5

    # Calculate 3D distance from hands to Object1 (relative distances)
    # This uses torch.norm for Euclidean distance, which is a relative distance.
    dist_right_hand_obj1 = torch.norm(object1.data.root_pos_w - right_hand_pos, dim=1)
    dist_left_hand_obj1 = torch.norm(object1.data.root_pos_w - left_hand_pos, dim=1)

    # Reward for hands being close to Object1 (e.g., within 0.3m).
    # This is a continuous reward that increases as hands get closer.
    hand_proximity_reward = 0.3 / (0.3 + torch.min(dist_right_hand_obj1, dist_left_hand_obj1))

    # Calculate alignment: robot's pelvis/hands should be behind Object1 relative to Object4.
    # Assuming push direction is generally along the vector from Object1 to Object4.
    # This uses relative positions to define vectors.
    push_dir_vec = object4.data.root_pos_w[:, :2] - object1.data.root_pos_w[:, :2] # 2D push direction
    push_dir_magnitude = torch.norm(push_dir_vec, dim=1, keepdim=True)
    # Normalize the push direction vector, handling division by zero for stability.
    push_dir_norm = push_dir_vec / (push_dir_magnitude + 1e-6)

    # Vector from Object1 to Pelvis (2D for alignment in the horizontal plane)
    obj1_to_pelvis_vec = pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2]

    # Dot product to check if pelvis is "behind" Object1 relative to push direction.
    # A negative dot product means the pelvis is generally in the opposite direction of the push from Object1.
    alignment_dot_product = torch.sum(obj1_to_pelvis_vec * push_dir_norm, dim=1)

    # Perpendicular distance from pelvis to the line defined by Object1 and Object4.
    # This measures how well the pelvis is aligned with the push axis.
    # Line equation: (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1) = 0
    # Here, (x1, y1) = Object1, (x2, y2) = Object4, (x, y) = Pelvis
    # (x2-x1) is push_dir_vec[:,0], (y2-y1) is push_dir_vec[:,1]
    line_dist_numerator = torch.abs((pelvis_pos[:, 0] - object1.data.root_pos_w[:, 0]) * push_dir_vec[:, 1] - \
                                    (pelvis_pos[:, 1] - object1.data.root_pos_w[:, 1]) * push_dir_vec[:, 0])
    line_dist = line_dist_numerator / (push_dir_magnitude.squeeze(1) + 1e-6) # Squeeze to match dimensions

    # Reward for alignment: more positive if pelvis is behind Object1 (negative dot product)
    # and for being close to the line of push (small perpendicular distance).
    # The `torch.where` makes the reward continuous and only active when behind.
    alignment_reward_behind = torch.where(alignment_dot_product < 0, -alignment_dot_product, torch.tensor(0.0, device=env.device))
    alignment_reward_line = 0.5 / (0.5 + line_dist) # Continuous reward for being close to the line

    alignment_reward = alignment_reward_behind + alignment_reward_line

    # Total shaping reward 1
    reward = hand_proximity_reward + alignment_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_stability_and_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    """
    Shaping reward to encourage robot stability, avoid collisions with other objects,
    and ensure the robot stays near Object1 after the push.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']

    # Access robot body part positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object dimensions (hardcoded from object configuration)
    # From object config: "Object1": "first 0.5m cubed block" (size 0.5m cubed)
    object1_size = 0.5
    # From object config: "Object2": "second 0.5m cubed block" (size 0.5m cubed)
    object2_size = 0.5
    # From object config: "Object3": "third 0.5m cubed block" (size 0.5m cubed)
    object3_size = 0.5
    # From object config: "Object4": "platform" (x=2m, y=2m, z=0.001)
    platform_x_dim = 2.0
    platform_y_dim = 2.0
    platform_z_height = 0.001

    # Pelvis height reward for stability (target 0.7m).
    # This uses an absolute Z-position for the pelvis, which is allowed for height.
    pelvis_z_target = 0.7
    stability_reward = 1.0 / (1.0 + torch.abs(pelvis_pos[:, 2] - pelvis_z_target))

    # Collision avoidance with other blocks (Object2, Object3) and platform (Object4).
    collision_penalty = torch.zeros_like(pelvis_pos[:, 0]) # Initialize penalty tensor

    # Distance to Object2 (relative 3D distance)
    dist_pelvis_obj2 = torch.norm(object2.data.root_pos_w - pelvis_pos, dim=1)
    # Penalty if pelvis is too close to Object2. Threshold is object radius + small clearance.
    collision_penalty += torch.where(dist_pelvis_obj2 < (object2_size / 2.0 + 0.1), -1.0 / (dist_pelvis_obj2 + 1e-6), 0.0)

    # Distance to Object3 (relative 3D distance)
    dist_pelvis_obj3 = torch.norm(object3.data.root_pos_w - pelvis_pos, dim=1)
    # Penalty if pelvis is too close to Object3.
    collision_penalty += torch.where(dist_pelvis_obj3 < (object3_size / 2.0 + 0.1), -1.0 / (dist_pelvis_obj3 + 1e-6), 0.0)

    # Distance to Object4 (platform) - avoid walking onto it prematurely.
    # Check if foot is over the platform in XY and below a certain Z threshold.
    # These are relative distances to the platform's center.
    # Left foot check
    dist_left_foot_obj4_x = left_foot_pos[:, 0] - object4.data.root_pos_w[:, 0]
    dist_left_foot_obj4_y = left_foot_pos[:, 1] - object4.data.root_pos_w[:, 1]
    is_left_foot_over_platform_xy = (torch.abs(dist_left_foot_obj4_x) < (platform_x_dim / 2.0 - 0.05)) & \
                                    (torch.abs(dist_left_foot_obj4_y) < (platform_y_dim / 2.0 - 0.05)) # Small buffer
    # Check if foot is too low (i.e., stepping on or through the platform).
    # This uses absolute Z for foot and platform root Z.
    is_left_foot_too_low_z = (left_foot_pos[:, 2] < (platform_z_height + 0.1)) # 0.1m clearance above platform surface
    collision_penalty += torch.where(is_left_foot_over_platform_xy & is_left_foot_too_low_z, -5.0, 0.0) # Large penalty for stepping on platform

    # Right foot check (same logic as left foot)
    dist_right_foot_obj4_x = right_foot_pos[:, 0] - object4.data.root_pos_w[:, 0]
    dist_right_foot_obj4_y = right_foot_pos[:, 1] - object4.data.root_pos_w[:, 1]
    is_right_foot_over_platform_xy = (torch.abs(dist_right_foot_obj4_x) < (platform_x_dim / 2.0 - 0.05)) & \
                                     (torch.abs(dist_right_foot_obj4_y) < (platform_y_dim / 2.0 - 0.05))
    is_right_foot_too_low_z = (right_foot_pos[:, 2] < (platform_z_height + 0.1))
    collision_penalty += torch.where(is_right_foot_over_platform_xy & is_right_foot_too_low_z, -5.0, 0.0)

    # Robot proximity to Object1 at the end of the skill.
    # Ensure robot stays near Object1 after pushing it, ready for next skill.
    # This is a relative 2D distance.
    dist_pelvis_obj1_2d = torch.norm(object1.data.root_pos_w[:, :2] - pelvis_pos[:, :2], dim=1)

    # Reward for being within a reasonable distance (e.g., 0.5m to 1.0m) from Object1.
    # Target 0.75m from block center. This encourages the robot to stay with the block.
    proximity_to_block_reward = 1.0 / (1.0 + torch.abs(dist_pelvis_obj1_2d - 0.75))

    # Total shaping reward 2
    reward = stability_reward + collision_penalty + proximity_to_block_reward

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
    # Main reward for moving the block towards the platform
    MainPushBlockTowardsPlatformReward = RewTerm(func=main_push_block_towards_platform_reward, weight=1.0,
                                                 params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for robot hand proximity and alignment to the block
    RobotHandProximityAndAlignmentToBlockReward = RewTerm(func=robot_hand_proximity_and_alignment_to_block_reward, weight=0.6,
                                                          params={"normalise": True, "normaliser_name": "shaping_reward_1"})

    # Shaping reward for robot stability and collision avoidance
    RobotStabilityAndCollisionAvoidanceReward = RewTerm(func=robot_stability_and_collision_avoidance_reward, weight=0.3,
                                                        params={"normalise": True, "normaliser_name": "shaping_reward_2"})