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


def main_position_small_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'Position_Small_Block_for_robot_interaction' skill.
    This reward guides the robot to first approach the Small Block (Object1),
    then push it into a designated target position relative to the Medium Block (Object2),
    while maintaining a suitable pushing posture.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (MANDATORY: using approved pattern)
    object1 = env.scene['Object1']  # Small Block for robot interaction
    object2 = env.scene['Object2']  # Medium Block for robot interaction

    # Access required robot part(s) (MANDATORY: using approved pattern)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Object dimensions (MANDATORY: hardcoded from object config, as RigidObjectData has no size attribute)
    object1_x_dim = 1.0
    object1_y_dim = 1.0
    object1_z_dim = 0.3

    # Object2 dimensions (MANDATORY: hardcoded from object config)
    object2_x_dim = 1.0
    object2_y_dim = 1.0
    object2_z_dim = 0.6

    # Define target position for Object1 relative to Object2 (MANDATORY: relative distances)
    # Object1 is pushed to be adjacent to Object2, forming a step.
    # Assuming Object1 is pushed such that its center x aligns with Object2's center x,
    # and its y is offset to be adjacent to Object2 with a small gap.
    # The Z position is half of Object1's height, placing it on the ground.
    target_object1_x = object2.data.root_pos_w[:, 0]
    target_object1_y = object2.data.root_pos_w[:, 1] - (object1_y_dim / 2.0 + object2_y_dim / 2.0 + 0.1) # 0.1m gap
    target_object1_z = object1_z_dim / 2.0 # On the ground, centered vertically

    # Phase 1: Robot approaches Object1
    # Encourage robot to get close to Object1 in XY plane.
    dist_pelvis_to_object1_xy = torch.norm(pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2], dim=1)
    reward_approach = -dist_pelvis_to_object1_xy * 0.5 # (MANDATORY: continuous reward)

    # Phase 2 & 3: Object1 moves towards target and robot stays close for pushing
    # Distance of Object1 to its target position (MANDATORY: relative distances)
    dist_object1_to_target_x = torch.abs(object1.data.root_pos_w[:, 0] - target_object1_x)
    dist_object1_to_target_y = torch.abs(object1.data.root_pos_w[:, 1] - target_object1_y)
    dist_object1_to_target_z = torch.abs(object1.data.root_pos_w[:, 2] - target_object1_z)
    dist_object1_to_target_xyz = torch.sqrt(dist_object1_to_target_x**2 + dist_object1_to_target_y**2 + dist_object1_to_target_z**2)

    # Reward for Object1 reaching its target (MANDATORY: continuous reward)
    reward_object1_target = -dist_object1_to_target_xyz * 1.0

    # Reward for pelvis staying close to Object1's back face for pushing (MANDATORY: relative distances)
    # Assuming push is generally in the positive X direction relative to Object1's initial position.
    # Robot should be slightly behind Object1 in X, and aligned in Y. Pelvis Z at standing height.
    target_pelvis_x_for_push = object1.data.root_pos_w[:, 0] - (object1_x_dim / 2.0 + 0.2) # 0.2m behind block
    target_pelvis_y_for_push = object1.data.root_pos_w[:, 1]
    desired_pelvis_z_for_push = object1_z_dim / 2.0 + 0.7 # Pelvis at standing height relative to block's center Z

    dist_pelvis_to_push_pos_x = torch.abs(pelvis_pos_x - target_pelvis_x_for_push)
    dist_pelvis_to_push_pos_y = torch.abs(pelvis_pos_y - target_pelvis_y_for_push)
    dist_pelvis_to_push_pos_z = torch.abs(pelvis_pos_z - desired_pelvis_z_for_push)

    # Combined distance for pelvis to push position (MANDATORY: continuous reward)
    reward_pelvis_push_pos = -(dist_pelvis_to_push_pos_x + dist_pelvis_to_push_pos_y + dist_pelvis_to_push_pos_z) * 0.3

    # Combine rewards based on phases (MANDATORY: continuous reward, handles batching)
    # If Object1 is far from target, focus on robot approaching Object1.
    # Once Object1 is close to target, focus on Object1 reaching target and robot maintaining push position.
    object1_far_from_target_condition = dist_object1_to_target_xyz > 0.5 # Threshold for "far"

    reward = torch.where(object1_far_from_target_condition,
                         reward_approach,
                         reward_object1_target + reward_pelvis_push_pos)

    # MANDATORY: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward to penalize collisions between the robot's pelvis and all objects in the scene.
    Encourages safe navigation.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (MANDATORY: using approved pattern)
    object1 = env.scene['Object1']  # Small Block
    object2 = env.scene['Object2']  # Medium Block
    object3 = env.scene['Object3']  # Large Block

    # Access required robot part(s) (MANDATORY: using approved pattern)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object dimensions (MANDATORY: hardcoded from object config)
    object1_x_dim = 1.0
    object1_y_dim = 1.0
    object1_z_dim = 0.3
    object2_x_dim = 1.0
    object2_y_dim = 1.0
    object2_z_dim = 0.6
    object3_x_dim = 1.0
    object3_y_dim = 1.0
    object3_z_dim = 0.9

    # Define collision thresholds for pelvis with objects (half-dimensions + small buffer)
    # Using a simple distance check to the center, with a threshold based on half-dimensions.
    # This creates a "box" around the object for collision detection.

    # Collision with Object1
    dist_pelvis_obj1_x = torch.abs(pelvis_pos[:, 0] - object1.data.root_pos_w[:, 0])
    dist_pelvis_obj1_y = torch.abs(pelvis_pos[:, 1] - object1.data.root_pos_w[:, 1])
    dist_pelvis_obj1_z = torch.abs(pelvis_pos[:, 2] - object1.data.root_pos_w[:, 2])

    collision_threshold_obj1_x = object1_x_dim / 2.0 + 0.1 # 0.1m buffer
    collision_threshold_obj1_y = object1_y_dim / 2.0 + 0.1
    collision_threshold_obj1_z = object1_z_dim / 2.0 + 0.1

    collision_obj1_condition = (dist_pelvis_obj1_x < collision_threshold_obj1_x) & \
                               (dist_pelvis_obj1_y < collision_threshold_obj1_y) & \
                               (dist_pelvis_obj1_z < collision_threshold_obj1_z)

    # Penalty for collision (MANDATORY: continuous reward, using torch.where for smooth transition)
    # The plan explicitly uses torch.where(-1.0, 0.0), so I will follow that.
    reward_collision_obj1 = torch.where(collision_obj1_condition, -1.0, 0.0)

    # Collision with Object2
    dist_pelvis_obj2_x = torch.abs(pelvis_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_pelvis_obj2_y = torch.abs(pelvis_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_pelvis_obj2_z = torch.abs(pelvis_pos[:, 2] - object2.data.root_pos_w[:, 2])

    collision_threshold_obj2_x = object2_x_dim / 2.0 + 0.1
    collision_threshold_obj2_y = object2_y_dim / 2.0 + 0.1
    collision_threshold_obj2_z = object2_z_dim / 2.0 + 0.1

    collision_obj2_condition = (dist_pelvis_obj2_x < collision_threshold_obj2_x) & \
                               (dist_pelvis_obj2_y < collision_threshold_obj2_y) & \
                               (dist_pelvis_obj2_z < collision_threshold_obj2_z)

    reward_collision_obj2 = torch.where(collision_obj2_condition, -1.0, 0.0)

    # Collision with Object3
    dist_pelvis_obj3_x = torch.abs(pelvis_pos[:, 0] - object3.data.root_pos_w[:, 0])
    dist_pelvis_obj3_y = torch.abs(pelvis_pos[:, 1] - object3.data.root_pos_w[:, 1])
    dist_pelvis_obj3_z = torch.abs(pelvis_pos[:, 2] - object3.data.root_pos_w[:, 2])

    collision_threshold_obj3_x = object3_x_dim / 2.0 + 0.1
    collision_threshold_obj3_y = object3_y_dim / 2.0 + 0.1
    collision_threshold_obj3_z = object3_z_dim / 2.0 + 0.1

    collision_obj3_condition = (dist_pelvis_obj3_x < collision_threshold_obj3_x) & \
                               (dist_pelvis_obj3_y < collision_threshold_obj3_y) & \
                               (dist_pelvis_obj3_z < collision_threshold_obj3_z)

    reward_collision_obj3 = torch.where(collision_obj3_condition, -1.0, 0.0)

    reward = reward_collision_obj1 + reward_collision_obj2 + reward_collision_obj3

    # MANDATORY: Complete normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_pelvis_height_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable standing posture
    by keeping its pelvis at a desired height and its feet on the ground.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part(s) (MANDATORY: using approved pattern)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Desired pelvis height for standing (MANDATORY: relative to ground, which is absolute Z)
    desired_pelvis_z = 0.7

    # Reward for maintaining pelvis height (MANDATORY: continuous reward)
    # Penalize deviation from desired height.
    reward_pelvis_height = -torch.abs(pelvis_pos_z - desired_pelvis_z) * 0.1

    # Reward for feet being on the ground (MANDATORY: relative to ground, which is absolute Z)
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2]
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2]

    # Feet should be close to ground (z=0), with a small buffer for foot thickness/contact.
    # Penalize deviation from a small positive Z value.
    reward_feet_on_ground = -(torch.abs(left_foot_pos_z - 0.05) + torch.abs(right_foot_pos_z - 0.05)) * 0.05 # 0.05m buffer for foot height

    reward = reward_pelvis_height + reward_feet_on_ground

    # MANDATORY: Complete normalization implementation
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
    Configuration for the reward terms used in the 'Position_Small_Block_for_robot_interaction' skill.
    """
    # Main reward for positioning the small block (MANDATORY: weight ~1.0)
    MainPositionSmallBlockReward = RewTerm(func=main_position_small_block_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance (MANDATORY: lower weight <1.0)
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "collision_reward"})

    # Shaping reward for pelvis height and stability (MANDATORY: lower weight <1.0)
    ShapingPelvisHeightStabilityReward = RewTerm(func=shaping_pelvis_height_stability_reward, weight=0.3,
                                                 params={"normalise": True, "normaliser_name": "pelvis_stability_reward"})