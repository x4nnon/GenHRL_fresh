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


def position_large_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for positioning the Large Block (Object3) relative to the Medium Block (Object2).
    This reward encourages Object3 to be placed behind Object2, aligned in Y, and on the ground,
    forming the third step of the stairs.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object2 = env.scene['Object2'] # Medium Block for robot interaction
    object3 = env.scene['Object3'] # Large Block for robot interaction

    # Access object positions
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Define target relative position for Object3 based on Object2.
    # For stairs, Object3's center should be 1.0m behind Object2's center in x,
    # aligned in y, and on the ground (z=0.0).
    # This creates a step where the robot can land on Object2 and then step onto Object3.
    # This uses relative positioning based on Object2's current position.
    target_object3_x = object2_pos[:, 0] - 1.0 # Target x for Object3: 1.0m behind Object2's center
    target_object3_y = object2_pos[:, 1] # Target y for Object3: Aligned with Object2's y
    target_object3_z = 0.0 # Target z for Object3: On the ground. This is an absolute Z, which is allowed for ground positioning.

    # Calculate distance components for Object3 to its target relative position
    # Using absolute differences for each dimension to create a continuous reward.
    # This ensures the reward is always negative or zero, increasing as distance decreases.
    distance_x_obj3 = torch.abs(object3_pos[:, 0] - target_object3_x)
    distance_y_obj3 = torch.abs(object3_pos[:, 1] - target_object3_y)
    distance_z_obj3 = torch.abs(object3_pos[:, 2] - target_object3_z)

    # Reward is negative absolute distance, so it increases as distance decreases.
    # This provides a continuous shaping reward that guides the block to the target.
    reward = -distance_x_obj3 - distance_y_obj3 - distance_z_obj3

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def approach_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_reward") -> torch.Tensor:
    """
    This reward encourages the robot to approach the Large Block (Object3) and get its hands close to it,
    facilitating the pushing action. It focuses on the horizontal distance (x, y) between the robot's
    hands and Object3. This reward is active when the robot is still some distance away from the final
    pushing position and Object3 is not yet at its final target.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object2 = env.scene['Object2'] # Medium Block (needed for activation condition)
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Calculate average hand position for simplicity in approach
    # This is a relative calculation between robot parts.
    avg_hand_pos_x = (left_hand_pos[:, 0] + right_hand_pos[:, 0]) / 2
    avg_hand_pos_y = (left_hand_pos[:, 1] + right_hand_pos[:, 1]) / 2
    avg_hand_pos_z = (left_hand_pos[:, 2] + right_hand_pos[:, 2]) / 2

    # Calculate relative distances from average hand position to Object3
    # These are relative distances between robot hands and Object3.
    distance_x_hands_obj3 = torch.abs(object3.data.root_pos_w[:, 0] - avg_hand_pos_x)
    distance_y_hands_obj3 = torch.abs(object3.data.root_pos_w[:, 1] - avg_hand_pos_y)
    distance_z_hands_obj3 = torch.abs(object3.data.root_pos_w[:, 2] - avg_hand_pos_z)

    # Activation condition:
    # 1. Robot hands are within a certain range of Object3 (e.g., 2.0m in XY plane).
    # 2. Object3 is not yet at its final target X position (to prevent conflict with primary reward).
    #    Target X for Object3 is object2.x - 1.0. So, if Object3.x is greater than (object2.x - 1.0 - 0.1)
    #    it means it's still further away or slightly past the target, but not settled.
    # This condition uses relative distances and object positions.
    distance_to_obj3_xy = torch.sqrt(distance_x_hands_obj3**2 + distance_y_hands_obj3**2)
    obj3_not_at_target_x = object3.data.root_pos_w[:, 0] > (object2.data.root_pos_w[:, 0] - 1.0 - 0.1)

    activation_condition = (distance_to_obj3_xy < 2.0) & obj3_not_at_target_x

    # Reward for getting hands close to Object3 (negative absolute distance).
    # This is a continuous reward that becomes more positive as hands get closer.
    reward = -distance_x_hands_obj3 - distance_y_hands_obj3 - distance_z_hands_obj3

    # Apply activation condition: reward is 0.0 if condition is not met.
    # This ensures the reward is only active when relevant.
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_posture_avoid_collision_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_collision_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain a stable, upright posture (pelvis at a reasonable height)
    and avoid collisions with any of the blocks (Object1, Object2, Object3) while performing the task.
    This is crucial for overall task success and preparing for the next skill (climbing).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Reward for maintaining pelvis height
    # Target pelvis height for stability (hardcoded as per plan). This is an absolute Z target, which is allowed.
    pelvis_target_z = 0.7
    # Reward is negative absolute difference, so it's maximized when pelvis_pos_z is close to pelvis_target_z.
    reward_pelvis_height = -torch.abs(pelvis_pos_z - pelvis_target_z)

    # Collision avoidance for robot pelvis/legs with blocks
    # Define a collision threshold. For a 1m block, half_dim = 0.5m.
    # A threshold of 0.7m means the pelvis center should be at least 0.7m from the block center.
    # This accounts for the block's half-dimension and a safety margin. This is a hardcoded threshold.
    collision_threshold = 0.7

    # Calculate absolute distances from pelvis to each block's center in each dimension
    # These are relative distances between the robot's pelvis and each object.
    # Object1 (Small Block)
    dist_pelvis_obj1_x = torch.abs(pelvis_pos[:, 0] - object1.data.root_pos_w[:, 0])
    dist_pelvis_obj1_y = torch.abs(pelvis_pos[:, 1] - object1.data.root_pos_w[:, 1])
    dist_pelvis_obj1_z = torch.abs(pelvis_pos[:, 2] - object1.data.root_pos_w[:, 2])

    # Object2 (Medium Block)
    dist_pelvis_obj2_x = torch.abs(pelvis_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_pelvis_obj2_y = torch.abs(pelvis_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_pelvis_obj2_z = torch.abs(pelvis_pos[:, 2] - object2.data.root_pos_w[:, 2])

    # Object3 (Large Block)
    dist_pelvis_obj3_x = torch.abs(pelvis_pos[:, 0] - object3.data.root_pos_w[:, 0])
    dist_pelvis_obj3_y = torch.abs(pelvis_pos[:, 1] - object3.data.root_pos_w[:, 1])
    dist_pelvis_obj3_z = torch.abs(pelvis_pos[:, 2] - object3.data.root_pos_w[:, 2])

    # Find the minimum distance to any block in each dimension
    # This ensures the robot avoids collision with any of the blocks.
    min_dist_x = torch.min(torch.min(dist_pelvis_obj1_x, dist_pelvis_obj2_x), dist_pelvis_obj3_x)
    min_dist_y = torch.min(torch.min(dist_pelvis_obj1_y, dist_pelvis_obj2_y), dist_pelvis_obj3_y)
    min_dist_z = torch.min(torch.min(dist_pelvis_obj1_z, dist_pelvis_obj2_z), dist_pelvis_obj3_z)

    # Reward for avoiding collision:
    # This reward is 0 if the distance is greater than or equal to the threshold (safe).
    # It becomes negative (penalty) if the distance is less than the threshold (collision imminent/occurring).
    # This creates a continuous penalty that increases as the robot gets closer than the safe distance.
    # This uses a continuous reward based on the difference from the threshold.
    reward_collision_x = torch.where(min_dist_x < collision_threshold, min_dist_x - collision_threshold, 0.0)
    reward_collision_y = torch.where(min_dist_y < collision_threshold, min_dist_y - collision_threshold, 0.0)
    reward_collision_z = torch.where(min_dist_z < collision_threshold, min_dist_z - collision_threshold, 0.0)

    # Combine pelvis height reward and collision avoidance rewards
    reward = reward_pelvis_height + reward_collision_x + reward_collision_y + reward_collision_z

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
    # Primary reward for positioning the Large Block (Object3)
    # Weight 1.0 as it's the main objective for this skill.
    PositionLargeBlockMainReward = RewTerm(func=position_large_block_main_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for encouraging robot hands to approach Object3
    # Weight 0.4 as it's a supporting reward to guide the robot.
    ApproachLargeBlockReward = RewTerm(func=approach_large_block_reward, weight=0.4,
                                       params={"normalise": True, "normaliser_name": "approach_reward"})

    # Shaping reward for maintaining posture and avoiding collisions
    # Weight 0.2 as it's a general behavior shaping reward, less critical than direct task progress.
    MaintainPostureAvoidCollisionReward = RewTerm(func=maintain_posture_avoid_collision_reward, weight=0.2,
                                                  params={"normalise": True, "normaliser_name": "posture_collision_reward"})