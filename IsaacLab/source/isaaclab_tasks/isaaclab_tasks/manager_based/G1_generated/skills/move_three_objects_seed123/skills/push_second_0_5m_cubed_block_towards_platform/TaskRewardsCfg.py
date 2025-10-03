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
    Main reward for pushing the 'second 0.5m cubed block' (Object2) towards the 'platform' (Object4).
    The reward encourages moving Object2 closer to Object4 along the y-axis (as per environment setup)
    and aligning it along the x-axis, while penalizing overshooting the target y-position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object4 = env.scene['Object4'] # platform

    # Access object positions using approved patterns
    object2_pos = object2.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # The task description implies the platform is at x=2m, y=2m and the triangle of blocks is around x=2m, y=0m.
    # This means the primary push direction is along the y-axis.
    # The goal is "moved significantly closer... but not yet fully on it".
    # The block is 0.5m cubed. To be "not yet fully on it", its center should be some distance before the platform's center.
    # Let's define the target y-position for Object2's center to be 0.5m before Object4's center along the y-axis.
    # This allows for the block to be near the platform's edge without being fully on it.
    # This is a relative target position based on the platform's position.
    target_y_pos_obj2 = object4_pos[:, 1] - 0.5 # Target y for Object2's center
    # For x-alignment, the block's x-position should align with the platform's x-position.
    # This is a relative target position based on the platform's position.
    target_x_pos_obj2 = object4_pos[:, 0] # Target x for Object2's center

    # Reward for moving Object2 towards the target y-position.
    # Using negative absolute difference for a continuous reward that is maximized at 0 difference.
    # This uses relative distance between object2 and the target y position relative to object4.
    reward_push_y = -torch.abs(object2_pos[:, 1] - target_y_pos_obj2)

    # Reward for aligning Object2 with the platform's x-position.
    # Using negative absolute difference for a continuous reward.
    # This uses relative distance between object2 and the target x position relative to object4.
    reward_align_x = -torch.abs(object2_pos[:, 0] - target_x_pos_obj2)

    # Combine rewards. Prioritize y-axis movement as it's the primary push direction.
    # Adding a constant to make the reward generally positive and encourage progress.
    primary_reward = (reward_push_y * 2.0) + (reward_align_x * 1.0) + 5.0

    # Condition to prevent overshooting: if Object2's y-position is past the target_y_pos_obj2, penalize.
    # This ensures "not yet fully on it". A large negative penalty for overshooting.
    # This condition is based on relative positions.
    overshoot_condition = object2_pos[:, 1] > target_y_pos_obj2
    primary_reward = torch.where(overshoot_condition, primary_reward - 10.0, primary_reward)

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def shaping_hand_proximity_contact_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot's hands to be close to and make contact with
    'second 0.5m cubed block' (Object2) from the correct pushing side (negative y-direction relative to block).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object2 = env.scene['Object2'] # second 0.5m cubed block

    # Access robot part positions using approved patterns
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    # Access object position using approved patterns
    object2_pos = object2.data.root_pos_w

    # Object2 dimensions: 0.5m cubed. Half size = 0.25m. (Hardcoded from object configuration as per rules)
    block_half_size = 0.25

    # Define target hand position relative to the block for pushing.
    # The robot should push from the side of the block facing away from the platform (negative y-direction).
    # Target y: slightly behind the block's edge (e.g., -block_half_size - 0.05m buffer).
    # Target x: aligned with the block's x-center.
    # Target z: aligned with the block's z-center (assuming block is on ground, its center is at 0.25m height).
    # These are relative offsets from the block's center.
    target_hand_y_rel_obj2 = -block_half_size - 0.05 # Relative y-offset from block's center
    target_hand_z_rel_obj2 = 0.0 # Relative z-offset from block's center (at block's height)

    # Calculate relative distances for right hand to Object2
    # These are relative distances between the hand and the object.
    distance_right_hand_obj2_y = right_hand_pos[:, 1] - object2_pos[:, 1]
    distance_right_hand_obj2_x = right_hand_pos[:, 0] - object2_pos[:, 0]
    distance_right_hand_obj2_z = right_hand_pos[:, 2] - object2_pos[:, 2]

    # Reward for right hand being close to the target pushing position relative to Object2.
    # Using negative absolute differences for continuous rewards.
    # Reward is based on relative distances to the target relative position.
    reward_right_hand_proximity = -torch.abs(distance_right_hand_obj2_y - target_hand_y_rel_obj2) \
                                -torch.abs(distance_right_hand_obj2_x) \
                                -torch.abs(distance_right_hand_obj2_z - target_hand_z_rel_obj2)

    # Calculate relative distances for left hand to Object2
    # These are relative distances between the hand and the object.
    distance_left_hand_obj2_y = left_hand_pos[:, 1] - object2_pos[:, 1]
    distance_left_hand_obj2_x = left_hand_pos[:, 0] - object2_pos[:, 0]
    distance_left_hand_obj2_z = left_hand_pos[:, 2] - object2_pos[:, 2]

    # Reward for left hand being close to the target pushing position relative to Object2.
    # Reward is based on relative distances to the target relative position.
    reward_left_hand_proximity = -torch.abs(distance_left_hand_obj2_y - target_hand_y_rel_obj2) \
                                -torch.abs(distance_left_hand_obj2_x) \
                                -torch.abs(distance_left_hand_obj2_z - target_hand_z_rel_obj2)

    shaping_reward1 = (reward_right_hand_proximity + reward_left_hand_proximity) * 0.5 + 2.0

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1

def shaping_robot_stability_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_collision_reward") -> torch.Tensor:
    """
    Shaping reward to encourage robot stability (pelvis height) and penalize collisions
    of non-hand robot parts (pelvis, ankles) with all objects in the scene.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1'] # first 0.5m cubed block
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object3 = env.scene['Object3'] # third 0.5m cubed block
    object4 = env.scene['Object4'] # platform

    # Access robot part positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx]
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx]

    # Pelvis height reward for stability. Target height is 0.7m.
    # Using negative absolute difference for a continuous reward.
    # This is one of the few cases where an absolute Z position is allowed, as it relates to stability on the ground.
    target_pelvis_z = 0.7
    reward_pelvis_height = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Collision avoidance for feet and pelvis with all blocks and platform.
    # Define a safe distance for collision avoidance.
    # Block half size = 0.25m. Assuming average robot part radius of 0.1m.
    # So, a collision threshold for blocks could be 0.25 (block half size) + 0.1 (part radius) = 0.35m.
    # For the platform, it's a large surface, so a general threshold to its center might be different.
    # Let's use a buffer of 0.1m for general clearance.
    # These thresholds are hardcoded based on object configuration and assumed robot part sizes, as per rules.
    block_collision_threshold = 0.35 + 0.1 # 0.45m from center to center
    platform_collision_threshold = 0.5 # A general threshold for the large platform

    # Initialize collision penalty
    collision_penalty = torch.zeros_like(reward_pelvis_height)

    # List of objects to avoid and robot parts to check
    objects_to_avoid = [object1, object2, object3, object4]
    robot_parts_to_check = [pelvis_pos, right_ankle_pos, left_ankle_pos]

    # Iterate through each object and robot part to calculate penalties
    for obj in objects_to_avoid:
        obj_pos = obj.data.root_pos_w # Access object position using approved patterns
        current_threshold = block_collision_threshold if obj != object4 else platform_collision_threshold

        for part_pos in robot_parts_to_check:
            # Calculate Euclidean distance between robot part and object center
            # This uses relative distance between robot part and object.
            dist = torch.norm(part_pos - obj_pos, dim=1)
            # Apply continuous penalty if distance is below the threshold.
            # Penalty scales with how much the distance is below the threshold.
            penalty_term = torch.where(dist < current_threshold, -(current_threshold - dist) * 5.0, 0.0)
            collision_penalty += penalty_term

    # Combine stability reward and collision penalty. Add a constant to keep it generally positive.
    shaping_reward2 = reward_pelvis_height + collision_penalty + 1.0

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2

@configclass
class TaskRewardsCfg:
    # Main reward for moving the block towards the platform, with higher weight.
    main_push_block_towards_platform_reward = RewTerm(func=main_push_block_towards_platform_reward, weight=1.0,
                                                      params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for hand proximity and contact with the block, with a moderate weight.
    shaping_hand_proximity_contact_reward = RewTerm(func=shaping_hand_proximity_contact_reward, weight=0.6,
                                                    params={"normalise": True, "normaliser_name": "hand_proximity_reward"})

    # Shaping reward for robot stability and collision avoidance, with a lower weight.
    shaping_robot_stability_collision_avoidance_reward = RewTerm(func=shaping_robot_stability_collision_avoidance_reward, weight=0.3,
                                                                 params={"normalise": True, "normaliser_name": "stability_collision_reward"})