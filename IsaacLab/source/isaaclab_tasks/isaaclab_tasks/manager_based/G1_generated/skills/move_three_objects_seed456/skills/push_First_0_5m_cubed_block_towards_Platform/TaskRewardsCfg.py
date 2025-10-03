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


def push_block_onto_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Primary reward for pushing Object1 (First 0.5m cubed block) onto Object4 (Platform).
    Rewards Object1 being centered on the platform and at the correct height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1']  # First 0.5m cubed block
    object4 = env.scene['Object4']  # Platform

    # Access object positions using approved patterns
    object1_pos = object1.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcode object dimensions from the object configuration
    # Object1 is 0.5m cubed
    object1_half_size = 0.5 / 2.0  # 0.25m
    # Object4 is x=2m, y=2m, z=0.001. Assuming root_pos_w is its center.
    # The Z value 0.001 likely refers to its thickness or a base height.
    # Let's assume the platform's top surface is at object4_pos[:, 2] + (0.001 / 2) if 0.001 is thickness,
    # or simply 0.001 if root_pos_w[:, 2] is already the top surface.
    # Given "z=0.001", it's most likely the platform's top surface is at Z=0.001.
    # So, Object1's center Z should be 0.001 (platform top) + object1_half_size.
    platform_top_z_absolute = 0.001 # From object config, assuming this is the absolute Z of the platform's top surface.
    platform_half_x = 2.0 / 2.0 # 1.0m
    platform_half_y = 2.0 / 2.0 # 1.0m

    # Calculate target Z position for Object1's center
    # Object1's center should be at platform_top_z_absolute + object1_half_size
    target_object1_z = platform_top_z_absolute + object1_half_size

    # Calculate relative distances for Object1 to Object4's center
    # Reward for X and Y alignment: closer to Object4's center is better.
    # Using negative absolute distance for continuous reward.
    # The reward should be capped or shaped to allow for "on or very close to the edge".
    # Let's use a smooth penalty that increases as Object1 moves outside the platform bounds.
    # A simple negative squared distance or negative absolute distance works well.
    # For "on or very close to the edge", we can define a target region.
    # The target X/Y for Object1 is Object4's center.

    # Distance in X and Y from Object1's center to Object4's center
    distance_obj1_obj4_x = object1_pos[:, 0] - object4_pos[:, 0]
    distance_obj1_obj4_y = object1_pos[:, 1] - object4_pos[:, 1]

    # Distance in Z from Object1's center to the target Z height
    distance_obj1_z = object1_pos[:, 2] - target_object1_z

    # Reward for X and Y alignment: Penalize deviation from platform center.
    # Use a smooth negative exponential or inverse distance for continuous reward.
    # For "on or very close to the edge", we can use a threshold.
    # Let's use a negative squared distance for smoothness, which heavily penalizes large deviations.
    # We want to reward being within the platform's extent.
    # A simple negative absolute distance is also continuous and effective.
    # Let's use negative absolute distance, which is linear and continuous.
    reward_x_pos = -torch.abs(distance_obj1_obj4_x)
    reward_y_pos = -torch.abs(distance_obj1_obj4_y)

    # Reward for Z alignment: Penalize deviation from target Z height.
    reward_z_pos = -torch.abs(distance_obj1_z)

    # Combine rewards. The sum provides a continuous gradient towards the goal.
    reward = reward_x_pos + reward_y_pos + reward_z_pos

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hands_to_block_pushing_side_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward1") -> torch.Tensor:
    """
    Shaping reward 1: Encourages robot's hands to be close to Object1's pushing side.
    This facilitates the pushing action by positioning the robot behind the block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object1 = env.scene['Object1']  # First 0.5m cubed block
    robot = env.scene["robot"]

    # Access robot part indices and positions using approved patterns
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    # Access object position using approved pattern
    object1_pos = object1.data.root_pos_w

    # Hardcode object dimensions from the object configuration
    # Object1 is 0.5m cubed
    object1_half_size = 0.5 / 2.0  # 0.25m

    # Determine the target pushing side of Object1.
    # Assuming the platform is generally in the positive X direction from the initial block position.
    # The robot should push Object1 from its negative X side towards positive X.
    # Target X for hands: slightly behind Object1's negative X edge.
    # A small buffer (e.g., 0.05m) behind the block's edge.
    target_hand_x_offset = -object1_half_size - 0.05 # 0.05m behind the block's edge

    # Calculate target hand positions relative to Object1's center
    # Target X: object1_pos[:, 0] + target_hand_x_offset
    # Target Y: object1_pos[:, 1] (aligned with block center)
    # Target Z: object1_pos[:, 2] (aligned with block center)

    # Reward for right hand proximity to the target pushing side
    # Use negative absolute distance for continuous reward.
    reward_rh_x = -torch.abs(right_hand_pos[:, 0] - (object1_pos[:, 0] + target_hand_x_offset))
    reward_rh_y = -torch.abs(right_hand_pos[:, 1] - object1_pos[:, 1])
    reward_rh_z = -torch.abs(right_hand_pos[:, 2] - object1_pos[:, 2])
    reward_right_hand = reward_rh_x + reward_rh_y + reward_rh_z

    # Reward for left hand proximity to the target pushing side
    reward_lh_x = -torch.abs(left_hand_pos[:, 0] - (object1_pos[:, 0] + target_hand_x_offset))
    reward_lh_y = -torch.abs(left_hand_pos[:, 1] - object1_pos[:, 1])
    reward_lh_z = -torch.abs(left_hand_pos[:, 2] - object1_pos[:, 2])
    reward_left_hand = reward_lh_x + reward_lh_y + reward_lh_z

    # Sum of rewards for both hands. This encourages either or both hands to be well-placed.
    reward = reward_right_hand + reward_left_hand

    # Activation condition: This reward should be active when Object1 is not yet on the platform.
    # Use the primary reward's Z component as a proxy for "not yet on platform".
    # If Object1's Z is significantly below the target_object1_z, then activate.
    # Re-calculate target_object1_z for activation check
    platform_top_z_absolute = 0.001 # From object config
    target_object1_z_for_activation = platform_top_z_absolute + object1_half_size
    # Activate if Object1's Z is more than 0.05m below the target Z.
    activation_condition = (object1_pos[:, 2] < (target_object1_z_for_activation - 0.05))
    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_and_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward2") -> torch.Tensor:
    """
    Shaping reward 2: Encourages collision avoidance with Object2 and Object3,
    and maintains a stable, upright posture by keeping the pelvis at a reasonable height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object2 = env.scene['Object2']  # Second 0.5m cubed block
    object3 = env.scene['Object3']  # Third 0.5m cubed block
    robot = env.scene["robot"]

    # Access robot part index and position using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Access object positions using approved patterns
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Hardcode object dimensions from the object configuration
    # Blocks are 0.5m cubed
    block_half_size = 0.5 / 2.0  # 0.25m

    # Collision avoidance for Object2 and Object3
    # Define a safe clearance distance. This is a hardcoded value based on general robot/object sizes.
    # It's a buffer beyond the block's half-size to ensure the robot doesn't get too close.
    safe_clearance = 0.2 # 0.2m buffer
    collision_threshold = block_half_size + safe_clearance # e.g., 0.25 + 0.2 = 0.45m

    # Calculate Euclidean distance from pelvis to Object2
    dist_pelvis_obj2 = torch.norm(pelvis_pos - object2_pos, dim=1)
    # Penalize if distance is below the threshold. Use max(0, threshold - distance) for continuous penalty.
    # The negative sign makes it a reward (less penalty = higher reward).
    reward_avoid_obj2 = torch.where(dist_pelvis_obj2 < collision_threshold, -(collision_threshold - dist_pelvis_obj2), 0.0)

    # Calculate Euclidean distance from pelvis to Object3
    dist_pelvis_obj3 = torch.norm(pelvis_pos - object3_pos, dim=1)
    # Penalize if distance is below the threshold.
    reward_avoid_obj3 = torch.where(dist_pelvis_obj3 < collision_threshold, -(collision_threshold - dist_pelvis_obj3), 0.0)

    # Pelvis height stability reward
    # Encourage pelvis to stay around a target height (e.g., 0.7m for a humanoid robot).
    # This is a hardcoded value representing a stable standing height.
    target_pelvis_z = 0.7
    # Penalize deviation from the target Z height using negative absolute distance.
    reward_pelvis_height = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Combine all shaping rewards
    reward = reward_avoid_obj2 + reward_avoid_obj3 + reward_pelvis_height

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for getting Object1 onto the Platform
    PushBlockOntoPlatformReward = RewTerm(func=push_block_onto_platform_reward, weight=1.0,
                                          params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward 1: Encourage hands to be near the pushing side of Object1
    HandsToBlockPushingSideReward = RewTerm(func=hands_to_block_pushing_side_reward, weight=0.3,
                                            params={"normalise": True, "normaliser_name": "shaping_reward1"})

    # Shaping reward 2: Collision avoidance and pelvis height stability
    CollisionAvoidanceAndPelvisHeightReward = RewTerm(func=collision_avoidance_and_pelvis_height_reward, weight=0.2,
                                                      params={"normalise": True, "normaliser_name": "shaping_reward2"})