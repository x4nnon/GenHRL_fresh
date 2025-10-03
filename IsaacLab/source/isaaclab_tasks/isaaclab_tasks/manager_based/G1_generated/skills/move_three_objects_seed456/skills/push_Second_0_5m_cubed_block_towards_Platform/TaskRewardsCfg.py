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


def main_push_block_to_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for pushing the Second 0.5m cubed block (Object2) towards and onto the Platform (Object4).
    This reward encourages the block to approach the platform, align its Z-height, and position itself
    on the platform close to the edge, ready for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object2 = env.scene['Object2'] # Second 0.5m cubed block
    object4 = env.scene['Object4'] # Platform

    # Access object positions using approved patterns
    object2_pos = object2.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcoded object dimensions from object configuration (0.5m cubed block, platform height 0.001m, platform 2x2m)
    object2_half_size = 0.25 # Half size of a 0.5m cubed block
    object4_height = 0.001 # Height of the platform
    platform_half_x = 1.0 # Half width of the 2m platform
    platform_half_y = 1.0 # Half depth of the 2m platform

    # Calculate the horizontal distance (XY plane) between Object2 and Object4
    # This encourages Object2 to move closer to Object4 horizontally.
    horizontal_distance_obj2_obj4 = torch.norm(object2_pos[:, :2] - object4_pos[:, :2], dim=1)
    reward_approach_platform = -horizontal_distance_obj2_obj4 # Continuous reward: smaller distance -> higher reward

    # Calculate the target Z-height for Object2's center when it's on the platform.
    # Object2's center should be at platform_top_z + object2_half_size.
    platform_top_z = object4_pos[:, 2] + object4_height / 2.0
    target_obj2_z = platform_top_z + object2_half_size
    # Reward for Z-alignment: penalize deviation from the target Z-height.
    reward_z_alignment = -torch.abs(object2_pos[:, 2] - target_obj2_z) # Continuous reward: closer to target Z -> higher reward

    # Reward for Object2 being on the platform and close to the edge.
    # Assuming the block is pushed along the X-axis towards the platform.
    # The platform is 4m from the triangle of cubes, robot at 0,0,0.
    # If the platform is centered at X_platform_center, we want Object2 to be at
    # X_platform_center - platform_half_x + object2_half_size + small_offset (e.g., 0.1m into the platform).
    # This encourages it to be on the platform but not too deep, ready for the next skill.
    target_obj2_x_on_platform = object4_pos[:, 0] - platform_half_x + object2_half_size + 0.1
    reward_x_on_platform = -torch.abs(object2_pos[:, 0] - target_obj2_x_on_platform) # Continuous reward: closer to target X -> higher reward

    # Define a threshold to switch between "approach" and "on-platform" rewards.
    # When far, prioritize approaching. When close, prioritize Z-alignment and X-positioning.
    approach_threshold = 0.5 # meters
    is_approaching = horizontal_distance_obj2_obj4 > approach_threshold

    # Combine rewards using torch.where for smooth transition
    # If approaching, reward is based on horizontal distance.
    # If close/on platform, reward is a sum of Z-alignment and X-positioning.
    reward = torch.where(is_approaching,
                         reward_approach_platform,
                         reward_z_alignment + reward_x_on_platform)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hand_proximity_to_object2_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot's right palm to be close to Object2 and positioned correctly for pushing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object2 = env.scene['Object2'] # Second 0.5m cubed block
    object4 = env.scene['Object4'] # Platform (used for directional context)

    robot = env.scene["robot"]
    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    # Access object positions using approved patterns
    object2_pos = object2.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcoded object dimensions
    object2_half_size = 0.25 # Half size of a 0.5m cubed block

    # Calculate horizontal proximity of right palm to Object2
    # This encourages the hand to be close to the block in the XY plane.
    horizontal_distance_hand_obj2 = torch.norm(right_palm_pos[:, :2] - object2_pos[:, :2], dim=1)
    reward_hand_proximity_xy = -horizontal_distance_hand_obj2 # Continuous reward: smaller distance -> higher reward

    # Calculate target Z-height for the hand (center of Object2)
    # This encourages the hand to be at a suitable height for pushing the block.
    target_hand_z = object2_pos[:, 2]
    reward_hand_proximity_z = -torch.abs(right_palm_pos[:, 2] - target_hand_z) # Continuous reward: closer to target Z -> higher reward

    # Condition: Hand should be behind Object2 relative to Object4 (assuming push along X-axis).
    # If Object4 (platform) is at a higher X than Object2, the hand should be at a lower X than Object2.
    # This ensures the robot is pushing from the correct side.
    # A small buffer (e.g., 0.05m) is added to allow the hand to be slightly inside the block for pushing.
    condition_hand_behind_obj2 = (right_palm_pos[:, 0] < object2_pos[:, 0] + object2_half_size - 0.05)

    # Combine rewards: Apply penalty if hand is not correctly positioned for pushing.
    # If the hand is behind the block, sum the proximity rewards. Otherwise, apply a large penalty.
    shaping_reward = torch.where(condition_hand_behind_obj2,
                                 reward_hand_proximity_xy + reward_hand_proximity_z,
                                 torch.tensor(-10.0, device=env.device)) # Large penalty for incorrect pushing position

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward)
        RewNormalizer.update_stats(normaliser_name, shaping_reward)
        return scaled_reward
    return shaping_reward


def collision_avoidance_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_stability_reward") -> torch.Tensor:
    """
    Shaping reward to penalize collisions with non-target objects (Object1, Object3, Object4)
    and encourage the robot to maintain a stable, upright posture.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object1 = env.scene['Object1'] # First 0.5m cubed block
    object3 = env.scene['Object3'] # Third 0.5m cubed block
    object4 = env.scene['Object4'] # Platform

    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_palm_idx = robot.body_names.index('left_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]

    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    # Access object positions using approved patterns
    object1_pos = object1.data.root_pos_w
    object3_pos = object3.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcoded object dimensions
    object_half_size = 0.25 # Half size for 0.5m cubed blocks
    platform_height = 0.001 # Height of the platform
    platform_half_x = 1.0 # Half width of the 2m platform
    platform_half_y = 1.0 # Half depth of the 2m platform

    # Define a small buffer for collision avoidance (e.g., 5 cm)
    collision_buffer = 0.05

    # Collision avoidance for robot parts with Object1 (First 0.5m cubed block)
    # Penalize if any specified robot part gets too close to or penetrates Object1.
    # Minimum distance to object surface (center to surface is half_size) + buffer.
    min_dist_obj = object_half_size + collision_buffer
    
    dist_pelvis_obj1 = torch.norm(pelvis_pos - object1_pos, dim=1)
    dist_left_palm_obj1 = torch.norm(left_palm_pos - object1_pos, dim=1)
    dist_right_palm_obj1 = torch.norm(right_palm_pos - object1_pos, dim=1)
    
    collision_penalty_obj1 = torch.where((dist_pelvis_obj1 < min_dist_obj) |
                                         (dist_left_palm_obj1 < min_dist_obj) |
                                         (dist_right_palm_obj1 < min_dist_obj),
                                         torch.tensor(-10.0, device=env.device), torch.tensor(0.0, device=env.device))

    # Collision avoidance for robot parts with Object3 (Third 0.5m cubed block)
    # Similar logic as for Object1.
    dist_pelvis_obj3 = torch.norm(pelvis_pos - object3_pos, dim=1)
    dist_left_palm_obj3 = torch.norm(left_palm_pos - object3_pos, dim=1)
    dist_right_palm_obj3 = torch.norm(right_palm_pos - object3_pos, dim=1)
    
    collision_penalty_obj3 = torch.where((dist_pelvis_obj3 < min_dist_obj) |
                                         (dist_left_palm_obj3 < min_dist_obj) |
                                         (dist_right_palm_obj3 < min_dist_obj),
                                         torch.tensor(-10.0, device=env.device), torch.tensor(0.0, device=env.device))

    # Collision avoidance with Object4 (Platform)
    # Penalize if robot body parts (pelvis, left hand) are on the platform.
    # The right hand is allowed to be on the platform if it's pushing Object2.
    platform_top_z = object4_pos[:, 2] + platform_height / 2.0

    # Check if pelvis is on the platform (within X/Y bounds and correct Z)
    is_pelvis_on_platform_xy = (pelvis_pos[:, 0] > object4_pos[:, 0] - platform_half_x + collision_buffer) & \
                               (pelvis_pos[:, 0] < object4_pos[:, 0] + platform_half_x - collision_buffer) & \
                               (pelvis_pos[:, 1] > object4_pos[:, 1] - platform_half_y + collision_buffer) & \
                               (pelvis_pos[:, 1] < object4_pos[:, 1] + platform_half_y - collision_buffer)
    # Pelvis Z should be above platform top, but not too high (e.g., within 0.5m above it)
    is_pelvis_on_platform_z = (pelvis_pos[:, 2] > platform_top_z - collision_buffer) & (pelvis_pos[:, 2] < platform_top_z + 0.5)
    collision_penalty_pelvis_platform = torch.where(is_pelvis_on_platform_xy & is_pelvis_on_platform_z,
                                                    torch.tensor(-10.0, device=env.device), torch.tensor(0.0, device=env.device))

    # Check if left palm is on the platform (similar logic)
    is_left_palm_on_platform_xy = (left_palm_pos[:, 0] > object4_pos[:, 0] - platform_half_x + collision_buffer) & \
                                  (left_palm_pos[:, 0] < object4_pos[:, 0] + platform_half_x - collision_buffer) & \
                                  (left_palm_pos[:, 1] > object4_pos[:, 1] - platform_half_y + collision_buffer) & \
                                  (left_palm_pos[:, 1] < object4_pos[:, 1] + platform_half_y - collision_buffer)
    is_left_palm_on_platform_z = (left_palm_pos[:, 2] > platform_top_z - collision_buffer) & (left_palm_pos[:, 2] < platform_top_z + 0.5)
    collision_penalty_left_palm_platform = torch.where(is_left_palm_on_platform_xy & is_left_palm_on_platform_z,
                                                       torch.tensor(-10.0, device=env.device), torch.tensor(0.0, device=env.device))

    # Robot stability: Encourage pelvis to stay at a reasonable height.
    # This prevents the robot from falling or crouching too low.
    target_pelvis_z = 0.7 # A typical standing height for the pelvis
    reward_pelvis_height = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z) # Continuous reward: closer to target Z -> higher reward

    # Combine all penalties and rewards for shaping reward 2
    shaping_reward = collision_penalty_obj1 + collision_penalty_obj3 + \
                     collision_penalty_pelvis_platform + collision_penalty_left_palm_platform + \
                     reward_pelvis_height

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward)
        RewNormalizer.update_stats(normaliser_name, shaping_reward)
        return scaled_reward
    return shaping_reward


@configclass
class TaskRewardsCfg:
    # Main reward for pushing Object2 onto the Platform
    MainPushBlockToPlatformReward = RewTerm(func=main_push_block_to_platform_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for robot hand proximity and correct pushing position
    HandProximityToObject2Reward = RewTerm(func=hand_proximity_to_object2_reward, weight=0.6,
                                           params={"normalise": True, "normaliser_name": "hand_proximity_reward"})

    # Shaping reward for collision avoidance and robot stability
    CollisionAvoidanceAndStabilityReward = RewTerm(func=collision_avoidance_and_stability_reward, weight=0.3,
                                                   params={"normalise": True, "normaliser_name": "collision_stability_reward"})