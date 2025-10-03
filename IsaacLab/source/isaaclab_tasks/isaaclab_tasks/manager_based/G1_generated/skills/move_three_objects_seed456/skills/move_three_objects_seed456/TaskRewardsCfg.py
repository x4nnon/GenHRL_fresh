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


def main_block_to_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Primary reward: Measures the total progress of all three blocks towards and onto the platform.
    It encourages each block to be as close as possible to the center of the platform's top surface.
    The reward is inversely proportional to the sum of the squared Euclidean distances (x, y, z components considered separately)
    from each block's center to the platform's center. A small constant is added to the denominator to prevent division by zero
    and ensure a continuous gradient.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    platform = env.scene['Object4']

    # Hardcoded platform dimensions from object configuration: 0.001m height.
    # Platform root_pos_w is at its base, so center z is root_pos_w[:, 2] + height / 2.0
    platform_height = 0.001
    platform_center_z = platform.data.root_pos_w[:, 2] + platform_height / 2.0

    # Calculate distance for Object1 to platform center
    # Using relative distances for x, y, z components
    dist_x1 = object1.data.root_pos_w[:, 0] - platform.data.root_pos_w[:, 0]
    dist_y1 = object1.data.root_pos_w[:, 1] - platform.data.root_pos_w[:, 1]
    dist_z1 = object1.data.root_pos_w[:, 2] - platform_center_z
    distance_sq1 = dist_x1**2 + dist_y1**2 + dist_z1**2

    # Calculate distance for Object2 to platform center
    dist_x2 = object2.data.root_pos_w[:, 0] - platform.data.root_pos_w[:, 0]
    dist_y2 = object2.data.root_pos_w[:, 1] - platform.data.root_pos_w[:, 1]
    dist_z2 = object2.data.root_pos_w[:, 2] - platform_center_z
    distance_sq2 = dist_x2**2 + dist_y2**2 + dist_z2**2

    # Calculate distance for Object3 to platform center
    dist_x3 = object3.data.root_pos_w[:, 0] - platform.data.root_pos_w[:, 0]
    dist_y3 = object3.data.root_pos_w[:, 1] - platform.data.root_pos_w[:, 1]
    dist_z3 = object3.data.root_pos_w[:, 2] - platform_center_z
    distance_sq3 = dist_x3**2 + dist_y3**2 + dist_z3**2

    # Sum of inverse distances for all three objects, encouraging all to be on the platform.
    # Add a small constant (0.1) to avoid division by zero and ensure a continuous gradient.
    reward = (1.0 / (distance_sq1 + 0.1)) + (1.0 / (distance_sq2 + 0.1)) + (1.0 / (distance_sq3 + 0.1))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hands_to_block_proximity_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hands_proximity_reward") -> torch.Tensor:
    """
    Shaping Reward 1: Encourages the robot's hands to be close to any of the three blocks.
    This guides the robot to approach the blocks for pushing. It uses the minimum distance
    from either hand to any of the three blocks.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access the required robot parts using approved patterns
    robot = env.scene["robot"]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Calculate squared Euclidean distance from left hand to Object1
    # Using relative distances for x, y, z components
    dist_lh_o1_x = object1.data.root_pos_w[:, 0] - left_hand_pos[:, 0]
    dist_lh_o1_y = object1.data.root_pos_w[:, 1] - left_hand_pos[:, 1]
    dist_lh_o1_z = object1.data.root_pos_w[:, 2] - left_hand_pos[:, 2]
    dist_lh_o1_sq = dist_lh_o1_x**2 + dist_lh_o1_y**2 + dist_lh_o1_z**2

    # Calculate squared Euclidean distance from right hand to Object1
    dist_rh_o1_x = object1.data.root_pos_w[:, 0] - right_hand_pos[:, 0]
    dist_rh_o1_y = object1.data.root_pos_w[:, 1] - right_hand_pos[:, 1]
    dist_rh_o1_z = object1.data.root_pos_w[:, 2] - right_hand_pos[:, 2]
    dist_rh_o1_sq = dist_rh_o1_x**2 + dist_rh_o1_y**2 + dist_rh_o1_z**2

    # Calculate squared Euclidean distance from left hand to Object2
    dist_lh_o2_x = object2.data.root_pos_w[:, 0] - left_hand_pos[:, 0]
    dist_lh_o2_y = object2.data.root_pos_w[:, 1] - left_hand_pos[:, 1]
    dist_lh_o2_z = object2.data.root_pos_w[:, 2] - left_hand_pos[:, 2]
    dist_lh_o2_sq = dist_lh_o2_x**2 + dist_lh_o2_y**2 + dist_lh_o2_z**2

    # Calculate squared Euclidean distance from right hand to Object2
    dist_rh_o2_x = object2.data.root_pos_w[:, 0] - right_hand_pos[:, 0]
    dist_rh_o2_y = object2.data.root_pos_w[:, 1] - right_hand_pos[:, 1]
    dist_rh_o2_z = object2.data.root_pos_w[:, 2] - right_hand_pos[:, 2]
    dist_rh_o2_sq = dist_rh_o2_x**2 + dist_rh_o2_y**2 + dist_rh_o2_z**2

    # Calculate squared Euclidean distance from left hand to Object3
    dist_lh_o3_x = object3.data.root_pos_w[:, 0] - left_hand_pos[:, 0]
    dist_lh_o3_y = object3.data.root_pos_w[:, 1] - left_hand_pos[:, 1]
    dist_lh_o3_z = object3.data.root_pos_w[:, 2] - left_hand_pos[:, 2]
    dist_lh_o3_sq = dist_lh_o3_x**2 + dist_lh_o3_y**2 + dist_lh_o3_z**2

    # Calculate squared Euclidean distance from right hand to Object3
    dist_rh_o3_x = object3.data.root_pos_w[:, 0] - right_hand_pos[:, 0]
    dist_rh_o3_y = object3.data.root_pos_w[:, 1] - right_hand_pos[:, 1]
    dist_rh_o3_z = object3.data.root_pos_w[:, 2] - right_hand_pos[:, 2]
    dist_rh_o3_sq = dist_rh_o3_x**2 + dist_rh_o3_y**2 + dist_rh_o3_z**2

    # Find the minimum squared distance from any hand to any object, encouraging the closest approach.
    min_dist_sq = torch.min(torch.stack([dist_lh_o1_sq, dist_rh_o1_sq,
                                         dist_lh_o2_sq, dist_rh_o2_sq,
                                         dist_lh_o3_sq, dist_rh_o3_sq]), dim=0).values

    # Reward is inversely proportional to the minimum distance, encouraging closeness.
    # Add a small constant (0.05) for stability and continuous gradient.
    reward = 1.0 / (min_dist_sq + 0.05)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_stability_and_platform_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_avoidance_reward") -> torch.Tensor:
    """
    Shaping Reward 2: Encourages the robot to maintain a stable standing posture and to avoid
    unnecessary contact with the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    platform = env.scene['Object4']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Target pelvis height for stability (a common stable standing height)
    # This is a constant value, not a hard-coded position, representing a desired posture.
    target_pelvis_z = 0.7

    # Reward for maintaining pelvis height: negative absolute difference, encouraging closeness to target.
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for avoiding pelvis contact with the platform.
    # Hardcoded platform dimensions based on task description: 2m x 2m for x, y and 0.001m height.
    # The root_pos_w is the center of the platform.
    platform_half_x = 1.0 # Half width of platform (2m total width)
    platform_half_y = 1.0 # Half length of platform (2m total length)
    platform_height = 0.001 # From object configuration

    platform_x_min = platform.data.root_pos_w[:, 0] - platform_half_x
    platform_x_max = platform.data.root_pos_w[:, 0] + platform_half_x
    platform_y_min = platform.data.root_pos_w[:, 1] - platform_half_y
    platform_y_max = platform.data.root_pos_w[:, 1] + platform_half_y
    platform_z_top = platform.data.root_pos_w[:, 2] + platform_height

    # Condition: Pelvis is horizontally over the platform and too low (e.g., within 0.2m above platform top).
    # This encourages the robot to not stand on the platform itself.
    pelvis_over_platform_x = (pelvis_pos[:, 0] > platform_x_min) & (pelvis_pos[:, 0] < platform_x_max)
    pelvis_over_platform_y = (pelvis_pos[:, 1] > platform_y_min) & (pelvis_pos[:, 1] < platform_y_max)
    pelvis_too_low_over_platform = (pelvis_pos[:, 2] < platform_z_top + 0.2) # 0.2m clearance above platform top

    # Negative reward if pelvis is too close to or on the platform.
    # This is a conditional penalty, active only when the condition is met.
    pelvis_platform_collision_penalty = torch.where(
        pelvis_over_platform_x & pelvis_over_platform_y & pelvis_too_low_over_platform,
        -10.0, # Large negative reward for being on/too close to platform
        0.0
    )

    reward = pelvis_height_reward + pelvis_platform_collision_penalty

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def hands_pushing_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pushing_height_reward") -> torch.Tensor:
    """
    Shaping Reward 3: Encourages the robot to keep its hands at a suitable height for pushing the 0.5m cubed blocks.
    The target height is around the center of the cube (0.25m above its base).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access the required robot parts using approved patterns
    robot = env.scene["robot"]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link') # Corrected: Changed robot.body.names to robot.body_names
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # Hardcoded block dimensions: 0.5m cube. Center is 0.25m above its base.
    # Target hand height relative to the block's base for effective pushing.
    target_hand_z_offset = 0.25

    # Calculate target Z for each block based on its base position and the offset.
    # Using relative Z positions for the target height.
    target_z_o1 = object1.data.root_pos_w[:, 2] + target_hand_z_offset
    target_z_o2 = object2.data.root_pos_w[:, 2] + target_hand_z_offset
    target_z_o3 = object3.data.root_pos_w[:, 2] + target_hand_z_offset

    # Calculate absolute height difference for left hand to each object's target Z.
    lh_z_diff_o1 = torch.abs(left_hand_pos[:, 2] - target_z_o1)
    lh_z_diff_o2 = torch.abs(left_hand_pos[:, 2] - target_z_o2)
    lh_z_diff_o3 = torch.abs(left_hand_pos[:, 2] - target_z_o3)

    # Calculate absolute height difference for right hand to each object's target Z.
    rh_z_diff_o1 = torch.abs(right_hand_pos[:, 2] - target_z_o1)
    rh_z_diff_o2 = torch.abs(right_hand_pos[:, 2] - target_z_o2)
    rh_z_diff_o3 = torch.abs(right_hand_pos[:, 2] - target_z_o3)

    # Find the minimum height difference for any hand to any object.
    # This encourages at least one hand to be at the correct pushing height for any block.
    min_z_diff = torch.min(torch.stack([lh_z_diff_o1, lh_z_diff_o2, lh_z_diff_o3,
                                        rh_z_diff_o1, rh_z_diff_o2, rh_z_diff_o3]), dim=0).values

    # Reward is negative of the minimum absolute height difference, encouraging hands to be at the correct height.
    reward = -min_z_diff

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
    # Primary reward for moving blocks onto the platform
    MainBlockToPlatformReward = RewTerm(func=main_block_to_platform_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for hands proximity to any block
    HandsToBlockProximityReward = RewTerm(func=hands_to_block_proximity_reward, weight=0.6,
                                          params={"normalise": True, "normaliser_name": "hands_proximity_reward"})

    # Shaping reward for robot stability and avoiding standing on the platform
    RobotStabilityAndPlatformAvoidanceReward = RewTerm(func=robot_stability_and_platform_avoidance_reward, weight=0.4,
                                                       params={"normalise": True, "normaliser_name": "stability_avoidance_reward"})

    # Shaping reward for hands being at the correct pushing height relative to blocks
    HandsPushingHeightReward = RewTerm(func=hands_pushing_height_reward, weight=0.3,
                                       params={"normalise": True, "normaliser_name": "pushing_height_reward"})