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


def main_push_object2_onto_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for ensuring Object2 is fully and stably positioned on Object4.
    This reward combines horizontal centering, correct Z-position, and being within horizontal bounds.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object2 = env.scene['Object2']
    object4 = env.scene['Object4']

    # Hardcode object dimensions from the task description/object configuration.
    # Object2 (Cube for robot to push): 0.5m x 0.5m x 0.5m
    object2_half_size_x = 0.5 / 2.0
    object2_half_size_y = 0.5 / 2.0
    object2_half_size_z = 0.5 / 2.0
    # Object4 (Platform for cubes): 2m x 2m x 0.001m
    object4_half_size_x = 2.0 / 2.0
    object4_half_size_y = 2.0 / 2.0
    object4_height = 0.001

    # Calculate target z-position for Object2 on Object4.
    # This is a relative calculation based on object positions and hardcoded dimensions.
    target_object2_z = object4.data.root_pos_w[:, 2] + object4_height / 2.0 + object2_half_size_z

    # Calculate horizontal distances between Object2's center and Object4's center.
    # These are relative distances.
    dist_obj2_obj4_x = object2.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    dist_obj2_obj4_y = object2.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]

    # Reward for Object2 being horizontally centered on Object4.
    # Uses absolute distances to penalize deviation from center, ensuring continuity.
    reward_horizontal_center = -torch.abs(dist_obj2_obj4_x) - torch.abs(dist_obj2_obj4_y)

    # Reward for Object2 being at the correct height on Object4.
    # Penalizes deviation from the target Z-position, ensuring continuity.
    reward_z_pos = -torch.abs(object2.data.root_pos_w[:, 2] - target_object2_z)

    # Calculate the horizontal boundaries of Object4 where Object2's center should be.
    # These are relative to Object4's position and account for Object2's size.
    obj4_min_x = object4.data.root_pos_w[:, 0] - object4_half_size_x + object2_half_size_x
    obj4_max_x = object4.data.root_pos_w[:, 0] + object4_half_size_x - object2_half_size_x
    obj4_min_y = object4.data.root_pos_w[:, 1] - object4_half_size_y + object2_half_size_y
    obj4_max_y = object4.data.root_pos_w[:, 1] + object4_half_size_y - object2_half_size_y

    # Check if Object2's center is within Object4's bounds.
    is_within_x = (object2.data.root_pos_w[:, 0] > obj4_min_x) & (object2.data.root_pos_w[:, 0] < obj4_max_x)
    is_within_y = (object2.data.root_pos_w[:, 1] > obj4_min_y) & (object2.data.root_pos_w[:, 1] < obj4_max_y)

    # Continuous reward for being within bounds: penalize distance from nearest edge if outside.
    # This ensures a smooth reward that encourages moving towards the bounds.
    reward_within_bounds_x = torch.where(is_within_x, torch.tensor(0.0, device=env.device),
                                         -torch.min(torch.abs(object2.data.root_pos_w[:, 0] - obj4_min_x),
                                                    torch.abs(object2.data.root_pos_w[:, 0] - obj4_max_x)))
    reward_within_bounds_y = torch.where(is_within_y, torch.tensor(0.0, device=env.device),
                                         -torch.min(torch.abs(object2.data.root_pos_w[:, 1] - obj4_min_y),
                                                    torch.abs(object2.data.root_pos_w[:, 1] - obj4_max_y)))

    # Combine rewards with appropriate weights.
    reward = reward_horizontal_center * 0.5 + reward_z_pos * 0.5 + reward_within_bounds_x * 0.2 + reward_within_bounds_y * 0.2

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_hand_proximity_to_object2_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_hand_proximity") -> torch.Tensor:
    """
    Shaping reward to encourage the robot's right hand to be close to Object2 for pushing.
    This reward is active when Object2 is not yet fully on Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot part using approved patterns
    object2 = env.scene['Object2']
    object4 = env.scene['Object4']
    robot = env.scene["robot"]
    robot_hand_idx = robot.body_names.index('right_palm_link')
    robot_hand_pos = robot.data.body_pos_w[:, robot_hand_idx]

    # Hardcode object dimensions for activation condition.
    object2_half_size_x = 0.5 / 2.0
    object2_half_size_y = 0.5 / 2.0
    object2_half_size_z = 0.5 / 2.0
    object4_half_size_x = 2.0 / 2.0
    object4_half_size_y = 2.0 / 2.0
    object4_height = 0.001

    # Calculate target z-position for Object2 on Object4 for activation check.
    target_object2_z = object4.data.root_pos_w[:, 2] + object4_height / 2.0 + object2_half_size_z

    # Check if Object2 is approximately on the platform (for activation condition).
    # These checks use relative distances and hardcoded dimensions with a tolerance.
    is_obj2_on_platform_z = torch.abs(object2.data.root_pos_w[:, 2] - target_object2_z) < 0.1
    is_obj2_on_platform_x = torch.abs(object2.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]) < object4_half_size_x - object2_half_size_x + 0.1
    is_obj2_on_platform_y = torch.abs(object2.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]) < object4_half_size_y - object2_half_size_y + 0.1

    # Activation condition: Reward is active if Object2 is NOT yet fully on Object4.
    # This ensures the robot keeps pushing until the primary reward takes over fully.
    activation_condition = ~(is_obj2_on_platform_z & is_obj2_on_platform_x & is_obj2_on_platform_y)

    # Calculate distances between robot's hand and Object2.
    # These are relative distances.
    dist_hand_obj2_x = object2.data.root_pos_w[:, 0] - robot_hand_pos[:, 0]
    dist_hand_obj2_y = object2.data.root_pos_w[:, 1] - robot_hand_pos[:, 1]
    dist_hand_obj2_z = object2.data.root_pos_w[:, 2] - robot_hand_pos[:, 2]

    # Reward for hand proximity to Object2.
    # Penalizes the absolute distance, encouraging the hand to be close.
    reward_hand_proximity = -torch.abs(dist_hand_obj2_x) - torch.abs(dist_hand_obj2_y) - torch.abs(dist_hand_obj2_z)

    # Apply activation condition: reward is 0 if not active.
    shaping_reward = torch.where(activation_condition, reward_hand_proximity, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward)
        RewNormalizer.update_stats(normaliser_name, shaping_reward)
        return scaled_reward
    return shaping_reward


def shaping_posture_and_hand_retraction_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_posture_hand_retraction") -> torch.Tensor:
    """
    Shaping reward for maintaining a stable, upright posture (pelvis height) and
    encouraging the robot's hand to retract to a ready state after Object2 is on the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object2 = env.scene['Object2']
    object4 = env.scene['Object4']
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_hand_idx = robot.body_names.index('right_palm_link')
    robot_hand_pos = robot.data.body_pos_w[:, robot_hand_idx]

    # Target pelvis height (hardcoded as it's a robot-centric posture goal).
    target_pelvis_z = 0.7

    # Hardcode object dimensions for activation condition.
    object2_half_size_x = 0.5 / 2.0
    object2_half_size_y = 0.5 / 2.0
    object2_half_size_z = 0.5 / 2.0
    object4_half_size_x = 2.0 / 2.0
    object4_half_size_y = 2.0 / 2.0
    object4_height = 0.001

    # Calculate target z-position for Object2 on Object4 for activation check.
    target_object2_z = object4.data.root_pos_w[:, 2] + object4_height / 2.0 + object2_half_size_z

    # Check if Object2 is approximately on the platform (for activation condition).
    # Tighter tolerance for final state check.
    is_obj2_on_platform_z = torch.abs(object2.data.root_pos_w[:, 2] - target_object2_z) < 0.05
    is_obj2_on_platform_x = torch.abs(object2.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]) < object4_half_size_x - object2_half_size_x + 0.05
    is_obj2_on_platform_y = torch.abs(object2.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]) < object4_half_size_y - object2_half_size_y + 0.05

    # Reward for pelvis height. Penalizes deviation from target Z-height.
    reward_pelvis_height = -torch.abs(robot_pelvis_pos[:, 2] - target_pelvis_z)

    # Calculate distance between robot's hand and Object2.
    # These are relative distances.
    dist_hand_obj2_x = object2.data.root_pos_w[:, 0] - robot_hand_pos[:, 0]
    dist_hand_obj2_y = object2.data.root_pos_w[:, 1] - robot_hand_pos[:, 1]
    dist_hand_obj2_z = object2.data.root_pos_w[:, 2] - robot_hand_pos[:, 2]
    hand_distance_from_obj2 = torch.sqrt(dist_hand_obj2_x**2 + dist_hand_obj2_y**2 + dist_hand_obj2_z**2)

    # Activation condition for hand retraction: only active when Object2 is on platform
    activation_condition_hand_retract = (is_obj2_on_platform_z & is_obj2_on_platform_x & is_obj2_on_platform_y)

    # Reward for hand retraction: penalize if too close (still pushing) or too far (not ready).
    # This creates a continuous reward that is 0 within the target range and negative outside.
    target_hand_retract_min = 0.2
    target_hand_retract_max = 0.5
    reward_hand_retract = torch.where(hand_distance_from_obj2 < target_hand_retract_min,
                                      -(target_hand_retract_min - hand_distance_from_obj2), # Penalize if too close
                                      torch.where(hand_distance_from_obj2 > target_hand_retract_max,
                                                  -(hand_distance_from_obj2 - target_hand_retract_max), # Penalize if too far
                                                  torch.tensor(0.0, device=env.device))) # Reward if in range

    # Combine rewards. Hand retraction reward is only applied when active.
    shaping_reward = reward_pelvis_height + torch.where(activation_condition_hand_retract, reward_hand_retract, torch.tensor(0.0, device=env.device))

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
    # Main reward for positioning Object2 on Object4. Weight 1.0 as it's the primary goal.
    MainPushObject2OntoPlatformReward = RewTerm(func=main_push_object2_onto_platform_reward, weight=1.0,
                                                params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for hand proximity to Object2. Weight 0.4 to guide initial pushing.
    ShapingHandProximityToObject2Reward = RewTerm(func=shaping_hand_proximity_to_object2_reward, weight=0.4,
                                                  params={"normalise": True, "normaliser_name": "shaping_hand_proximity"})

    # Shaping reward for posture and hand retraction. Weight 0.2 for general stability and post-skill readiness.
    ShapingPostureAndHandRetractionReward = RewTerm(func=shaping_posture_and_hand_retraction_reward, weight=0.2,
                                                    params={"normalise": True, "normaliser_name": "shaping_posture_hand_retraction"})