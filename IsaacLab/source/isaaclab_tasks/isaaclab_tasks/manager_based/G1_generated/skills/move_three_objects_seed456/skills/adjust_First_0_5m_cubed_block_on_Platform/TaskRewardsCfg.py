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

def main_adjust_block_on_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for ensuring the First 0.5m cubed block (Object1) is fully and stably positioned
    within the boundaries of the Platform (Object4).
    Rewards for correct x, y, and z positioning of Object1 relative to Object4.
    """
    # REQUIREMENT: MANDATORY REWARD NORMALIZATION - Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # REQUIREMENT: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1'] # First 0.5m cubed block
    object4 = env.scene['Object4'] # Platform

    # REQUIREMENT: THERE IS NO way to access the SIZE of an object. Hardcode values from config.
    # Object dimensions (from skill info: Object1 is 0.5m cubed, Object4 is 2m x 2m x 0.001m)
    block_half_size = 0.25 # 0.5m / 2
    platform_half_x_size = 1.0 # 2m / 2
    platform_half_y_size = 1.0 # 2m / 2
    platform_z_pos = 0.001 # Platform z-position (assuming its root_pos_w[:, 2] is its bottom)

    # Object1 positions
    object1_pos = object1.data.root_pos_w
    object1_pos_x = object1_pos[:, 0]
    object1_pos_y = object1_pos[:, 1]
    object1_pos_z = object1_pos[:, 2]

    # Object4 (Platform) positions - assuming its center is at its root_pos_w
    platform_center = object4.data.root_pos_w
    platform_center_x = platform_center[:, 0]
    platform_center_y = platform_center[:, 1]

    # REQUIREMENT: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Target Z: Object1's center should be at platform_z_pos + block_half_size
    target_object1_z = platform_z_pos + block_half_size
    # REQUIREMENT: Rewards should be continuous. Negative absolute difference for z-alignment.
    reward_z = -torch.abs(object1_pos_z - target_object1_z)

    # Target X boundaries for Object1's center on the platform
    # Object1's center must be within [platform_center_x - (platform_half_x_size - block_half_size), platform_center_x + (platform_half_x_size - block_half_size)]
    x_min_bound = platform_center_x - (platform_half_x_size - block_half_size)
    x_max_bound = platform_center_x + (platform_half_x_size - block_half_size)
    # REQUIREMENT: Rewards should be continuous. Penalize if outside bounds using torch.max(0, value).
    # REQUIREMENT: All operations must work with batched environments.
    reward_x = -torch.max(torch.zeros_like(object1_pos_x, device=env.device), object1_pos_x - x_max_bound) - \
               torch.max(torch.zeros_like(object1_pos_x, device=env.device), x_min_bound - object1_pos_x)

    # Target Y boundaries for Object1's center on the platform
    # Object1's center must be within [platform_center_y - (platform_half_y_size - block_half_size), platform_center_y + (platform_half_y_size - block_half_size)]
    y_min_bound = platform_center_y - (platform_half_y_size - block_half_size)
    y_max_bound = platform_center_y + (platform_half_y_size - block_half_size)
    # REQUIREMENT: Rewards should be continuous. Penalize if outside bounds using torch.max(0, value).
    # REQUIREMENT: All operations must work with batched environments.
    reward_y = -torch.max(torch.zeros_like(object1_pos_y, device=env.device), object1_pos_y - y_max_bound) - \
               torch.max(torch.zeros_like(object1_pos_y, device=env.device), y_min_bound - object1_pos_y)

    # Combine rewards
    reward = reward_x + reward_y + reward_z

    # REQUIREMENT: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_and_push_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_push_reward") -> torch.Tensor:
    """
    This reward encourages the robot to approach Object1 and push it towards Object4.
    It rewards reducing the distance between the robot's hand and Object1, and
    simultaneously reducing the distance between Object1 and Object4.
    """
    # REQUIREMENT: MANDATORY REWARD NORMALIZATION - Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # REQUIREMENT: Access objects directly - objects should always exist in the scene
    object1 = env.scene['Object1'] # First 0.5m cubed block
    object4 = env.scene['Object4'] # Platform
    robot = env.scene["robot"]

    # REQUIREMENT: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot_hand_idx = robot.body_names.index('right_palm_link') # Using right hand for pushing
    robot_hand_pos = robot.data.body_pos_w[:, robot_hand_idx]
    robot_hand_pos_x = robot_hand_pos[:, 0]
    robot_hand_pos_y = robot_hand_pos[:, 1]
    robot_hand_pos_z = robot_hand_pos[:, 2]

    # Object1 positions
    object1_pos = object1.data.root_pos_w
    object1_pos_x = object1_pos[:, 0]
    object1_pos_y = object1_pos[:, 1]
    object1_pos_z = object1_pos[:, 2]

    # Object4 (Platform) positions
    platform_center = object4.data.root_pos_w
    platform_center_x = platform_center[:, 0]
    platform_center_y = platform_center[:, 1]

    # REQUIREMENT: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Distance from robot hand to Object1
    dist_hand_to_object1_x = torch.abs(object1_pos_x - robot_hand_pos_x)
    dist_hand_to_object1_y = torch.abs(object1_pos_y - robot_hand_pos_y)
    dist_hand_to_object1_z = torch.abs(object1_pos_z - robot_hand_pos_z)

    # REQUIREMENT: Rewards should be continuous. Reward for getting close to Object1 (negative distance, so smaller distance is higher reward)
    reward_approach_object1 = - (dist_hand_to_object1_x + dist_hand_to_object1_y + dist_hand_to_object1_z)

    # Distance from Object1 to the center of Object4 (in x,y plane)
    dist_object1_to_platform_x = torch.abs(object1_pos_x - platform_center_x)
    dist_object1_to_platform_y = torch.abs(object1_pos_y - platform_center_y)

    # Reward for pushing Object1 towards Object4 (reducing distance)
    # This reward is active when Object1 is not yet fully on the platform
    # REQUIREMENT: THERE IS NO way to access the SIZE of an object. Hardcode values from config.
    block_half_size = 0.25 # 0.5m / 2
    platform_half_x_size = 1.0 # 2m / 2
    x_min_bound = platform_center_x - (platform_half_x_size - block_half_size)
    x_max_bound = platform_center_x + (platform_half_x_size - block_half_size)

    # REQUIREMENT: All operations must work with batched environments
    # Condition: Object1 is not yet within the target x-bounds of the platform
    condition_not_on_platform_x = (object1_pos_x < x_min_bound) | (object1_pos_x > x_max_bound)

    # Apply push reward only if Object1 is outside the target x-bounds
    # REQUIREMENT: Rewards should be continuous.
    reward_push_object1_to_platform = - (dist_object1_to_platform_x + dist_object1_to_platform_y)
    reward_push_object1_to_platform = torch.where(condition_not_on_platform_x, reward_push_object1_to_platform, torch.tensor(0.0, device=env.device))

    # Combine approach and push rewards
    reward = reward_approach_object1 + reward_push_object1_to_platform

    # REQUIREMENT: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stability_and_overshoot_penalty_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_overshoot_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain a stable, upright posture (pelvis at a reasonable height)
    throughout the task and penalizes it for pushing Object1 too far, off the other side of the platform.
    """
    # REQUIREMENT: MANDATORY REWARD NORMALIZATION - Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # REQUIREMENT: Access objects directly - objects should always exist in the scene
    object1 = env.scene['Object1'] # First 0.5m cubed block
    object4 = env.scene['Object4'] # Platform
    robot = env.scene["robot"]

    # REQUIREMENT: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # Object1 positions
    object1_pos_x = object1.data.root_pos_w[:, 0]

    # REQUIREMENT: THERE IS NO way to access the SIZE of an object. Hardcode values from config.
    # Object4 (Platform) dimensions
    platform_half_x_size = 1.0 # 2m / 2
    platform_center_x = object4.data.root_pos_w[:, 0]

    # REQUIREMENT: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # For pelvis height, it's relative to the ground (z=0), so an absolute z-position is acceptable here.
    target_pelvis_z = 0.7 # A reasonable height for a bipedal robot's pelvis

    # REQUIREMENT: Rewards should be continuous. Reward for maintaining pelvis height (negative absolute difference)
    reward_pelvis_height = -torch.abs(robot_pelvis_pos_z - target_pelvis_z)

    # Penalty for pushing Object1 off the far side of the platform
    # This assumes the robot pushes from lower x to higher x towards the platform
    # The "far side" would be beyond platform_center_x + platform_half_x_size
    # A small buffer can be added to the boundary to allow for slight overshoots before penalty
    overshoot_buffer = 0.1 # meters
    penalty_boundary_x = platform_center_x + platform_half_x_size + overshoot_buffer

    # REQUIREMENT: All operations must work with batched environments
    # Condition: Object1 has been pushed past the far edge of the platform
    condition_overshoot = (object1_pos_x > penalty_boundary_x)

    # REQUIREMENT: Rewards should be continuous. Apply a negative reward (penalty) if overshoot occurs
    # The penalty scales with how far past the boundary the object is
    reward_overshoot_penalty = torch.where(condition_overshoot, -10.0 * torch.abs(object1_pos_x - penalty_boundary_x), torch.tensor(0.0, device=env.device))

    # Combine rewards
    reward = reward_pelvis_height + reward_overshoot_penalty

    # REQUIREMENT: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # REQUIREMENT: Main reward with weight ~1.0
    main_adjust_block_on_platform_reward = RewTerm(func=main_adjust_block_on_platform_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    # REQUIREMENT: Supporting rewards with lower weights (e.g., 0.1-0.5)
    approach_and_push_reward = RewTerm(func=approach_and_push_reward, weight=0.6,
                              params={"normalise": True, "normaliser_name": "approach_push_reward"})

    stability_and_overshoot_penalty_reward = RewTerm(func=stability_and_overshoot_penalty_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "stability_overshoot_reward"})