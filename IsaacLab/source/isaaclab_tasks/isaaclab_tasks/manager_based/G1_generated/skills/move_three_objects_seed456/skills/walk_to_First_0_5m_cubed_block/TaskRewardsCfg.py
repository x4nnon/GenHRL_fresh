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


def walk_to_first_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_first_block_main_reward") -> torch.Tensor:
    """
    Main reward for the 'walk_to_First_0_5m_cubed_block' skill.
    Encourages the robot to walk to a position adjacent to 'Object1' (First 0.5m cubed block),
    preparing to push it. The goal is to be within pushing distance.
    Also rewards maintaining a stable pelvis height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot and target object using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1'] # Accessing Object1 directly as per requirements

    # Access the required robot part position (pelvis) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Access the required object position using approved patterns
    object1_pos = object1.data.root_pos_w # Accessing object position using approved pattern

    # Object1 dimensions (0.5m cubed block) - hardcoded from object configuration as per requirements
    block_half_size = 0.5 / 2.0 # 0.25m

    # Calculate horizontal distance from pelvis to Object1's center using relative distances
    # Using relative distances between object and robot part positions as required
    distance_x_pelvis_obj1 = object1_pos[:, 0] - pelvis_pos[:, 0]
    distance_y_pelvis_obj1 = object1_pos[:, 1] - pelvis_pos[:, 1]
    horizontal_distance_pelvis_obj1 = torch.sqrt(distance_x_pelvis_obj1**2 + distance_y_pelvis_obj1**2)

    # Target pushing distance from block center to pelvis center.
    # This should be slightly more than half the block size to avoid collision,
    # allowing the robot to be "adjacent" and ready to push.
    # Example: block_half_size + 0.05m (for robot's "pushing point" clearance)
    target_push_distance = block_half_size + 0.05 # e.g., 0.25 + 0.05 = 0.3m

    # Reward for being at the target pushing distance
    # Using a negative absolute difference to create a peak at target_push_distance, ensuring continuity
    reward_distance_to_obj1 = -torch.abs(horizontal_distance_pelvis_obj1 - target_push_distance)

    # Reward for maintaining a stable pelvis height (e.g., 0.7m)
    # Using relative distance from ground (0.0) to pelvis Z-position, Z is the only absolute position allowed sparingly
    target_pelvis_z = 0.7
    reward_pelvis_height = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards with weights
    # Main distance reward is weighted higher, pelvis height is a shaping reward
    reward = reward_distance_to_obj1 + 0.1 * reward_pelvis_height

    # Mandatory reward normalization as per requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def feet_on_ground_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_on_ground_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to keep its feet on the ground, promoting stability.
    It penalizes feet being too high off the ground.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot using approved patterns
    robot = env.scene["robot"]

    # Access the required robot part positions (feet) using approved patterns
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2] # Accessing robot part position using approved pattern
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2] # Accessing robot part position using approved pattern

    # Ground level is typically 0.0 in Isaac Lab
    ground_level = 0.0

    # Define a small tolerance for foot lift, allowing for walking steps
    # This is a hardcoded threshold as per requirements for specific measurements
    max_foot_lift_tolerance = 0.05 # 0.05m tolerance

    # Penalize if foot Z-position is above the ground level plus tolerance
    # Using torch.max to ensure penalty is only applied when foot is too high, and is continuous
    # Z-position relative to ground is allowed sparingly
    reward_left_foot_on_ground = -torch.max(torch.tensor(0.0, device=env.device), left_foot_pos_z - ground_level - max_foot_lift_tolerance)
    reward_right_foot_on_ground = -torch.max(torch.tensor(0.0, device=env.device), right_foot_pos_z - ground_level - max_foot_lift_tolerance)

    # Combine rewards for both feet
    reward = reward_left_foot_on_ground + reward_right_foot_on_ground

    # Mandatory reward normalization as per requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to encourage collision avoidance with non-target objects (Object2, Object3, Object4).
    It penalizes the robot's pelvis getting too close to these objects.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot and other objects using approved patterns
    robot = env.scene["robot"]
    object2 = env.scene['Object2'] # Accessing Object2 directly
    object3 = env.scene['Object3'] # Accessing Object3 directly
    object4 = env.scene['Object4'] # Accessing Object4 directly

    # Access the required robot part position (pelvis) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Access positions of other objects using approved patterns
    object2_pos = object2.data.root_pos_w # Accessing object position using approved pattern
    object3_pos = object3.data.root_pos_w # Accessing object position using approved pattern
    object4_pos = object4.data.root_pos_w # Accessing object position using approved pattern

    # Define a safe distance threshold for collision avoidance
    # This is a hardcoded threshold as per requirements for specific measurements
    safe_distance_threshold = 0.3 # 0.3m clearance

    # Calculate horizontal distances from pelvis to each non-target object
    # Using relative distances between object and robot part positions as required
    dist_x_pelvis_obj2 = object2_pos[:, 0] - pelvis_pos[:, 0]
    dist_y_pelvis_obj2 = object2_pos[:, 1] - pelvis_pos[:, 1]
    horizontal_dist_pelvis_obj2 = torch.sqrt(dist_x_pelvis_obj2**2 + dist_y_pelvis_obj2**2)

    dist_x_pelvis_obj3 = object3_pos[:, 0] - pelvis_pos[:, 0]
    dist_y_pelvis_obj3 = object3_pos[:, 1] - pelvis_pos[:, 1]
    horizontal_dist_pelvis_obj3 = torch.sqrt(dist_x_pelvis_obj3**2 + dist_y_pelvis_obj3**2)

    dist_x_pelvis_obj4 = object4_pos[:, 0] - pelvis_pos[:, 0]
    dist_y_pelvis_obj4 = object4_pos[:, 1] - pelvis_pos[:, 1]
    horizontal_dist_pelvis_obj4 = torch.sqrt(dist_x_pelvis_obj4**2 + dist_y_pelvis_obj4**2)

    # Penalize if horizontal distance is less than the safe threshold
    # The penalty increases linearly as distance decreases below the threshold, ensuring continuity
    # Using torch.where to ensure continuity and apply penalty only when needed
    reward_obj2_avoidance = torch.where(horizontal_dist_pelvis_obj2 < safe_distance_threshold,
                                        -(safe_distance_threshold - horizontal_dist_pelvis_obj2),
                                        torch.tensor(0.0, device=env.device))
    reward_obj3_avoidance = torch.where(horizontal_dist_pelvis_obj3 < safe_distance_threshold,
                                        -(safe_distance_threshold - horizontal_dist_pelvis_obj3),
                                        torch.tensor(0.0, device=env.device))
    reward_obj4_avoidance = torch.where(horizontal_dist_pelvis_obj4 < safe_distance_threshold,
                                        -(safe_distance_threshold - horizontal_dist_pelvis_obj4),
                                        torch.tensor(0.0, device=env.device))

    # Combine avoidance rewards for all non-target objects
    reward = reward_obj2_avoidance + reward_obj3_avoidance + reward_obj4_avoidance

    # Mandatory reward normalization as per requirements
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
    Configuration for the reward terms for the 'walk_to_First_0_5m_cubed_block' skill.
    """
    # Main reward for reaching the target block with weight 1.0 as per requirements
    WalkToFirstBlockMainReward = RewTerm(func=walk_to_first_block_main_reward, weight=1.0,
                                         params={"normalise": True, "normaliser_name": "walk_to_first_block_main_reward"})

    # Shaping reward for keeping feet on the ground with lower weight as per requirements
    FeetOnGroundReward = RewTerm(func=feet_on_ground_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "feet_on_ground_reward"})

    # Shaping reward for avoiding collisions with other objects with lower weight as per requirements
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.2, # Lower weight as it's a general avoidance
                                      params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})