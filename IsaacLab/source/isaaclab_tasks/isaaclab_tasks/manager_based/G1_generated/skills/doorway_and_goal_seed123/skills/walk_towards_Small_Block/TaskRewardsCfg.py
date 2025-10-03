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

def walk_towards_Small_Block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_towards_Small_Block_reward") -> torch.Tensor:
    """
    Primary reward for the robot to walk towards the Small Block (Object3).
    Encourages the robot's pelvis to be within a reasonable proximity to Object3,
    aligned in x, and at a stable height, ready for the final approach.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL OBJECT NAMING: Using 'Object3' as per the object configuration.
    object3 = env.scene['Object3'] # Small Block
    # Access object position using approved pattern
    object3_pos = object3.data.root_pos_w

    # Access the required robot part(s)
    # Using robot.body_names.index for approved access pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    # Access robot part position using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define target positions relative to Object3 and a stable height.
    # Target y-position: slightly in front of Object3's y-center.
    # The robot should stop before the block, so target y is Object3.y - (some_offset).
    # Using a relative distance for target_pelvis_y.
    target_y_offset = 0.7 # Distance from Object3's y-center to robot's pelvis target y
    target_pelvis_y = object3_pos[:, 1] - target_y_offset

    # Target x-position: aligned with Object3's x-center.
    # Using a relative distance for target_pelvis_x.
    target_pelvis_x = object3_pos[:, 0]

    # Target z-position: stable standing height.
    # This is an absolute height, which is allowed for Z.
    target_pelvis_z = 0.7 # Standard stable pelvis height

    # Calculate distances to target using relative distances.
    # Using torch.abs for continuous reward and handling batch environments.
    dist_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    dist_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for approaching target y. Negative exponential for continuous reward.
    # Reward is higher when dist_y is small.
    reward_y = -dist_y

    # Reward for aligning x-position.
    reward_x = -dist_x

    # Reward for maintaining stable pelvis height.
    reward_z = -dist_z

    # Combine rewards. Weights can be tuned.
    # Prioritize y movement, then x alignment, then z stability.
    reward = (reward_y * 0.5) + (reward_x * 0.3) + (reward_z * 0.2)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_walls_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_walls_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the walls (Object1 and Object2).
    Penalizes the robot if its pelvis or feet get too close to the walls.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL OBJECT NAMING: Using 'Object1' and 'Object2' as per the object configuration.
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    # Access object positions using approved pattern.
    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Access the required robot part(s)
    # Using robot.body_names.index for approved access pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    # Access robot part positions using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object dimensions (hardcoded from description, as per rules)
    wall_x_size = 0.5 # x of 0.5m for walls
    wall_y_size = 5.0 # y of 5m for walls

    # Define a safe clearance distance.
    # This is relative to the wall's half-size plus a buffer.
    clearance_x = wall_x_size / 2.0 + 0.3 # Half wall thickness + buffer
    clearance_y = wall_y_size / 2.0 + 0.3 # Half wall length + buffer

    # Calculate distances to Object1 (Wall 1) for pelvis and feet.
    # Using torch.abs for relative distances and handling batch environments.
    dist_pelvis_obj1_x = torch.abs(pelvis_pos[:, 0] - object1_pos[:, 0])
    dist_pelvis_obj1_y = torch.abs(pelvis_pos[:, 1] - object1_pos[:, 1])
    dist_lfoot_obj1_x = torch.abs(left_foot_pos[:, 0] - object1_pos[:, 0])
    dist_lfoot_obj1_y = torch.abs(left_foot_pos[:, 1] - object1_pos[:, 1])
    dist_rfoot_obj1_x = torch.abs(right_foot_pos[:, 0] - object1_pos[:, 0])
    dist_rfoot_obj1_y = torch.abs(right_foot_pos[:, 1] - object1_pos[:, 1])

    # Calculate distances to Object2 (Wall 2) for pelvis and feet.
    dist_pelvis_obj2_x = torch.abs(pelvis_pos[:, 0] - object2_pos[:, 0])
    dist_pelvis_obj2_y = torch.abs(pelvis_pos[:, 1] - object2_pos[:, 1])
    dist_lfoot_obj2_x = torch.abs(left_foot_pos[:, 0] - object2_pos[:, 0])
    dist_lfoot_obj2_y = torch.abs(left_foot_pos[:, 1] - object2_pos[:, 1])
    dist_rfoot_obj2_x = torch.abs(right_foot_pos[:, 0] - object2_pos[:, 0])
    dist_rfoot_obj2_y = torch.abs(right_foot_pos[:, 1] - object2_pos[:, 1])

    # Penalize if any part is too close to Object1 or Object2.
    # Reward is 0 if outside clearance, negative inverse distance if inside.
    # Using torch.where for conditional reward and adding a small epsilon (0.01) to avoid division by zero.
    reward_collision_obj1 = torch.where(
        (dist_pelvis_obj1_x < clearance_x) & (dist_pelvis_obj1_y < clearance_y),
        -1.0 / (dist_pelvis_obj1_x + dist_pelvis_obj1_y + 0.01), # Inverse distance penalty
        torch.tensor(0.0, device=env.device)
    ) + torch.where(
        (dist_lfoot_obj1_x < clearance_x) & (dist_lfoot_obj1_y < clearance_y),
        -1.0 / (dist_lfoot_obj1_x + dist_lfoot_obj1_y + 0.01),
        torch.tensor(0.0, device=env.device)
    ) + torch.where(
        (dist_rfoot_obj1_x < clearance_x) & (dist_rfoot_obj1_y < clearance_y),
        -1.0 / (dist_rfoot_obj1_x + dist_rfoot_obj1_y + 0.01),
        torch.tensor(0.0, device=env.device)
    )

    reward_collision_obj2 = torch.where(
        (dist_pelvis_obj2_x < clearance_x) & (dist_pelvis_obj2_y < clearance_y),
        -1.0 / (dist_pelvis_obj2_x + dist_pelvis_obj2_y + 0.01),
        torch.tensor(0.0, device=env.device)
    ) + torch.where(
        (dist_lfoot_obj2_x < clearance_x) & (dist_lfoot_obj2_y < clearance_y),
        -1.0 / (dist_lfoot_obj2_x + dist_lfoot_obj2_y + 0.01),
        torch.tensor(0.0, device=env.device)
    ) + torch.where(
        (dist_rfoot_obj2_x < clearance_x) & (dist_rfoot_obj2_y < clearance_y),
        -1.0 / (dist_rfoot_obj2_x + dist_rfoot_obj2_y + 0.01),
        torch.tensor(0.0, device=env.device)
    )

    reward = reward_collision_obj1 + reward_collision_obj2

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_small_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_small_block_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the Small Block (Object3).
    Penalizes the robot if its pelvis or feet get too close to or penetrate Object3.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # CRITICAL OBJECT NAMING: Using 'Object3' as per the object configuration.
    object3 = env.scene['Object3'] # Small Block
    # Access object position using approved pattern.
    object3_pos = object3.data.root_pos_w

    # Access the required robot part(s)
    # Using robot.body_names.index for approved access pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    # Access robot part positions using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object3 dimensions (hardcoded from description, as per rules)
    object3_size_x = 0.3
    object3_size_y = 0.3
    object3_size_z = 0.3

    # Define a safe clearance distance.
    # This is relative to the block's half-size plus a buffer.
    clearance_x_obj3 = object3_size_x / 2.0 + 0.1
    clearance_y_obj3 = object3_size_y / 2.0 + 0.1
    clearance_z_obj3 = object3_size_z / 2.0 + 0.1

    # Calculate distances to Object3 for pelvis and feet.
    # Using torch.abs for relative distances and handling batch environments.
    dist_pelvis_obj3_x = torch.abs(pelvis_pos[:, 0] - object3_pos[:, 0])
    dist_pelvis_obj3_y = torch.abs(pelvis_pos[:, 1] - object3_pos[:, 1])
    dist_pelvis_obj3_z = torch.abs(pelvis_pos[:, 2] - object3_pos[:, 2])

    dist_lfoot_obj3_x = torch.abs(left_foot_pos[:, 0] - object3_pos[:, 0])
    dist_lfoot_obj3_y = torch.abs(left_foot_pos[:, 1] - object3_pos[:, 1])
    dist_lfoot_obj3_z = torch.abs(left_foot_pos[:, 2] - object3_pos[:, 2])

    dist_rfoot_obj3_x = torch.abs(right_foot_pos[:, 0] - object3_pos[:, 0])
    dist_rfoot_obj3_y = torch.abs(right_foot_pos[:, 1] - object3_pos[:, 1])
    dist_rfoot_obj3_z = torch.abs(right_foot_pos[:, 2] - object3_pos[:, 2])

    # Penalize if any part is too close to Object3.
    # Stronger negative inverse distance penalty if inside clearance.
    # Using torch.where for conditional reward and adding a small epsilon (0.01) to avoid division by zero.
    reward = torch.where(
        (dist_pelvis_obj3_x < clearance_x_obj3) & (dist_pelvis_obj3_y < clearance_y_obj3) & (dist_pelvis_obj3_z < clearance_z_obj3),
        -2.0 / (dist_pelvis_obj3_x + dist_pelvis_obj3_y + dist_pelvis_obj3_z + 0.01), # Stronger penalty
        torch.tensor(0.0, device=env.device)
    ) + torch.where(
        (dist_lfoot_obj3_x < clearance_x_obj3) & (dist_lfoot_obj3_y < clearance_y_obj3) & (dist_lfoot_obj3_z < clearance_z_obj3),
        -2.0 / (dist_lfoot_obj3_x + dist_lfoot_obj3_y + dist_lfoot_obj3_z + 0.01),
        torch.tensor(0.0, device=env.device)
    ) + torch.where(
        (dist_rfoot_obj3_x < clearance_x_obj3) & (dist_rfoot_obj3_y < clearance_y_obj3) & (dist_rfoot_obj3_z < clearance_z_obj3),
        -2.0 / (dist_rfoot_obj3_x + dist_rfoot_obj3_y + dist_rfoot_obj3_z + 0.01),
        torch.tensor(0.0, device=env.device)
    )

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
    # Primary reward for walking towards the small block.
    # Weight 1.0 as it's the main objective.
    WalkTowardsSmallBlockReward = RewTerm(func=walk_towards_Small_Block_reward, weight=1.0,
                                          params={"normalise": True, "normaliser_name": "walk_towards_Small_Block_reward"})

    # Shaping reward for avoiding collisions with the walls.
    # Weight 0.6 to encourage avoiding obstacles but not overpower the main goal.
    CollisionAvoidanceWallsReward = RewTerm(func=collision_avoidance_walls_reward, weight=0.6,
                                            params={"normalise": True, "normaliser_name": "collision_avoidance_walls_reward"})

    # Shaping reward for avoiding collisions with the small block itself.
    # Weight 0.8 to strongly discourage hitting the target block.
    CollisionAvoidanceSmallBlockReward = RewTerm(func=collision_avoidance_small_block_reward, weight=0.8,
                                                 params={"normalise": True, "normaliser_name": "collision_avoidance_small_block_reward"})