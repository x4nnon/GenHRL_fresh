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


def walk_to_second_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_second_block_main_reward") -> torch.Tensor:
    """
    Main reward for the robot to walk to a position adjacent to the Second 0.5m cubed block,
    preparing to push it. The goal is for the robot's pelvis to be at a specific pushing distance
    and alignment relative to the target block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # CRITICAL OBJECT NAMING: Object2 is the "Second 0.5m cubed block"
    target_block = env.scene['Object2'] # Accessing object using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    target_block_pos = target_block.data.root_pos_w # Accessing object position using approved pattern

    # Object2 dimensions: 0.5m cubed block. Half-size is 0.25m.
    # Target distance from block center to robot pelvis for pushing.
    # We want the robot pelvis to be ~0.1m from the face of the block.
    # So, 0.25m (block half-size) + 0.1m (clearance) = 0.35m from block center.
    target_distance_from_block_center = 0.35 # meters (hardcoded from object config analysis)
    target_pelvis_z = 0.7 # meters (stable standing height for pelvis)

    # Calculate relative position of pelvis to target_block's center
    # All rewards MUST ONLY use relative distances between objects and robot parts
    relative_pos_x = pelvis_pos[:, 0] - target_block_pos[:, 0]
    relative_pos_y = pelvis_pos[:, 1] - target_block_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2] # Z is relative to ground (0,0,0) which is an allowed absolute reference for height

    # Reward for X distance: Encourage pelvis to be at -target_distance_from_block_center relative to block's X.
    # This assumes the robot approaches from the negative X side of the block.
    # Using negative absolute difference for continuous reward, closer to 0 is better.
    reward_x = -torch.abs(relative_pos_x + target_distance_from_block_center)

    # Reward for Y distance: Encourage pelvis to be aligned with block's Y center.
    # Using negative absolute difference, closer to 0 is better.
    reward_y = -torch.abs(relative_pos_y)

    # Reward for Z height: Encourage pelvis to be at a stable standing height.
    # Using negative absolute difference, closer to 0 is better.
    reward_z = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Combine rewards
    reward = reward_x + reward_y + reward_z

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_other_blocks_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_other_blocks_reward") -> torch.Tensor:
    """
    Penalizes the robot for getting too close to Object1 and Object3 (other blocks),
    ensuring it navigates around them safely.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # CRITICAL OBJECT NAMING: Object1 is "First 0.5m cubed block", Object3 is "Third 0.5m cubed block"
    block1 = env.scene['Object1'] # Accessing object using approved pattern
    block3 = env.scene['Object3'] # Accessing object using approved pattern
    robot = env.scene["robot"] # Accessing robot using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    block1_pos = block1.data.root_pos_w # Accessing object position using approved pattern
    block3_pos = block3.data.root_pos_w # Accessing object position using approved pattern

    # Object dimensions: 0.5m cubed block. Half-size is 0.25m.
    # Define a safe clearance distance. If block is 0.5m, its face is 0.25m from center.
    # We want a clearance of, say, 0.25m from the block's face, so 0.25 + 0.25 = 0.5m from block center.
    safe_distance = 0.5 # meters (hardcoded from object config analysis + desired clearance)

    # Calculate Euclidean distance from pelvis to Object1 and Object3
    # All rewards MUST ONLY use relative distances between objects and robot parts
    dist_to_block1 = torch.norm(pelvis_pos - block1_pos, dim=1)
    dist_to_block3 = torch.norm(pelvis_pos - block3_pos, dim=1)

    # Penalize if the distance is less than safe_distance.
    # Using a linear penalty: penalty increases as distance decreases below threshold.
    # The penalty is 0 if distance >= safe_distance.
    penalty_block1 = torch.where(
        dist_to_block1 < safe_distance,
        -(safe_distance - dist_to_block1), # Continuous penalty
        0.0
    )
    penalty_block3 = torch.where(
        dist_to_block3 < safe_distance,
        -(safe_distance - dist_to_block3), # Continuous penalty
        0.0
    )

    reward = penalty_block1 + penalty_block3

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def foot_contact_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "foot_contact_and_stability_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain stable foot contact with the ground and penalizes excessive
    vertical movement of the feet when they should be on the ground. Also encourages the pelvis
    to stay within a reasonable height range.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts
    robot = env.scene["robot"] # Accessing robot using approved pattern

    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2] # Accessing robot part position using approved pattern
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2] # Accessing robot part position using approved pattern
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2] # Accessing robot part position using approved pattern

    # Target Z for feet (ground level)
    ground_z = 0.0 # Assuming ground is at Z=0

    # Reward for feet being close to the ground.
    # Penalize if feet are too high off the ground when they should be in contact.
    # This encourages walking motion where feet lift and then return to ground.
    foot_clearance_threshold = 0.1 # Max height feet should be off ground for contact (hardcoded)

    # Penalty is proportional to how much the foot is above the threshold.
    # All rewards MUST ONLY use relative distances between objects and robot parts (here, foot to ground_z)
    penalty_left_foot_z = torch.where(
        left_foot_pos_z > foot_clearance_threshold,
        -(left_foot_pos_z - foot_clearance_threshold), # Continuous penalty
        0.0
    )
    penalty_right_foot_z = torch.where(
        right_foot_pos_z > foot_clearance_threshold,
        -(right_foot_pos_z - foot_clearance_threshold), # Continuous penalty
        0.0
    )

    # Reward for pelvis height stability
    # Encourage pelvis to stay within a reasonable range (e.g., 0.6m to 0.8m)
    min_pelvis_z = 0.6 # meters (hardcoded for stable standing)
    max_pelvis_z = 0.8 # meters (hardcoded for stable standing)
    target_pelvis_z_mid = (min_pelvis_z + max_pelvis_z) / 2.0

    reward_pelvis_height = torch.where(
        (pelvis_pos_z >= min_pelvis_z) & (pelvis_pos_z <= max_pelvis_z),
        0.1 * torch.ones_like(pelvis_pos_z), # Small constant positive reward for being in range
        -torch.abs(pelvis_pos_z - target_pelvis_z_mid) # Penalty for being outside range, proportional to distance from mid-point
    )

    reward = penalty_left_foot_z + penalty_right_foot_z + reward_pelvis_height

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
    """
    Reward terms for the walk_to_Second_0_5m_cubed_block skill.
    """
    # Main reward: Robot walks to a pushing position relative to Object2
    walk_to_second_block_main_reward = RewTerm(
        func=walk_to_second_block_main_reward,
        weight=1.0, # Primary reward, highest weight
        params={"normalise": True, "normaliser_name": "walk_to_second_block_main_reward"}
    )

    # Shaping reward 1: Avoid collision with other blocks (Object1 and Object3)
    collision_avoidance_other_blocks_reward = RewTerm(
        func=collision_avoidance_other_blocks_reward,
        weight=0.4, # Shaping reward, lower weight than main
        params={"normalise": True, "normaliser_name": "collision_avoidance_other_blocks_reward"}
    )

    # Shaping reward 2: Encourage stable foot contact and pelvis height
    foot_contact_and_stability_reward = RewTerm(
        func=foot_contact_and_stability_reward,
        weight=0.3, # Shaping reward, lower weight
        params={"normalise": True, "normaliser_name": "foot_contact_and_stability_reward"}
    )