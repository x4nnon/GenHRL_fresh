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

def move_three_objects_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_reward") -> torch.Tensor:
    """
    Primary reward for moving each of the three 0.5m cubed blocks onto the platform.
    This reward encourages blocks to be close to the platform's center in XY and at the correct Z-height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    platform = env.scene['Object4']

    # Object dimensions (hardcoded from skill info: 0.5m cubed blocks, platform z=0.001)
    # This follows the rule of hardcoding dimensions from the object configuration.
    block_half_size = 0.5 / 2.0 # 0.25m
    platform_z_height = 0.001

    # Calculate target Z for blocks on platform (platform_z_height + block_half_size)
    target_block_z = platform_z_height + block_half_size

    # Initialize total primary reward for the batch
    total_primary_reward = torch.zeros(env.num_envs, device=env.device)

    # Loop through each block to calculate its contribution to the reward
    for obj in [object1, object2, object3]:
        # Access object positions using approved pattern
        obj_pos = obj.data.root_pos_w
        platform_pos = platform.data.root_pos_w

        # Calculate relative distance from block to platform center (x, y components)
        # This uses relative distances between objects, as required.
        dist_xy_block_platform = torch.norm(obj_pos[:, :2] - platform_pos[:, :2], dim=1)

        # Reward for block being closer to platform center (negative distance, so closer is higher reward)
        # This is a continuous reward based on relative distance.
        distance_reward = -dist_xy_block_platform

        # Reward for block being at the correct Z-height on the platform
        # This uses relative Z-height to the target, ensuring the block is on top.
        z_height_reward = -torch.abs(obj_pos[:, 2] - target_block_z)

        # Sum weighted rewards for this block
        # Weights are chosen to prioritize XY position over Z-height.
        total_primary_reward += (distance_reward * 0.7) + (z_height_reward * 0.3)

    reward = total_primary_reward

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_approach_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_block_reward") -> torch.Tensor:
    """
    Shaping reward encouraging the robot to approach and make contact with the block
    that is currently furthest from the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    platform = env.scene['Object4']

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    left_palm_idx = robot.body_names.index('left_palm_link')
    right_palm_idx = robot.body_names.index('right_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    # Object dimensions (hardcoded from skill info: 0.5m cubed blocks)
    block_half_size = 0.5 / 2.0 # 0.25m
    platform_z_height = 0.001
    target_block_z = platform_z_height + block_half_size

    # Calculate current XY distance of each block to platform
    # This uses relative distances between objects.
    dist_obj1_platform = torch.norm(object1.data.root_pos_w[:, :2] - platform.data.root_pos_w[:, :2], dim=1)
    dist_obj2_platform = torch.norm(object2.data.root_pos_w[:, :2] - platform.data.root_pos_w[:, :2], dim=1)
    dist_obj3_platform = torch.norm(object3.data.root_pos_w[:, :2] - platform.data.root_pos_w[:, :2], dim=1)

    # Stack distances and find the index of the block furthest from the platform for each environment
    block_distances = torch.stack([dist_obj1_platform, dist_obj2_platform, dist_obj3_platform], dim=1)
    furthest_block_indices = torch.argmax(block_distances, dim=1)

    # Select the position of the furthest block for each environment in the batch
    # This ensures correct indexing for batched environments.
    all_block_positions = torch.stack([object1.data.root_pos_w, object2.data.root_pos_w, object3.data.root_pos_w], dim=1)
    # Use advanced indexing to avoid shape/expand issues
    env_ids = torch.arange(all_block_positions.shape[0], device=env.device)
    furthest_block_pos = all_block_positions[env_ids, furthest_block_indices, :]

    # Calculate 3D distance from robot hands to the furthest block
    # These are relative distances between robot parts and objects.
    dist_left_palm_to_block = torch.norm(left_palm_pos - furthest_block_pos, dim=1)
    dist_right_palm_to_block = torch.norm(right_palm_pos - furthest_block_pos, dim=1)

    # Reward for being close to the block (using the minimum distance of the two palms)
    # This is a continuous reward.
    approach_reward = -torch.min(dist_left_palm_to_block, dist_right_palm_to_block)

    # Condition: Only reward if the block is not yet on the platform
    # A block is considered "on platform" if its Z is close to target_block_z and its XY is close to platform.
    # For this shaping reward, we use a Z-height check as a primary indicator for activation.
    # The primary reward handles the XY proximity.
    is_block_on_platform_z = torch.abs(furthest_block_pos[:, 2] - target_block_z) < (block_half_size * 0.5) # within half the block's height
    is_block_on_platform_xy = torch.norm(furthest_block_pos[:, :2] - platform.data.root_pos_w[:, :2], dim=1) < (0.5 + 0.5) # block size + small margin

    # Activation condition: Only apply this reward if the furthest block is NOT yet on the platform
    # This prevents the robot from trying to push blocks already successfully placed.
    activation_condition = ~(is_block_on_platform_z & is_block_on_platform_xy)

    # Apply the reward only when the activation condition is met
    reward = torch.where(activation_condition, approach_reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    """
    Shaping reward encouraging the robot to maintain a stable, upright posture
    and stay within a reasonable operating area.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_quat = robot.data.body_quat_w[:, pelvis_idx] # Quaternion for orientation

    # Pelvis position components
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Target pelvis Z height for stability (around 0.7m for a humanoid)
    target_pelvis_z = 0.7

    # Reward for maintaining pelvis Z height
    # This is a continuous reward based on the absolute difference from a target Z-height.
    pelvis_z_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Reward for not straying too far in X and Y from initial robot position (0,0,0)
    # This helps keep the robot in a reasonable operating area relative to the blocks/platform.
    # This uses the absolute position of the pelvis relative to the world origin (0,0,0),
    # which is acceptable for general robot positioning/stability.
    pelvis_xy_deviation_reward = -torch.norm(pelvis_pos[:, :2], dim=1)

    # Reward for maintaining upright orientation (penalize large pitch/roll)
    # A common proxy for uprightness is to penalize the magnitude of the x and y components of the quaternion.
    # This encourages the quaternion to be close to [w, 0, 0, z] or [w, 0, 0, 0] for upright.
    # This is a continuous reward.
    pelvis_orientation_reward = -torch.norm(pelvis_quat[:, 1:3], dim=1) # Penalize x and y components of quat (assuming [w,x,y,z])

    # Combine stability rewards with chosen weights
    reward = (pelvis_z_reward * 0.4) + (pelvis_xy_deviation_reward * 0.3) + (pelvis_orientation_reward * 0.3)

    # Mandatory normalization implementation
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
    Configuration for the rewards used in the move_three_objects_seed123 skill.
    """
    # Primary reward for moving blocks onto the platform
    # Weight is 1.0 as it's the main objective.
    primary_block_placement_reward = RewTerm(
        func=move_three_objects_primary_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "primary_reward"}
    )

    # Shaping reward for robot approaching and contacting the furthest block
    # Weight is 0.5 to guide behavior without overpowering the primary goal.
    robot_approach_block_shaping_reward = RewTerm(
        func=robot_approach_block_reward,
        weight=0.5,
        params={"normalise": True, "normaliser_name": "approach_block_reward"}
    )

    # Shaping reward for robot stability and posture
    # Weight is 0.2 to encourage good robot behavior without distracting from the main task.
    robot_stability_shaping_reward = RewTerm(
        func=robot_stability_reward,
        weight=0.2,
        params={"normalise": True, "normaliser_name": "stability_reward"}
    )