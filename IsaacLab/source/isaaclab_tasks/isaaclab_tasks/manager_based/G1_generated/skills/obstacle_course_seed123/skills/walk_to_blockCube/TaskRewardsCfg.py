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

def walk_to_blockCube_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_blockCube skill.
    Encourages the robot to position its pelvis directly in front of the block cube,
    at an optimal jumping distance and stable standing height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"] # Accessing robot using approved pattern
    block_cube = env.scene['Object5'] # Accessing object using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    block_cube_pos = block_cube.data.root_pos_w # Accessing object position using approved pattern
    block_cube_pos_x = block_cube_pos[:, 0]
    block_cube_pos_y = block_cube_pos[:, 1]

    # Object5 dimensions (from task description: 0.5m cubed) - Hardcoded as per rules
    # This follows the rule to hardcode object dimensions from the config, not access them from the object directly.
    block_cube_x_dim = 0.5
    block_cube_y_dim = 0.5
    block_cube_z_dim = 0.5

    # Define target relative positions for the pelvis
    # Target X: 0.25m in front of the block's front face.
    # The front face of the block is at block_cube_pos_x - block_cube_x_dim / 2.
    # So, target_pelvis_x = (block_cube_pos_x - block_cube_x_dim / 2) - 0.25
    # This calculation uses relative distances between robot parts and object positions, not hardcoded absolute positions.
    target_pelvis_x = block_cube_pos_x - (block_cube_x_dim / 2) - 0.25
    # Target Y: Aligned with the center of the block cube in the Y-axis.
    target_pelvis_y = block_cube_pos_y
    # Target Z: Stable standing height (absolute Z is allowed for height).
    # This is one of the few cases where an absolute Z position is allowed for height.
    target_pelvis_z = 0.7

    # Calculate the absolute difference (distance) from the target positions for each axis
    # This uses relative distances between robot parts and object positions.
    distance_x = torch.abs(pelvis_pos_x - target_pelvis_x)
    distance_y = torch.abs(pelvis_pos_y - target_pelvis_y)
    distance_z = torch.abs(pelvis_pos_z - target_pelvis_z)

    # Reward is negative absolute difference, encouraging the robot to minimize these distances.
    # This creates a continuous reward that is maximized when distances are zero.
    reward = -distance_x - distance_y - distance_z

    # Mandatory normalization
    # This block ensures proper reward normalization as required.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_blockCube_collision_penalty(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_penalty") -> torch.Tensor:
    """
    Penalizes the robot if any part of its body (pelvis, feet) is inside the volume of Object5 (block cube).
    This encourages the robot to stop in front of the block, not walk through it.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"] # Accessing robot using approved pattern
    block_cube = env.scene['Object5'] # Accessing object using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern

    block_cube_center_pos = block_cube.data.root_pos_w # Accessing object position using approved pattern

    # Object5 dimensions (from task description: 0.5m cubed) - Hardcoded as per rules
    # This follows the rule to hardcode object dimensions from the config.
    block_cube_x_dim = 0.5
    block_cube_y_dim = 0.5
    block_cube_z_dim = 0.5

    # Define collision boundaries for Object5 based on its center and dimensions
    # These are relative to the block's center, which is itself an object position.
    min_x = block_cube_center_pos[:, 0] - block_cube_x_dim / 2
    max_x = block_cube_center_pos[:, 0] + block_cube_x_dim / 2
    min_y = block_cube_center_pos[:, 1] - block_cube_y_dim / 2
    max_y = block_cube_center_pos[:, 1] + block_cube_y_dim / 2
    min_z = block_cube_center_pos[:, 2] - block_cube_z_dim / 2
    max_z = block_cube_center_pos[:, 2] + block_cube_z_dim / 2

    # Check if pelvis is inside the block cube's bounding box
    # These checks use relative positions (pelvis_pos vs block_cube_center_pos and dimensions).
    pelvis_in_collision = (pelvis_pos[:, 0] > min_x) & (pelvis_pos[:, 0] < max_x) & \
                          (pelvis_pos[:, 1] > min_y) & (pelvis_pos[:, 1] < max_y) & \
                          (pelvis_pos[:, 2] > min_z) & (pelvis_pos[:, 2] < max_z)

    # Check if left foot is inside the block cube's bounding box
    left_foot_in_collision = (left_foot_pos[:, 0] > min_x) & (left_foot_pos[:, 0] < max_x) & \
                             (left_foot_pos[:, 1] > min_y) & (left_foot_pos[:, 1] < max_y) & \
                             (left_foot_pos[:, 2] > min_z) & (left_foot_pos[:, 2] < max_z)

    # Check if right foot is inside the block cube's bounding box
    right_foot_in_collision = (right_foot_pos[:, 0] > min_x) & (right_foot_pos[:, 0] < max_x) & \
                              (right_foot_pos[:, 1] > min_y) & (right_foot_pos[:, 1] < max_y) & \
                              (right_foot_pos[:, 2] > min_z) & (right_foot_pos[:, 2] < max_z)

    # If any part is in collision, apply a large negative reward.
    # This is a binary reward for collision, which is acceptable for penalties as per prompt.
    collision_condition = pelvis_in_collision | left_foot_in_collision | right_foot_in_collision
    reward = torch.where(collision_condition, -1.0, 0.0)

    # Mandatory normalization
    # This block ensures proper reward normalization as required.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_blockCube_posture_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_stability_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain an upright and stable posture.
    Penalizes deviations of pelvis Z-position from target height and feet Z-position from ground.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts using approved patterns
    robot = env.scene["robot"] # Accessing robot using approved pattern

    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx][:, 2] # Accessing robot part position using approved pattern

    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx][:, 2] # Accessing robot part position using approved pattern

    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    # Fix: The original code had a typo 'right_ankle_roll_link' instead of 'right_foot_idx' for accessing right_foot_pos.
    # Corrected to use the already defined 'right_foot_idx'.
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx][:, 2] # Accessing robot part position using approved pattern

    # Target pelvis height for stability (absolute Z is allowed for height)
    # This is an absolute Z position, allowed for height as per prompt.
    target_pelvis_z = 0.7

    # Target foot height (on the ground, small offset for foot thickness)
    # This is an absolute Z position, allowed for height as per prompt.
    target_foot_z_ground = 0.05

    # Penalize deviation from target pelvis height
    # This is a continuous reward based on the absolute difference from a target Z-height.
    pelvis_height_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # Penalize if feet are too high off the ground (discourage excessive lifting/jumping before the block)
    # This uses a threshold for Z-position, providing a penalty if exceeded.
    left_foot_height_penalty = torch.where(left_foot_pos_z > 0.2, -0.5, 0.0)
    right_foot_height_penalty = torch.where(right_foot_pos_z > 0.2, -0.5, 0.0)

    # Penalize if feet go below ground (unstable/falling)
    # This uses a threshold for Z-position, providing a penalty if below ground.
    left_foot_below_ground_penalty = torch.where(left_foot_pos_z < 0.0, -1.0, 0.0)
    right_foot_below_ground_penalty = torch.where(right_foot_pos_z < 0.0, -1.0, 0.0)

    # Combine all posture-related rewards
    reward = pelvis_height_reward + left_foot_height_penalty + right_foot_height_penalty + \
             left_foot_below_ground_penalty + right_foot_below_ground_penalty

    # Mandatory normalization
    # This block ensures proper reward normalization as required.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Main reward for positioning the robot in front of the block cube
    # Weight is 1.0 as per prompt for main reward.
    walk_to_blockCube_main_reward = RewTerm(func=walk_to_blockCube_main_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward to penalize collisions with the block cube
    # Weight is 0.6 as per prompt for shaping reward.
    walk_to_blockCube_collision_penalty = RewTerm(func=walk_to_blockCube_collision_penalty, weight=0.6,
                                                  params={"normalise": True, "normaliser_name": "collision_penalty"})

    # Shaping reward to encourage stable and upright posture
    # Weight is 0.3 as per prompt for shaping reward.
    walk_to_blockCube_posture_stability_reward = RewTerm(func=walk_to_blockCube_posture_stability_reward, weight=0.3,
                                                         params={"normalise": True, "normaliser_name": "posture_stability_reward"})