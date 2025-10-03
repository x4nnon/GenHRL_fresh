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


def navigate_and_jump_on_block_cube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "navigate_jump_block_reward") -> torch.Tensor:
    """
    Primary reward for the Navigate_and_Jump_On_BlockCube skill.
    Guides the robot to first approach the block cube (Object5) horizontally,
    then encourages it to jump and land its feet on top of the block,
    and finally rewards for maintaining stability on the block.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using the approved pattern.
    object5 = env.scene['Object5'] # Block cube
    object5_pos = object5.data.root_pos_w

    # Access the required robot parts using the approved pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_x = left_foot_pos[:, 0]
    left_foot_pos_y = left_foot_pos[:, 1]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Object5 dimensions (from task description: 0.5m cubed). Hardcoded as per requirements.
    block_height = 0.5
    block_half_x = 0.25
    block_half_y = 0.25

    # Calculate block's top surface z-coordinate relative to its root position.
    block_top_z = object5_pos[:, 2] + block_height / 2.0

    # Phase 1: Approach the block horizontally (pelvis)
    # Reward for reducing horizontal distance to the block's center.
    # Using absolute differences for relative distances.
    dist_x_pelvis_block = torch.abs(pelvis_pos_x - object5_pos[:, 0])
    dist_y_pelvis_block = torch.abs(pelvis_pos_y - object5_pos[:, 1])
    # Reward is negative, so minimizing distance increases reward.
    approach_reward = -0.5 * dist_x_pelvis_block - 0.5 * dist_y_pelvis_block

    # Phase 2 & 3: Land on top of the block and maintain stability (feet and pelvis)
    # Condition for being "on top" of the block horizontally for left foot.
    on_block_x_condition_left = (left_foot_pos_x > object5_pos[:, 0] - block_half_x) & \
                                (left_foot_pos_x < object5_pos[:, 0] + block_half_x)
    on_block_y_condition_left = (left_foot_pos_y > object5_pos[:, 1] - block_half_y) & \
                                (left_foot_pos_y < object5_pos[:, 1] + block_half_y)
    on_block_horizontal_left = on_block_x_condition_left & on_block_y_condition_left

    # Condition for being "on top" of the block horizontally for right foot.
    on_block_x_condition_right = (right_foot_pos_x > object5_pos[:, 0] - block_half_x) & \
                                 (right_foot_pos_x < object5_pos[:, 0] + block_half_x)
    on_block_y_condition_right = (right_foot_pos_y > object5_pos[:, 1] - block_half_y) & \
                                 (right_foot_pos_y < object5_pos[:, 1] + block_half_y)
    on_block_horizontal_right = on_block_x_condition_right & on_block_y_condition_right

    # Reward for feet being at the correct height relative to the block's top surface.
    # Target height for feet is block_top_z.
    feet_height_reward_left = -torch.abs(left_foot_pos_z - block_top_z)
    feet_height_reward_right = -torch.abs(right_foot_pos_z - block_top_z)

    # Reward for pelvis being at a stable height above the block's top surface.
    # Target pelvis height relative to block top: 0.7m (pelvis_z = 0.7 is default standing height).
    target_pelvis_z_on_block = block_top_z + 0.7
    pelvis_stability_reward = -torch.abs(pelvis_pos_z - target_pelvis_z_on_block)

    # Combine rewards based on conditions.
    # If both feet are horizontally on the block AND above or slightly below its top surface,
    # prioritize landing and stability rewards.
    feet_on_block_condition = on_block_horizontal_left & on_block_horizontal_right & \
                              (left_foot_pos_z > block_top_z - 0.1) & (right_foot_pos_z > block_top_z - 0.1)

    # Use torch.where for conditional reward application, ensuring continuity.
    reward = torch.where(feet_on_block_condition,
                         (feet_height_reward_left + feet_height_reward_right) * 0.5 + pelvis_stability_reward * 0.5,
                         approach_reward) # If not on block, reward for approaching.

    # Add a small constant positive reward for being on the block to encourage staying there.
    reward = torch.where(feet_on_block_condition, reward + 0.1, reward)

    # Transform to positive and continuous using exponential.
    reward = torch.exp(reward)

    # Mandatory normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_block_cube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_block_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the block cube (Object5).
    Penalizes the robot for colliding with the sides of Object5, especially during approach and jump.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using the approved pattern.
    object5 = env.scene['Object5'] # Block cube
    object5_pos = object5.data.root_pos_w

    # Access the required robot parts using the approved pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Object5 dimensions (from task description: 0.5m cubed). Hardcoded as per requirements.
    block_height = 0.5
    block_half_x = 0.25
    block_half_y = 0.25

    # Calculate block's top surface z-coordinate.
    block_top_z = object5_pos[:, 2] + block_height / 2.0

    # Distance from pelvis to block's center (horizontal).
    dist_x_pelvis_block = torch.abs(pelvis_pos_x - object5_pos[:, 0])
    dist_y_pelvis_block = torch.abs(pelvis_pos_y - object5_pos[:, 1])

    # Condition: Robot is horizontally close to the block but below its top surface.
    # This prevents penalizing when already on top or clearly past it.
    # A small buffer (0.3m) is added to the block's half-dimensions to define "close".
    close_to_block_horizontal = (dist_x_pelvis_block < block_half_x + 0.3) & \
                                (dist_y_pelvis_block < block_half_y + 0.3)

    # Pelvis is below the top surface of the block (with a small buffer).
    below_block_top = (pelvis_pos_z < block_top_z - 0.1)

    # Activation condition for the penalty.
    activation_condition = close_to_block_horizontal & below_block_top

    # Penalize if pelvis is inside the block's horizontal bounds when below its top.
    # `clearance_x` and `clearance_y` will be positive if there's penetration, 0 otherwise.
    clearance_x = torch.max(torch.tensor(0.0, device=env.device), block_half_x - dist_x_pelvis_block)
    clearance_y = torch.max(torch.tensor(0.0, device=env.device), block_half_y - dist_y_pelvis_block)

    # Reward is negative for penetration, 0 otherwise. Scaled for impact.
    collision_reward = - (clearance_x + clearance_y) * 5.0

    # Apply the penalty only when the activation condition is met.
    reward = torch.where(activation_condition, collision_reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "upright_posture_reward") -> torch.Tensor:
    """
    Shaping reward for maintaining an upright and stable posture.
    Penalizes the pelvis dropping too low or going too high unnecessarily.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts using the approved pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Target pelvis height for standing (default stable height). Hardcoded as per requirements.
    target_pelvis_z_standing = 0.7

    # Reward for maintaining pelvis close to target_pelvis_z_standing.
    # This reward is always active to encourage general stability.
    # Using negative absolute difference for a continuous reward.
    posture_reward = -torch.abs(pelvis_pos_z - target_pelvis_z_standing)

    # Mandatory normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, posture_reward)
        RewNormalizer.update_stats(normaliser_name, posture_reward)
        return scaled_reward
    return posture_reward


@configclass
class TaskRewardsCfg:
    """
    Reward terms for the Navigate_and_Jump_On_BlockCube skill.
    """
    # Primary reward for approaching, jumping, landing, and stabilizing on the block.
    NavigateJumpOnBlockCubeReward = RewTerm(func=navigate_and_jump_on_block_cube_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "navigate_jump_block_reward"})

    # Shaping reward for avoiding collisions with the block.
    CollisionAvoidanceBlockCubeReward = RewTerm(func=collision_avoidance_block_cube_reward, weight=0.4,
                                                params={"normalise": True, "normaliser_name": "collision_avoidance_block_reward"})

    # Shaping reward for maintaining an upright posture.
    MaintainUprightPostureReward = RewTerm(func=maintain_upright_posture_reward, weight=0.2,
                                           params={"normalise": True, "normaliser_name": "upright_posture_reward"})