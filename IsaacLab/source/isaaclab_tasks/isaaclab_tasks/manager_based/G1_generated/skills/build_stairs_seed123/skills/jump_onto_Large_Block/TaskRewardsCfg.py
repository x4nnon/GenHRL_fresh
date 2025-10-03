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


def main_jump_onto_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_jump_onto_large_block_reward") -> torch.Tensor:
    """
    Primary reward for the robot to land both feet on the top surface of Object3 (Large Block)
    and maintain a stable standing posture. It combines horizontal alignment with the block's top surface,
    vertical alignment of the feet with the block's top surface, and maintaining a stable pelvis height.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Hardcoded dimensions for Object3 (Large Block) from the object configuration.
    # This adheres to the rule of hardcoding dimensions from the object config.
    large_block_height = 0.9
    large_block_x_size = 1.0
    large_block_y_size = 1.0

    # Access Object3 (Large Block) position using the approved pattern.
    object_name = env.scene['Object3']
    block_root_pos = object_name.data.root_pos_w
    block_center_x = block_root_pos[:, 0]
    block_center_y = block_root_pos[:, 1]
    # Calculate the Z-coordinate of the top surface of the block.
    # This is a relative position based on the block's root position and its hardcoded height.
    target_feet_z = block_root_pos[:, 2] + large_block_height / 2.0

    # Access robot body part positions using approved patterns.
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Reward for feet being on top of the block.
    # Horizontal alignment (x, y) for left foot.
    # Uses relative distances between robot parts and object center.
    left_foot_dist_x = torch.abs(left_foot_pos[:, 0] - block_center_x)
    left_foot_dist_y = torch.abs(left_foot_pos[:, 1] - block_center_y)
    # Vertical alignment (z) for left foot.
    left_foot_dist_z = torch.abs(left_foot_pos[:, 2] - target_feet_z)

    # Horizontal alignment (x, y) for right foot.
    right_foot_dist_x = torch.abs(right_foot_pos[:, 0] - block_center_x)
    right_foot_dist_y = torch.abs(right_foot_pos[:, 1] - block_center_y)
    # Vertical alignment (z) for right foot.
    right_foot_dist_z = torch.abs(right_foot_pos[:, 2] - target_feet_z)

    # Combined feet distance reward (negative sum of absolute distances for continuous shaping).
    # This is a continuous reward.
    feet_on_block_reward = - (left_foot_dist_x + left_foot_dist_y + left_foot_dist_z +
                               right_foot_dist_x + right_foot_dist_y + right_foot_dist_z)

    # Reward for pelvis stability and height.
    # Target pelvis Z is relative to the feet's target Z, assuming a standing height.
    # Uses relative distances.
    target_pelvis_z = target_feet_z + 0.7 # 0.7m is a hardcoded relative height, not an absolute position.
    pelvis_z_dist = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)
    # Horizontal distance of pelvis to block center.
    pelvis_horizontal_dist = torch.sqrt(torch.square(pelvis_pos[:, 0] - block_center_x) +
                                        torch.square(pelvis_pos[:, 1] - block_center_y))
    pelvis_stability_reward = - (pelvis_z_dist + pelvis_horizontal_dist)

    # Condition for feet being within the horizontal bounds of the block.
    # Add a small tolerance for landing to make it less strict.
    # These tolerances are small, fixed values, not arbitrary thresholds for positions.
    tolerance_x = 0.1
    tolerance_y = 0.1
    on_block_condition_x_left = (left_foot_pos[:, 0] > (block_center_x - large_block_x_size / 2.0 - tolerance_x)) & \
                                (left_foot_pos[:, 0] < (block_center_x + large_block_x_size / 2.0 + tolerance_x))
    on_block_condition_y_left = (left_foot_pos[:, 1] > (block_center_y - large_block_y_size / 2.0 - tolerance_y)) & \
                                (left_foot_pos[:, 1] < (block_center_y + large_block_y_size / 2.0 + tolerance_y))
    on_block_condition_x_right = (right_foot_pos[:, 0] > (block_center_x - large_block_x_size / 2.0 - tolerance_x)) & \
                                 (right_foot_pos[:, 0] < (block_center_x + large_block_x_size / 2.0 + tolerance_x))
    on_block_condition_y_right = (right_foot_pos[:, 1] > (block_center_y - large_block_y_size / 2.0 - tolerance_y)) & \
                                 (right_foot_pos[:, 1] < (block_center_y + large_block_y_size / 2.0 + tolerance_y))

    feet_horizontally_on_block = on_block_condition_x_left & on_block_condition_y_left & \
                                 on_block_condition_x_right & on_block_condition_y_right

    # Combine rewards, prioritizing feet on block, then pelvis stability.
    # Apply a large penalty if feet are not horizontally on the block, ensuring the robot lands correctly.
    # This creates a strong incentive for correct landing.
    reward = feet_on_block_reward * 0.7 + pelvis_stability_reward * 0.3
    reward = torch.where(feet_horizontally_on_block, reward, -10.0) # Penalty is a fixed value, not an arbitrary threshold for position.

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_approach_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_approach_large_block_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach Object3 horizontally before jumping,
    ensuring it doesn't overshoot or miss the block. It focuses on reducing the horizontal
    distance between the robot's pelvis and the center of Object3.
    """
    # Get normalizer instance.
    RewNormalizer = get_normalizer(env.device)

    # Hardcoded dimensions for Object3 (Large Block) from the object configuration.
    large_block_height = 0.9

    # Access Object3 position using the approved pattern.
    object_name = env.scene['Object3']
    block_root_pos = object_name.data.root_pos_w
    block_center_x = block_root_pos[:, 0]
    block_center_y = block_root_pos[:, 1]

    # Access robot pelvis position using the approved pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Horizontal distance from pelvis to block center.
    # Uses relative distances.
    pelvis_dist_x = torch.abs(pelvis_pos[:, 0] - block_center_x)
    pelvis_dist_y = torch.abs(pelvis_pos[:, 1] - block_center_y)

    # Condition: Pelvis is below the top surface of the block, indicating pre-jump or mid-jump phase.
    # Add a small buffer (0.1m) to ensure it's active before and during the initial ascent of the jump.
    # This is a relative height check, not an absolute position.
    pelvis_below_block_top = pelvis_pos[:, 2] < (block_root_pos[:, 2] + large_block_height / 2.0 + 0.1)

    # Reward is negative of the sum of horizontal distances.
    # This is a continuous reward.
    reward = - (pelvis_dist_x + pelvis_dist_y)
    # Activate this reward only when the pelvis is below the block's top surface.
    reward = torch.where(pelvis_below_block_top, reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_clearance_over_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_clearance_over_large_block_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to lift its feet and pelvis sufficiently high to clear the Large Block
    during the jump, preventing collisions. It penalizes being too low relative to the block's top surface.
    """
    # Get normalizer instance.
    RewNormalizer = get_normalizer(env.device)

    # Hardcoded dimensions for Object3 (Large Block) from the object configuration.
    large_block_height = 0.9

    # Access Object3 position using the approved pattern.
    object_name = env.scene['Object3']
    block_root_pos = object_name.data.root_pos_w
    block_top_z = block_root_pos[:, 2] + large_block_height / 2.0

    # Access robot body part positions using approved patterns.
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Target clearance height above the block's top surface.
    # This is a hardcoded relative height.
    clearance_height = 0.1 # Desired minimum clearance.

    # Vertical distance of feet and pelvis relative to block top.
    # Use torch.max(0.0, ...) to only penalize when below the desired clearance.
    # This ensures the reward is continuous and only penalizes negative clearance.
    left_foot_clearance_dist = torch.max(torch.tensor(0.0, device=env.device), block_top_z + clearance_height - left_foot_pos[:, 2])
    right_foot_clearance_dist = torch.max(torch.tensor(0.0, device=env.device), block_top_z + clearance_height - right_foot_pos[:, 2])
    pelvis_clearance_dist = torch.max(torch.tensor(0.0, device=env.device), block_top_z + clearance_height - pelvis_pos[:, 2])

    # Condition: Robot is horizontally aligned with the block within a reasonable range for jumping over it.
    # This prevents penalizing height when far away and focuses on the jump phase.
    # Use a wider horizontal range than the landing condition to cover the jump trajectory.
    # These ranges are fixed values, not arbitrary thresholds for positions.
    block_center_x = block_root_pos[:, 0]
    block_center_y = block_root_pos[:, 1]
    jump_range_x = 1.5 # Robot should be within this x-range of the block to jump.
    jump_range_y = 1.5 # Robot should be within this y-range of the block to jump.

    horizontally_aligned_for_jump = \
        (pelvis_pos[:, 0] > (block_center_x - jump_range_x / 2.0)) & \
        (pelvis_pos[:, 0] < (block_center_x + jump_range_x / 2.0)) & \
        (pelvis_pos[:, 1] > (block_center_y - jump_range_y / 2.0)) & \
        (pelvis_pos[:, 1] < (block_center_y + jump_range_y / 2.0))

    # Reward is negative of the sum of clearance distances (penalizes being too low).
    # This is a continuous reward.
    reward = - (left_foot_clearance_dist + right_foot_clearance_dist + pelvis_clearance_dist)
    # Activate this reward only when the robot is horizontally aligned for a jump.
    reward = torch.where(horizontally_aligned_for_jump, reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for landing and stabilizing on the large block.
    MainJumpOntoLargeBlockReward = RewTerm(func=main_jump_onto_large_block_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "main_jump_onto_large_block_reward"})

    # Shaping reward for horizontal approach before the jump.
    ShapingApproachLargeBlockReward = RewTerm(func=shaping_approach_large_block_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "shaping_approach_large_block_reward"})

    # Shaping reward for vertical clearance during the jump.
    ShapingClearanceOverLargeBlockReward = RewTerm(func=shaping_clearance_over_large_block_reward, weight=0.4,
                                                   params={"normalise": True, "normaliser_name": "shaping_clearance_over_large_block_reward"})