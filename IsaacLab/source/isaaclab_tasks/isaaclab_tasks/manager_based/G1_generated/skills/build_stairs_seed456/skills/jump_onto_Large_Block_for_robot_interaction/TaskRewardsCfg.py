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


def main_jump_onto_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the robot to land stably on top of the Large Block (Object3).
    This reward encourages the feet to be horizontally within the block's bounds and vertically on its top surface,
    while the pelvis is at a stable standing height above the block.
    """
    # Get normalizer instance as per mandatory requirement 2.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved pattern env.scene['ObjectName'] as per requirement 4.
    large_block = env.scene['Object3'] # Accessing Object3 (Large Block) as per requirements

    # Access the required robot part(s) using approved pattern robot.body_names.index('part_name') as per requirement 4.
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index

    # Accessing robot part positions using approved pattern robot.data.body_pos_w as per requirement 4.
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position
    # Accessing object position using approved pattern object.data.root_pos_w as per requirement 4.
    large_block_pos = large_block.data.root_pos_w # Accessing object position

    # Object3 dimensions (Large Block: x=1m y=1m z=0.9m) - Hardcoded from object configuration as per requirement 8.
    block_height = 0.9
    block_half_x = 0.5
    block_half_y = 0.5

    # Target Z position for feet (top of the block) - Calculated using relative distance as per requirement 1.
    # Relative distance: target_feet_z is block's center Z + half its height
    target_feet_z = large_block_pos[:, 2] + block_height / 2.0

    # Target Z position for pelvis (0.7m above the block's top surface) - Calculated using relative distance as per requirement 1.
    # Relative distance: target_pelvis_z is target_feet_z + 0.7m
    target_pelvis_z = target_feet_z + 0.7

    # Horizontal distance of feet to block center (relative distances) as per requirement 1.
    left_foot_dist_x = torch.abs(left_foot_pos[:, 0] - large_block_pos[:, 0])
    left_foot_dist_y = torch.abs(left_foot_pos[:, 1] - large_block_pos[:, 1])
    right_foot_dist_x = torch.abs(right_foot_pos[:, 0] - large_block_pos[:, 0])
    right_foot_dist_y = torch.abs(right_foot_pos[:, 1] - large_block_pos[:, 1])

    # Vertical distance of feet to block top (relative distances) as per requirement 1.
    left_foot_dist_z = torch.abs(left_foot_pos[:, 2] - target_feet_z)
    right_foot_dist_z = torch.abs(right_foot_pos[:, 2] - target_feet_z)

    # Vertical distance of pelvis to target height (relative distance) as per requirement 1.
    pelvis_dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for feet being within horizontal bounds of the block. Continuous reward as per requirement 7.
    # Penalizes feet being outside the block's horizontal dimensions.
    # Reward is continuous, decreasing linearly as feet move beyond the half-dimensions.
    feet_horizontal_penalty = (
        torch.max(torch.zeros_like(left_foot_dist_x), left_foot_dist_x - block_half_x) * 2.0 +
        torch.max(torch.zeros_like(left_foot_dist_y), left_foot_dist_y - block_half_y) * 2.0 +
        torch.max(torch.zeros_like(right_foot_dist_x), right_foot_dist_x - block_half_x) * 2.0 +
        torch.max(torch.zeros_like(right_foot_dist_y), right_foot_dist_y - block_half_y) * 2.0
    )
    feet_horizontal_reward = -feet_horizontal_penalty

    # Reward for feet being at the correct vertical height. Continuous reward as per requirement 7.
    # Penalizes deviation from the target Z height for the feet.
    feet_vertical_reward = -left_foot_dist_z - right_foot_dist_z

    # Reward for pelvis being at the target stable height. Continuous reward as per requirement 7.
    # Penalizes deviation from the target Z height for the pelvis.
    pelvis_height_reward = -pelvis_dist_z

    # Combine rewards
    reward = feet_horizontal_reward + feet_vertical_reward + pelvis_height_reward

    # Mandatory normalization as per requirement 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def approach_large_block_horizontal_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_reward") -> torch.Tensor:
    """
    This reward encourages the robot to approach the Large Block (Object3) horizontally,
    aligning its pelvis with the block's center and penalizing overshooting in the x-direction.
    """
    # Get normalizer instance as per mandatory requirement 2.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved pattern env.scene['ObjectName'] as per requirement 4.
    large_block = env.scene['Object3'] # Accessing Object3 (Large Block)

    # Access the required robot part(s) using approved pattern robot.body_names.index('part_name') as per requirement 4.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index
    # Accessing robot part position using approved pattern robot.data.body_pos_w as per requirement 4.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position
    # Accessing object position using approved pattern object.data.root_pos_w as per requirement 4.
    large_block_pos = large_block.data.root_pos_w # Accessing object position

    # Horizontal distance of pelvis to the block's center in x and y (relative distances) as per requirement 1.
    pelvis_block_dist_x = large_block_pos[:, 0] - pelvis_pos[:, 0]
    pelvis_block_dist_y = large_block_pos[:, 1] - pelvis_pos[:, 1]

    # Object3 dimensions (Large Block: x=1m y=1m z=0.9m) - Hardcoded from object configuration as per requirement 8.
    block_half_x = 0.5
    block_half_y = 0.5

    # Reward for approaching the block horizontally (x-axis). Continuous reward as per requirement 7.
    # Reward is higher when pelvis is closer to the block's x-center.
    approach_x_reward = -torch.abs(pelvis_block_dist_x)

    # Penalize if pelvis moves too far past the block's x-center (e.g., more than half block width). Continuous penalty as per requirement 7.
    # This helps prevent overshooting the jump. Penalty activates when pelvis_block_dist_x is negative and
    # its absolute value exceeds block_half_x.
    overshoot_penalty_x = torch.where(pelvis_block_dist_x < -block_half_x, torch.abs(pelvis_block_dist_x + block_half_x) * 5.0, 0.0)

    # Reward for being aligned with the block in y-axis. Continuous reward as per requirement 7.
    # Penalizes deviation from the block's y-center.
    align_y_reward = -torch.abs(pelvis_block_dist_y)

    reward = approach_x_reward - overshoot_penalty_x + align_y_reward

    # Mandatory normalization as per requirement 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def jump_and_land_vertical_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "vertical_dynamics_reward") -> torch.Tensor:
    """
    This reward encourages the robot to gain sufficient vertical height during the jump to clear the Large Block (Object3)
    and then to reduce its vertical velocity for a stable landing. It also includes a collision avoidance component.
    """
    # Get normalizer instance as per mandatory requirement 2.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved pattern env.scene['ObjectName'] as per requirement 4.
    large_block = env.scene['Object3'] # Accessing Object3 (Large Block)

    # Access the required robot part(s) using approved pattern robot.body_names.index('part_name') as per requirement 4.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index

    # Accessing robot part positions and velocities using approved patterns as per requirement 4.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position
    # Accessing object position using approved pattern object.data.root_pos_w as per requirement 4.
    large_block_pos = large_block.data.root_pos_w # Accessing object position

    pelvis_vel_z = robot.data.body_vel_w[:, pelvis_idx, 2] # Accessing robot part vertical velocity
    left_foot_vel_z = robot.data.body_vel_w[:, left_foot_idx, 2] # Accessing robot part vertical velocity
    right_foot_vel_z = robot.data.body_vel_w[:, right_foot_idx, 2] # Accessing robot part vertical velocity

    # Object3 dimensions (Large Block: x=1m y=1m z=0.9m) - Hardcoded from object configuration as per requirement 8.
    block_height = 0.9
    block_half_x = 0.5
    block_half_y = 0.5

    # Define block's bounding box for collision check (relative to block's center) as per requirement 1.
    block_min_x = large_block_pos[:, 0] - block_half_x
    block_max_x = large_block_pos[:, 0] + block_half_x
    block_min_y = large_block_pos[:, 1] - block_half_y
    block_max_y = large_block_pos[:, 1] + block_half_y
    block_min_z = large_block_pos[:, 2] - block_height / 2.0
    block_max_z = large_block_pos[:, 2] + block_height / 2.0

    # Check if pelvis is inside the block's volume (relative positions) as per requirement 1.
    pelvis_in_block_x = (pelvis_pos[:, 0] > block_min_x) & (pelvis_pos[:, 0] < block_max_x)
    pelvis_in_block_y = (pelvis_pos[:, 1] > block_min_y) & (pelvis_pos[:, 1] < block_max_y)
    pelvis_in_block_z = (pelvis_pos[:, 2] > block_min_z) & (pelvis_pos[:, 2] < block_max_z)
    pelvis_collision_condition = pelvis_in_block_x & pelvis_in_block_y & pelvis_in_block_z

    # Collision penalty: large negative reward if pelvis is inside the block. Binary penalty.
    collision_penalty = torch.where(pelvis_collision_condition, -10.0, 0.0)

    # Reward for gaining height during jump (when pelvis is below block top). Continuous reward as per requirement 7.
    # This encourages jumping over the block. Reward is continuous, increasing with height relative to block bottom.
    height_gain_reward = torch.where(
        pelvis_pos[:, 2] < block_max_z,
        (pelvis_pos[:, 2] - block_min_z) * 0.5, # Scale to make it less dominant than landing
        0.0
    )

    # Reward for reducing vertical velocity for stable landing (when feet are near block top). Continuous reward as per requirement 7.
    # This encourages a soft landing. Reward is continuous, penalizing high vertical velocity.
    # The condition checks if feet are within a small vertical range around the block's top.
    landing_vel_reward = torch.where(
        ((left_foot_pos[:, 2] < block_max_z + 0.1) & (left_foot_pos[:, 2] > block_max_z - 0.1)) |
        ((right_foot_pos[:, 2] < block_max_z + 0.1) & (right_foot_pos[:, 2] > block_max_z - 0.1)),
        -torch.abs(left_foot_vel_z) - torch.abs(right_foot_vel_z),
        0.0
    )

    reward = collision_penalty + height_gain_reward + landing_vel_reward

    # Mandatory normalization as per requirement 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for landing stably on the large block with weight 1.0 as per requirement 12.
    main_jump_onto_large_block_reward = RewTerm(func=main_jump_onto_large_block_reward, weight=1.0,
                                                params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for horizontal approach and alignment with weight 0.4 as per requirement 12.
    approach_large_block_horizontal_reward = RewTerm(func=approach_large_block_horizontal_reward, weight=0.4,
                                                     params={"normalise": True, "normaliser_name": "approach_reward"})

    # Shaping reward for vertical jump dynamics and landing stability with weight 0.3 as per requirement 12.
    jump_and_land_vertical_reward = RewTerm(func=jump_and_land_vertical_reward, weight=0.3,
                                            params={"normalise": True, "normaliser_name": "vertical_dynamics_reward"})