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

def reward_feet_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_on_block_reward") -> torch.Tensor:
    """
    Primary reward for the robot's feet to land and stay on top of the block cube (Object5).
    It measures the combined distance of both feet to the center of the block's top surface,
    considering x, y, and z components separately. The reward is maximized when both feet
    are within the block's x and y bounds and at the correct z-height.
    """
    # Get normalizer instance for this reward function.
    RewNormalizer = get_normalizer(env.device)

    # Access the required object (Object5: block cube) using the approved pattern.
    block_cube = env.scene['Object5']
    block_pos = block_cube.data.root_pos_w

    # Access the required robot parts (left and right feet) using the approved pattern.
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcode block dimensions from the task description (0.5m cubed).
    # This adheres to the rule of hardcoding dimensions from the object configuration.
    block_height = 0.5
    block_half_x = 0.5 / 2.0 # 0.25m
    block_half_y = 0.5 / 2.0 # 0.25m

    # Calculate target z-position for feet (top of the block).
    # The block's root_pos_w is its center. For a 0.5m high block, its top surface is at root_pos_w[:, 2] + 0.25m.
    # This uses the block's position and hardcoded height, adhering to relative distance rules.
    target_z = block_pos[:, 2] + block_height / 2.0

    # Calculate distances for left foot relative to the block's center.
    # These are relative distances, adhering to the rule.
    dist_lx = torch.abs(left_foot_pos[:, 0] - block_pos[:, 0])
    dist_ly = torch.abs(left_foot_pos[:, 1] - block_pos[:, 1])
    dist_lz = torch.abs(left_foot_pos[:, 2] - target_z)

    # Calculate distances for right foot relative to the block's center.
    # These are relative distances, adhering to the rule.
    dist_rx = torch.abs(right_foot_pos[:, 0] - block_pos[:, 0])
    dist_ry = torch.abs(right_foot_pos[:, 1] - block_pos[:, 1])
    dist_rz = torch.abs(right_foot_pos[:, 2] - target_z)

    # Reward for x and y position (within block bounds).
    # Reward is higher when feet are closer to the center and within bounds.
    # Using a continuous reward: 0.1 - distance if inside, -distance if outside.
    # This encourages being within bounds and penalizes being outside.
    # The penalty for being outside is doubled to make it stronger, ensuring continuous reward.
    reward_x_l = torch.where(dist_lx < block_half_x, 0.1 - dist_lx, -dist_lx * 2.0)
    reward_y_l = torch.where(dist_ly < block_half_y, 0.1 - dist_ly, -dist_ly * 2.0)
    reward_x_r = torch.where(dist_rx < block_half_x, 0.1 - dist_rx, -dist_rx * 2.0)
    reward_y_r = torch.where(dist_ry < block_half_y, 0.1 - dist_ry, -dist_ry * 2.0)

    # Reward for z position (at target height).
    # Negative absolute distance to encourage being exactly at the target height, ensuring continuous reward.
    reward_z_l = -dist_lz
    reward_z_r = -dist_rz

    # Combine rewards for both feet.
    # The sum is scaled to keep the reward magnitude reasonable.
    primary_reward = (reward_x_l + reward_y_l + reward_z_l + reward_x_r + reward_y_r + reward_z_r) * 0.5

    # Mandatory reward normalization.
    # This block is required in every reward function.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def reward_approach_block_x(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_block_x_reward") -> torch.Tensor:
    """
    Shaping reward to guide the robot to approach the block cube (Object5) along the x-axis.
    It encourages the pelvis to move closer to the block's x-center. This reward is active
    until the robot's pelvis has passed the block's x-center, at which point the primary
    reward for landing on the block should dominate.
    """
    # Get normalizer instance.
    RewNormalizer = get_normalizer(env.device)

    # Access the required object (Object5: block cube).
    block_cube = env.scene['Object5']
    block_pos = block_cube.data.root_pos_w

    # Access the required robot part (pelvis).
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcode block dimensions for activation condition.
    # This adheres to the rule of hardcoding dimensions from the object configuration.
    block_half_x = 0.5 / 2.0 # 0.25m

    # Calculate distance to block's x-center.
    # This is a relative distance between pelvis x and block x.
    distance_x_pelvis_to_block = torch.abs(block_pos[:, 0] - pelvis_pos[:, 0])

    # Activation condition: Pelvis is behind or at the x-center of the block.
    # Assuming robot starts before the block and moves in positive x direction.
    # The condition `pelvis_pos[:, 0] < block_pos[:, 0] + block_half_x` means the pelvis
    # is still approaching or just entering the block's x-span.
    # This uses relative positions and hardcoded block dimension.
    activation_condition = (pelvis_pos[:, 0] < block_pos[:, 0] + block_half_x)

    # Reward is negative distance, so minimizing distance maximizes reward.
    # This is a continuous reward.
    shaping_reward_1 = -distance_x_pelvis_to_block

    # Apply activation condition: reward is zero if condition is not met.
    # This ensures the reward is only active during the approach phase.
    shaping_reward_1 = torch.where(activation_condition, shaping_reward_1, torch.zeros_like(shaping_reward_1))

    # Mandatory reward normalization.
    # This block is required in every reward function.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_1)
        return scaled_reward
    return shaping_reward_1

def reward_stability_on_block(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_on_block_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable, upright posture on top of the block
    and penalize "collisions" (pelvis too low and inside the block). It also encourages the pelvis
    to be at a reasonable height (e.g., 0.7m above the block's top surface) once on the block.
    """
    # Get normalizer instance.
    RewNormalizer = get_normalizer(env.device)

    # Access the required object (Object5: block cube).
    block_cube = env.scene['Object5']
    block_pos = block_cube.data.root_pos_w

    # Access the required robot parts (pelvis, left and right feet).
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcode block dimensions from the task description (0.5m cubed).
    # This adheres to the rule of hardcoding dimensions from the object configuration.
    block_height = 0.5
    block_half_x = 0.5 / 2.0
    block_half_y = 0.5 / 2.0

    # Target pelvis height relative to the ground when on top of the block.
    # Block top surface is at block_pos[:, 2] + block_height / 2.0.
    # Target pelvis height is 0.7m above the block's top surface.
    # This uses relative positions and hardcoded dimensions.
    target_pelvis_z = block_pos[:, 2] + block_height / 2.0 + 0.7

    # Activation condition: Feet are approximately on the block.
    # Check if both feet are within the x and y bounds of the block and above a certain z-height.
    # This ensures the reward is active only when the robot has successfully landed.
    # All conditions use relative distances and hardcoded dimensions.
    feet_on_block_condition = (
        (torch.abs(left_foot_pos[:, 0] - block_pos[:, 0]) < block_half_x) &
        (torch.abs(left_foot_pos[:, 1] - block_pos[:, 1]) < block_half_y) &
        (left_foot_pos[:, 2] > block_pos[:, 2] + block_height * 0.4) & # Foot is above a certain height (e.g., 40% of block height from block center)
        (torch.abs(right_foot_pos[:, 0] - block_pos[:, 0]) < block_half_x) &
        (torch.abs(right_foot_pos[:, 1] - block_pos[:, 1]) < block_half_y) &
        (right_foot_pos[:, 2] > block_pos[:, 2] + block_height * 0.4)
    )

    # Reward for pelvis height stability.
    # Negative absolute distance to encourage the pelvis to be at the target stable height.
    # This is a continuous reward based on relative distance.
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Collision avoidance proxy: Penalize if pelvis is too low and inside the block's horizontal bounds.
    # This implies the robot is falling through or colliding with the block's body.
    # All conditions use relative distances and hardcoded dimensions.
    pelvis_too_low_and_inside = (
        (pelvis_pos[:, 2] < block_pos[:, 2] + block_height / 2.0 - 0.1) & # Pelvis below block top surface by a margin
        (torch.abs(pelvis_pos[:, 0] - block_pos[:, 0]) < block_half_x) &
        (torch.abs(pelvis_pos[:, 1] - block_pos[:, 1]) < block_half_y)
    )
    # Apply a significant negative penalty for this "collision" state.
    # This is a continuous reward (0 or -10.0) based on a condition.
    collision_penalty = torch.where(pelvis_too_low_and_inside, -10.0, 0.0)

    # Combine pelvis height reward and collision penalty.
    shaping_reward_2 = pelvis_height_reward + collision_penalty

    # Apply activation condition: reward is zero if feet are not on the block.
    # This ensures the reward is only active when the robot is on the block.
    shaping_reward_2 = torch.where(feet_on_block_condition, shaping_reward_2, torch.zeros_like(shaping_reward_2))

    # Mandatory reward normalization.
    # This block is required in every reward function.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_2)
        return scaled_reward
    return shaping_reward_2

@configclass
class TaskRewardsCfg:
    # Primary reward for feet landing and staying on the block.
    # Weight: 1.0, as this is the main objective.
    FeetOnBlockReward = RewTerm(func=reward_feet_on_block, weight=1.0,
                                params={"normalise": True, "normaliser_name": "feet_on_block_reward"})

    # Shaping reward for approaching the block along the x-axis.
    # Weight: 0.4, to guide initial movement without dominating.
    ApproachBlockXReward = RewTerm(func=reward_approach_block_x, weight=0.4,
                                   params={"normalise": True, "normaliser_name": "approach_block_x_reward"})

    # Shaping reward for stability and pelvis height on top of the block, and collision avoidance.
    # Weight: 0.6, important for stable landing and final posture.
    StabilityOnBlockReward = RewTerm(func=reward_stability_on_block, weight=0.6,
                                     params={"normalise": True, "normaliser_name": "stability_on_block_reward"})