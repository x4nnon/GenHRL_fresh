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


def main_stabilize_on_blockCube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_stabilize_reward") -> torch.Tensor:
    '''Main reward for stabilize_on_blockCube.

    This reward encourages the robot to achieve a stable standing posture on top of the block cube.
    It combines rewards for feet being on the block's top surface and within its horizontal bounds,
    and for the pelvis being at an appropriate upright height relative to the block.
    '''
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (mandatory: env.scene['ObjectName'].data.root_pos_w)
    block_cube = env.scene['Object5']
    block_cube_pos = block_cube.data.root_pos_w

    # Access the required robot parts (mandatory: robot.data.body_pos_w[:, robot.body_names.index('part_name')])
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object5 (block cube) dimensions: 0.5m cubed. Root is at center.
    # Hardcode dimensions from object configuration (mandatory: no data.size or similar)
    block_half_size_xy = 0.25  # Half of 0.5m side length for X and Y
    block_half_size_z = 0.25   # Half of 0.5m side length for Z

    # 1. Feet Z-position reward: Encourage feet to be on the block's top surface
    # Target Z for feet is block_top_z (relative distance calculation)
    block_top_z = block_cube_pos[:, 2] + block_half_size_z
    # Using negative absolute difference for continuous reward, closer to 0 is better
    reward_feet_z = -torch.abs(left_foot_pos[:, 2] - block_top_z) - torch.abs(right_foot_pos[:, 2] - block_top_z)

    # 2. Feet XY-position reward: Encourage feet to be within the block's horizontal bounds
    # Calculate distances from block center in X and Y (relative distance calculation)
    left_foot_dist_x = torch.abs(left_foot_pos[:, 0] - block_cube_pos[:, 0])
    left_foot_dist_y = torch.abs(left_foot_pos[:, 1] - block_cube_pos[:, 1])
    right_foot_dist_x = torch.abs(right_foot_pos[:, 0] - block_cube_pos[:, 0])
    right_foot_dist_y = torch.abs(right_foot_pos[:, 1] - block_cube_pos[:, 1])

    # Threshold for being "on" the block, slightly larger than half_size to allow for foot width
    # (mandatory: no arbitrary thresholds, derived from object size + buffer)
    xy_threshold = block_half_size_xy - 0.1 # Add a small buffer for foot size

    # Penalize if feet are outside the block's XY bounds (continuous penalty)
    reward_feet_xy = -torch.sqrt(left_foot_dist_x**2 + left_foot_dist_y**2) - torch.sqrt(right_foot_dist_x**2 + right_foot_dist_y**2)

    # 3. Pelvis Z-position reward: Encourage upright posture relative to block
    # Target pelvis Z is block_top_z + 0.7 (stable standing height, relative to block top)
    target_pelvis_z = block_top_z + 0.7
    # Using negative absolute difference for continuous reward
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    reward_pelvis_xy = -torch.abs(pelvis_pos[:, 0] - block_cube_pos[:, 0]) - torch.abs(pelvis_pos[:, 1] - block_cube_pos[:, 1])

    # Combine primary rewards
    reward = 2.0*reward_feet_xy + 2.0*reward_pelvis_xy

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_over_feet_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_over_feet_reward") -> torch.Tensor:
    '''Shaping reward 1: Encourages the robot to keep its pelvis horizontally centered above its feet.

    This promotes balance and an upright stance, preventing the robot from leaning too far.
    '''
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts (mandatory: robot.data.body_pos_w[:, robot.body_names.index('part_name')])
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Calculate the midpoint between the two feet in X and Y (relative distance calculation)
    feet_midpoint_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2.0
    feet_midpoint_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2.0

    # Reward for pelvis X and Y being close to the feet midpoint (continuous reward)
    # Using negative absolute difference, closer to 0 is better
    reward = -torch.abs(pelvis_pos[:, 0] - feet_midpoint_x) - torch.abs(pelvis_pos[:, 1] - feet_midpoint_y)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_block_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_collision_reward") -> torch.Tensor:
    '''Shaping reward 2: Encourages collision avoidance between the robot's pelvis and the block cube.

    This ensures the robot doesn't clip into the block while stabilizing.
    '''
    # Get normalizer instance (mandatory)
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects (mandatory: env.scene['ObjectName'].data.root_pos_w)
    block_cube = env.scene['Object5']
    block_cube_pos = block_cube.data.root_pos_w

    # Access the required robot part (mandatory: robot.data.body_pos_w[:, robot.body_names.index('part_name')])
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Block cube dimensions: 0.5m cubed. Hardcode from object configuration.
    block_half_size_x = 0.25
    block_half_size_y = 0.25
    block_half_size_z = 0.25

    # Calculate distances from pelvis to block's center in each dimension (relative distance calculation)
    dist_x = torch.abs(pelvis_pos[:, 0] - block_cube_pos[:, 0])
    dist_y = torch.abs(pelvis_pos[:, 1] - block_cube_pos[:, 1])
    dist_z = torch.abs(pelvis_pos[:, 2] - block_cube_pos[:, 2])

    # Define a safe distance threshold (block_half_size + small buffer for robot body)
    # (mandatory: no arbitrary thresholds, derived from object size + buffer)
    safe_dist_x = block_half_size_x + 0.1 # Add a buffer for pelvis size
    safe_dist_y = block_half_size_y + 0.1
    safe_dist_z = block_half_size_z + 0.1

    # Penalize if pelvis is too close or inside the block's volume (continuous penalty)
    # The reward is negative when collision is imminent/occurring
    reward_collision = 0.0
    reward_collision -= torch.max(torch.tensor(0.0, device=env.device), safe_dist_x - dist_x) * 2.0
    reward_collision -= torch.max(torch.tensor(0.0, device=env.device), safe_dist_y - dist_y) * 2.0
    reward_collision -= torch.max(torch.tensor(0.0, device=env.device), safe_dist_z - dist_z) * 2.0

    reward = reward_collision

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
    # Main reward for stabilizing on the block cube
    MainStabilizeOnBlockCubeReward = RewTerm(func=main_stabilize_on_blockCube_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_stabilize_reward"})

    # Shaping reward for keeping pelvis centered over feet
    PelvisOverFeetReward = RewTerm(func=pelvis_over_feet_reward, weight=0.0,
                                   params={"normalise": True, "normaliser_name": "pelvis_over_feet_reward"})

    # Shaping reward for avoiding collision between pelvis and block
    PelvisBlockCollisionAvoidanceReward = RewTerm(func=pelvis_block_collision_avoidance_reward, weight=0.0,
                                                  params={"normalise": True, "normaliser_name": "pelvis_collision_reward"})