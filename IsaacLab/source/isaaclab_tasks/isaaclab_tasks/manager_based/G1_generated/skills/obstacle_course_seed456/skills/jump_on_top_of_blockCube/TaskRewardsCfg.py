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

def main_jump_on_top_of_blockCube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for jump_on_top_of_blockCube.

    This reward encourages the robot to land stably on top of the block cube (Object5).
    It combines rewards for horizontal alignment of feet, vertical alignment of feet, and stable pelvis height.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    block_cube = env.scene['Object5'] # Accessing Object5 (block cube) directly as per requirements
    block_cube_pos = block_cube.data.root_pos_w # Accessing object position using approved pattern

    # Access the required robot part(s)
    robot = env.scene["robot"] # Accessing robot object
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Object5 (block cube) is 0.5m cubed. Assuming root is at center, top surface is at z + 0.25m.
    # Hardcoding dimensions from object configuration as per requirements
    block_half_size = 0.25 # 0.5m / 2
    block_top_z = block_cube_pos[:, 2] + block_half_size # Relative distance: block top Z is block root Z + half size
    block_center_x = block_cube_pos[:, 0]
    block_center_y = block_cube_pos[:, 1]

    # Reward for feet being on top of the block (horizontal alignment)
    # Use average of both feet for horizontal position
    avg_foot_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2

    # Distance from average foot horizontal position to block center
    dist_feet_x_to_block_center = torch.abs(avg_foot_x - block_center_x) # Relative distance in x
    dist_feet_y_to_block_center = torch.abs(avg_foot_y - block_center_y) # Relative distance in y

    # Reward for feet being within the horizontal bounds of the block
    # Use a negative exponential decay for distance outside the block, and a small penalty for being off-center inside.
    # This encourages being centered and within bounds.
    horizontal_alignment_reward_x = torch.exp(-5.0 * torch.max(torch.tensor(0.0, device=env.device), dist_feet_x_to_block_center - block_half_size)) \
                                  - 0.5 * dist_feet_x_to_block_center # Penalize being off-center even if within bounds
    horizontal_alignment_reward_y = torch.exp(-5.0 * torch.max(torch.tensor(0.0, device=env.device), dist_feet_y_to_block_center - block_half_size)) \
                                  - 0.5 * dist_feet_y_to_block_center # Penalize being off-center even if within bounds
    horizontal_alignment_reward = horizontal_alignment_reward_x + horizontal_alignment_reward_y

    # Reward for feet being at the correct height (vertical alignment)
    # Target height for feet is block_top_z. Use negative absolute difference.
    dist_left_foot_z_to_block_top = torch.abs(left_foot_pos[:, 2] - block_top_z) # Relative distance in z
    dist_right_foot_z_to_block_top = torch.abs(right_foot_pos[:, 2] - block_top_z) # Relative distance in z
    vertical_alignment_reward = -(dist_left_foot_z_to_block_top + dist_right_foot_z_to_block_top) / 2.0 # Continuous reward

    # Reward for pelvis being at a stable standing height relative to the block's top surface
    # Target pelvis height relative to block top: 0.7m (approx. default standing height for robot)
    target_pelvis_z_relative_to_block = block_top_z + 0.7 # Relative target Z position
    dist_pelvis_z_to_target = torch.abs(pelvis_pos[:, 2] - target_pelvis_z_relative_to_block) # Relative distance in z
    pelvis_height_reward = -dist_pelvis_z_to_target # Continuous reward

    # Combine rewards
    reward = horizontal_alignment_reward + vertical_alignment_reward + pelvis_height_reward

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_block_x_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_reward") -> torch.Tensor:
    '''Shaping reward 1: Encourages the robot to approach Object5 (block cube) along the x-axis,
    positioning itself in front of it for the jump.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    block_cube = env.scene['Object5'] # Accessing Object5 (block cube) directly
    block_cube_pos = block_cube.data.root_pos_w # Accessing object position

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position
    pelvis_pos_x = pelvis_pos[:, 0]

    # Object5 (block cube) is 0.5m cubed.
    # Hardcoding dimensions from object configuration
    block_half_size_x = 0.25
    block_front_x = block_cube_pos[:, 0] - block_half_size_x # Relative position: front edge of the block

    # Reward for approaching the block along the x-axis
    # Target approach distance: 0.5m in front of the block's front edge
    target_approach_x = block_front_x - 0.5 # Relative target X position

    # Reward is higher (less negative) as pelvis_pos_x gets closer to target_approach_x
    approach_reward = -torch.abs(pelvis_pos_x - target_approach_x) # Continuous reward based on relative distance

    # Activation condition: Robot is still behind the block's front edge and not yet on top.
    # This prevents the reward from interfering once the robot is jumping or on the block.
    # A small buffer (e.g., 0.1m) is added to block_front_x to ensure the robot is clearly behind it.
    activation_condition = (pelvis_pos_x < block_front_x - 0.1) # Relative condition

    reward = torch.where(activation_condition, approach_reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def jump_height_over_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_height_reward") -> torch.Tensor:
    '''Shaping reward 2: Encourages the robot to gain sufficient vertical height during the jump to clear Object5.
    It rewards the pelvis for being above the top surface of the block, with a small clearance.
    This reward is active when the robot is horizontally aligned with the block (i.e., over the block)
    but not yet landed (feet are in the air).
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    block_cube = env.scene['Object5'] # Accessing Object5 (block cube) directly
    block_cube_pos = block_cube.data.root_pos_w # Accessing object position

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position

    # Object5 (block cube) is 0.5m cubed.
    # Hardcoding dimensions from object configuration
    block_half_size_x = 0.25
    block_top_z = block_cube_pos[:, 2] + block_half_size_x # Relative distance: block top Z is block root Z + half size
    block_center_x = block_cube_pos[:, 0]

    # Target height for pelvis during jump (e.g., 0.2m above block top)
    target_jump_pelvis_z = block_top_z + 0.2 # Relative target Z position

    # Reward for pelvis being above the block's top surface
    # Reward is positive when pelvis_pos_z is greater than block_top_z, encouraging clearance.
    jump_height_reward = torch.where(pelvis_pos_z > block_top_z, pelvis_pos_z - block_top_z, torch.tensor(0.0, device=env.device)) # Continuous reward

    # Activation condition: Robot's pelvis is horizontally over the block (within x-bounds)
    # AND both feet are above the block's top surface (indicating it's in the air, not landed)
    horizontal_over_block_condition = (pelvis_pos_x > block_center_x - block_half_size_x) & \
                                      (pelvis_pos_x < block_center_x + block_half_size_x) # Relative condition

    # Feet in air condition: feet are above the block's top surface with a small buffer (0.05m)
    feet_in_air_condition = (left_foot_pos[:, 2] > block_top_z + 0.05) & \
                            (right_foot_pos[:, 2] > block_top_z + 0.05) # Relative condition

    activation_condition = horizontal_over_block_condition & feet_in_air_condition

    reward = torch.where(activation_condition, jump_height_reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Main reward for landing on top of the block
    MainJumpOnTopReward = RewTerm(func=main_jump_on_top_of_blockCube_reward, weight=1.0,
                                  params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for approaching the block along the x-axis
    ApproachBlockXReward = RewTerm(func=approach_block_x_reward, weight=0.4,
                                   params={"normalise": True, "normaliser_name": "approach_reward"})

    # Shaping reward for gaining jump height over the block
    JumpHeightOverBlockReward = RewTerm(func=jump_height_over_block_reward, weight=0.3,
                                        params={"normalise": True, "normaliser_name": "jump_height_reward"})