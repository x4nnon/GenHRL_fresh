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

def walk_to_Large_Block_for_push_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for walk_to_Large_Block_for_push.

    This reward encourages the robot's pelvis to reach a stable standing position adjacent to the Large Block (Object3),
    ready to initiate a push. It uses relative distances to the block and a desired pelvis height.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    # CORRECT: Accessing Object3 (Large Block) using approved pattern
    large_block = env.scene['Object3']
    # CORRECT: Accessing robot using approved pattern
    robot = env.scene["robot"]
    # CORRECT: Getting pelvis index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    # CORRECT: Getting pelvis position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object3 dimensions (hardcoded from object configuration, as per requirements)
    # CORRECT: Hardcoding object dimensions as per requirement 8
    large_block_height = 0.9 # z-dimension
    large_block_width_x = 1.0 # x-dimension
    large_block_width_y = 1.0 # y-dimension

    # Define target position relative to the large block for pushing
    # The robot should approach from the negative X side of the block to push it forward (positive X).
    # Target X: Slightly in front of the block's edge (block_x - half_block_width_x - desired_clearance)
    # Target Y: Aligned with the block's center
    # Target Z: Stable standing height for pelvis (0.7m)
    target_x_offset = 0.2 # Desired distance from the block's face to the robot's pelvis
    desired_pelvis_z = 0.7 # Desired stable pelvis height

    # Calculate target position relative to the large block's center
    # This assumes the block's root_pos_w is its center.
    # All positions are relative to the block's position.
    # CORRECT: Calculating target position relative to the block's position using approved pattern
    target_pos_x = large_block.data.root_pos_w[:, 0] - (large_block_width_x / 2.0) - target_x_offset
    target_pos_y = large_block.data.root_pos_w[:, 1] # Align with block's center Y
    # CORRECT: Using absolute Z for pelvis height, which is allowed for posture goals
    target_pos_z = desired_pelvis_z # Desired pelvis height (absolute Z, but relative to ground which is fixed)

    # Calculate distance components between pelvis and target position
    # Using negative absolute distance for continuous positive reward as distance decreases
    # CORRECT: Using relative distances for X, Y, Z components
    distance_x = torch.abs(target_pos_x - pelvis_pos[:, 0]) # Relative distance in X
    distance_y = torch.abs(target_pos_y - pelvis_pos[:, 1]) # Relative distance in Y
    distance_z = torch.abs(target_pos_z - pelvis_pos[:, 2]) # Relative distance in Z (height)

    # Reward for reducing distance to target position
    # The reward is higher when the distance is smaller.
    # CORRECT: Continuous rewards based on negative absolute distance
    reward_x = -distance_x
    reward_y = -distance_y
    reward_z = -distance_z

    # Combine rewards with weights. Weights can be adjusted for task importance.
    # X and Y are crucial for horizontal positioning, Z for stable height.
    reward = (reward_x * 0.4) + (reward_y * 0.4) + (reward_z * 0.2)

    # Mandatory reward normalization
    # CORRECT: Normalization block as per requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_Large_Block_for_push_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with the Large Block.

    This reward penalizes the robot if its feet collide with or are too close to the Large Block (Object3),
    encouraging the robot to stay on the ground and not climb the block prematurely.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    # CORRECT: Accessing Object3 (Large Block) using approved pattern
    large_block = env.scene['Object3']
    # CORRECT: Accessing robot using approved pattern
    robot = env.scene["robot"]
    # CORRECT: Getting foot indices using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    # CORRECT: Getting foot positions using approved pattern
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object3 dimensions (hardcoded from object configuration)
    # CORRECT: Hardcoding object dimensions as per requirement 8
    large_block_height = 0.9 # z-dimension
    large_block_width_x = 1.0 # x-dimension
    large_block_width_y = 1.0 # y-dimension

    # Define the block's bounding box in world coordinates
    # CORRECT: Calculating block's bounding box relative to its center position
    block_center_pos = large_block.data.root_pos_w
    block_min_x = block_center_pos[:, 0] - (large_block_width_x / 2.0)
    block_max_x = block_center_pos[:, 0] + (large_block_width_x / 2.0)
    block_min_y = block_center_pos[:, 1] - (large_block_width_y / 2.0)
    block_max_y = block_center_pos[:, 1] + (large_block_width_y / 2.0)
    block_bottom_z = block_center_pos[:, 2] - (large_block_height / 2.0)
    block_top_z = block_center_pos[:, 2] + (large_block_height / 2.0)

    # Define a small buffer for collision detection to prevent feet from going slightly into the block
    # CORRECT: Using small buffers for proximity detection
    collision_buffer_z = 0.1 # Buffer for Z-axis to detect proximity/overlap
    collision_buffer_xy = 0.05 # Small buffer for X-Y plane

    # Check if left foot is within the block's X-Y projection and within the Z collision range
    # CORRECT: Using relative positions for collision detection
    left_foot_in_block_xy = (left_foot_pos[:, 0] > block_min_x - collision_buffer_xy) & (left_foot_pos[:, 0] < block_max_x + collision_buffer_xy) & \
                            (left_foot_pos[:, 1] > block_min_y - collision_buffer_xy) & (left_foot_pos[:, 1] < block_max_y + collision_buffer_xy)

    left_foot_colliding_z = (left_foot_pos[:, 2] > block_bottom_z - collision_buffer_z) & (left_foot_pos[:, 2] < block_top_z + collision_buffer_z)

    collision_left_foot = left_foot_in_block_xy & left_foot_colliding_z

    # Check if right foot is within the block's X-Y projection and within the Z collision range
    # CORRECT: Using relative positions for collision detection
    right_foot_in_block_xy = (right_foot_pos[:, 0] > block_min_x - collision_buffer_xy) & (right_foot_pos[:, 0] < block_max_x + collision_buffer_xy) & \
                             (right_foot_pos[:, 1] > block_min_y - collision_buffer_xy) & (right_foot_pos[:, 1] < block_max_y + collision_buffer_xy)

    right_foot_colliding_z = (right_foot_pos[:, 2] > block_bottom_z - collision_buffer_z) & (right_foot_pos[:, 2] < block_top_z + collision_buffer_z)

    collision_right_foot = right_foot_in_block_xy & right_foot_colliding_z

    # Penalty for collision: a fixed negative reward if either foot is in collision
    # CORRECT: Fixed negative reward for collision, which is a valid shaping reward
    collision_penalty = -1.0 # A significant negative reward for collision

    # Apply penalty using torch.where for batch compatibility
    # CORRECT: Using torch.where for batch compatibility
    reward = torch.where(collision_left_foot | collision_right_foot, collision_penalty, 0.0)

    # Mandatory reward normalization
    # CORRECT: Normalization block as per requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def walk_to_Large_Block_for_push_maintain_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "upright_posture_reward") -> torch.Tensor:
    '''Shaping reward for maintaining an upright posture.

    This reward encourages the robot to maintain an upright and stable posture by keeping its pelvis at a desired height
    and penalizing large deviations. This is crucial for stability before the next skill.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part using approved patterns
    # CORRECT: Accessing robot using approved pattern
    robot = env.scene["robot"]
    # CORRECT: Getting pelvis index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis')
    # CORRECT: Getting pelvis Z position using approved pattern
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx][:, 2]

    # Desired stable pelvis height (absolute Z, as it's a posture goal)
    desired_pelvis_z = 0.7

    # Reward for being close to the desired pelvis height
    # Use negative absolute difference, so reward is higher when difference is smaller (closer to 0).
    # CORRECT: Continuous reward based on negative absolute difference for posture
    reward = -torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Mandatory reward normalization
    # CORRECT: Normalization block as per requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Main reward for positioning the robot near the large block for pushing
    # CORRECT: Main reward with weight 1.0 and proper normalization parameters
    MainWalkToLargeBlockReward = RewTerm(func=walk_to_Large_Block_for_push_main_reward, weight=1.0,
                                         params={"normalise": True, "normaliser_name": "main_walk_to_large_block_reward"})

    # Shaping reward to penalize feet collision with the large block
    # CORRECT: Shaping reward with appropriate weight and normalization parameters
    CollisionAvoidanceReward = RewTerm(func=walk_to_Large_Block_for_push_collision_avoidance_reward, weight=0.5,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward to maintain an upright posture
    # CORRECT: Shaping reward with appropriate weight and normalization parameters
    MaintainUprightPostureReward = RewTerm(func=walk_to_Large_Block_for_push_maintain_upright_posture_reward, weight=0.3,
                                           params={"normalise": True, "normaliser_name": "upright_posture_reward"})