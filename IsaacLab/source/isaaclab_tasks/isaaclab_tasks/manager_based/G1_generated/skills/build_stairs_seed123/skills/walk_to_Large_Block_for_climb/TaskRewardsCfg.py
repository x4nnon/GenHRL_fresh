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


def walk_to_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_large_block_reward") -> torch.Tensor:
    '''
    Primary reward for the robot to walk to a specific position in front of the Large Block (Object3).
    The target position is defined relative to the Large Block's center, ensuring the robot is ready to climb.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    # Requirement: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    large_block = env.scene['Object3']
    robot = env.scene["robot"] # Requirement: Access robot object
    # Requirement: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object3 dimensions (from description: x=1m y=1m z=0.9m)
    # Requirement: There is no way to access the SIZE of an object. Hardcode values from object config.
    large_block_half_x = 0.5
    large_block_half_y = 0.5
    large_block_height = 0.9

    # Target position relative to large_block's center
    # Target X: 0.6m in front of the block's face. Block's face is at large_block.x - large_block_half_x.
    # So, target_x_offset = large_block_half_x + 0.6 = 0.5 + 0.6 = 1.1m
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Requirement: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    target_pelvis_x = large_block.data.root_pos_w[:, 0] - (large_block_half_x + 0.6)
    target_pelvis_y = large_block.data.root_pos_w[:, 1]
    target_pelvis_z = 0.7 # Stable standing height, this is an absolute Z position, which is allowed for height goals.

    # Calculate distances to target
    # Requirement: Use absolute distances for components.
    dist_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    dist_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine distances for primary reward
    # Use negative exponential or inverse for continuous positive reward as distance decreases
    # Requirement: Rewards should be continuous and positive (or negative for penalties).
    # Using negative sum of distances to encourage reduction.
    reward = -(dist_x + dist_y + dist_z)

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_object3_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_object3_reward") -> torch.Tensor:
    '''
    Shaping reward that penalizes the robot if its pelvis or feet get too close to or inside Object3.
    This ensures it positions itself *in front* of the block rather than on top or through it.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    large_block = env.scene['Object3']
    robot = env.scene["robot"] # Requirement: Access robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object3 dimensions (x=1m y=1m z=0.9m)
    # Requirement: Hardcode dimensions from object configuration.
    large_block_half_x = 0.5
    large_block_half_y = 0.5
    large_block_height = 0.9

    # Define collision boundaries relative to Object3's center
    # Assuming block.data.root_pos_w is the center of the block.
    block_min_x = large_block.data.root_pos_w[:, 0] - large_block_half_x
    block_max_x = large_block.data.root_pos_w[:, 0] + large_block_half_x
    block_min_y = large_block.data.root_pos_w[:, 1] - large_block_half_y
    block_max_y = large_block.data.root_pos_w[:, 1] + large_block_half_y
    block_min_z = large_block.data.root_pos_w[:, 2] - (large_block_height / 2.0)
    block_max_z = large_block.data.root_pos_w[:, 2] + (large_block_height / 2.0)

    # Check for pelvis collision
    # Requirement: All operations must work with batched environments.
    pelvis_collision_x = (pelvis_pos[:, 0] > block_min_x) & (pelvis_pos[:, 0] < block_max_x)
    pelvis_collision_y = (pelvis_pos[:, 1] > block_min_y) & (pelvis_pos[:, 1] < block_max_y)
    pelvis_collision_z = (pelvis_pos[:, 2] > block_min_z) & (pelvis_pos[:, 2] < block_max_z + 0.1) # Small buffer above block

    pelvis_colliding = pelvis_collision_x & pelvis_collision_y & pelvis_collision_z

    # Check for feet collision
    left_foot_collision_x = (left_foot_pos[:, 0] > block_min_x) & (left_foot_pos[:, 0] < block_max_x)
    left_foot_collision_y = (left_foot_pos[:, 1] > block_min_y) & (left_foot_pos[:, 1] < block_max_y)
    left_foot_collision_z = (left_foot_pos[:, 2] > block_min_z) & (left_foot_pos[:, 2] < block_max_z + 0.1)

    right_foot_collision_x = (right_foot_pos[:, 0] > block_min_x) & (right_foot_pos[:, 0] < block_max_x)
    right_foot_collision_y = (right_foot_pos[:, 1] > block_min_y) & (right_foot_pos[:, 1] < block_max_y)
    right_foot_collision_z = (right_foot_pos[:, 2] > block_min_z) & (right_foot_pos[:, 2] < block_max_z + 0.1)

    feet_colliding = (left_foot_collision_x & left_foot_collision_y & left_foot_collision_z) | \
                     (right_foot_collision_x & right_foot_collision_y & right_foot_collision_z)

    # Combine collision conditions
    is_colliding_with_block3 = pelvis_colliding | feet_colliding

    # Apply negative reward if colliding
    # Requirement: Rewards should be continuous (or penalizing).
    reward = torch.where(is_colliding_with_block3, -10.0, 0.0)

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_other_blocks_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_other_blocks_reward") -> torch.Tensor:
    '''
    Shaping reward that penalizes the robot if it collides with Object1 (Small Block) or Object2 (Medium Block)
    while navigating to Object3. This ensures the robot takes a clear path.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    small_block = env.scene['Object1']
    medium_block = env.scene['Object2']
    robot = env.scene["robot"] # Requirement: Access robot object

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object1 dimensions (x=1m y=1m z=0.3m)
    # Requirement: Hardcode dimensions from object configuration.
    small_block_half_x = 0.5
    small_block_half_y = 0.5
    small_block_height = 0.3

    # Object2 dimensions (x=1m y=1m z=0.6m)
    # Requirement: Hardcode dimensions from object configuration.
    medium_block_half_x = 0.5
    medium_block_half_y = 0.5
    medium_block_height = 0.6

    def check_collision_with_block(robot_part_pos, block_obj, block_half_x, block_half_y, block_height):
        '''Helper function to check if a robot part is inside a given block's bounding box.'''
        # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
        block_min_x = block_obj.data.root_pos_w[:, 0] - block_half_x
        block_max_x = block_obj.data.root_pos_w[:, 0] + block_half_x
        block_min_y = block_obj.data.root_pos_w[:, 1] - block_half_y
        block_max_y = block_obj.data.root_pos_w[:, 1] + block_half_y
        block_min_z = block_obj.data.root_pos_w[:, 2] - (block_height / 2.0)
        block_max_z = block_obj.data.root_pos_w[:, 2] + (block_height / 2.0)

        colliding_x = (robot_part_pos[:, 0] > block_min_x) & (robot_part_pos[:, 0] < block_max_x)
        colliding_y = (robot_part_pos[:, 1] > block_min_y) & (robot_part_pos[:, 1] < block_max_y)
        colliding_z = (robot_part_pos[:, 2] > block_min_z) & (robot_part_pos[:, 2] < block_max_z + 0.1) # Small buffer

        return colliding_x & colliding_y & colliding_z

    # Check collision for Object1 (Small Block)
    pelvis_colliding_obj1 = check_collision_with_block(pelvis_pos, small_block, small_block_half_x, small_block_half_y, small_block_height)
    left_foot_colliding_obj1 = check_collision_with_block(left_foot_pos, small_block, small_block_half_x, small_block_half_y, small_block_height)
    right_foot_colliding_obj1 = check_collision_with_block(right_foot_pos, small_block, small_block_half_x, small_block_half_y, small_block_height)
    is_colliding_with_obj1 = pelvis_colliding_obj1 | left_foot_colliding_obj1 | right_foot_colliding_obj1

    # Check collision for Object2 (Medium Block)
    pelvis_colliding_obj2 = check_collision_with_block(pelvis_pos, medium_block, medium_block_half_x, medium_block_half_y, medium_block_height)
    left_foot_colliding_obj2 = check_collision_with_block(left_foot_pos, medium_block, medium_block_half_x, medium_block_half_y, medium_block_height)
    right_foot_colliding_obj2 = check_collision_with_block(right_foot_pos, medium_block, medium_block_half_x, medium_block_half_y, medium_block_height)
    is_colliding_with_obj2 = pelvis_colliding_obj2 | left_foot_colliding_obj2 | right_foot_colliding_obj2

    # Combine collision conditions for other blocks
    is_colliding_with_other_blocks = is_colliding_with_obj1 | is_colliding_with_obj2

    # Apply negative reward if colliding
    reward = torch.where(is_colliding_with_other_blocks, -5.0, 0.0)

    # Requirement: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Requirement: Main reward with weight ~1.0
    WalkToLargeBlockReward = RewTerm(func=walk_to_large_block_reward, weight=1.0,
                                     params={"normalise": True, "normaliser_name": "walk_to_large_block_reward"})

    # Requirement: Supporting rewards with lower weights (<1.0)
    CollisionAvoidanceObject3Reward = RewTerm(func=collision_avoidance_object3_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "collision_avoidance_object3_reward"})

    CollisionAvoidanceOtherBlocksReward = RewTerm(func=collision_avoidance_other_blocks_reward, weight=0.4,
                                                  params={"normalise": True, "normaliser_name": "collision_avoidance_other_blocks_reward"})