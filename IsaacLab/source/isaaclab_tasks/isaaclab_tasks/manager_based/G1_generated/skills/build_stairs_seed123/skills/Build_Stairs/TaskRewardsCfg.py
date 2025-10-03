# CORRECT: Standard imports - do not modify these, do not add any other imports or import reward stats from anywhere else, it is handled in get_normalizer
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats # this automatically sets up the RewNormalizer instance.
from genhrl.generation.objects import get_object_volume
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv # Corrected: Added ManagerBasedRLEnv import
import torch

from isaaclab.envs import mdp
# Import custom MDP functions from genhrl
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

# Get normalizer instance is generated in the import.
RewNormalizer = get_normalizer(torch.device("cuda")) # Initialize with a default device, will be updated by env.device

def build_stairs_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "build_stairs_main_reward") -> torch.Tensor:
    """
    Primary reward for arranging the three blocks (Small, Medium, Large) into a stair-like configuration
    and the robot ending near the base of the stairs.
    """
    # Ensure RewNormalizer is initialized for the correct device
    global RewNormalizer
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    # Object1: Small Block (1x1x0.3m)
    # Object2: Medium Block (1x1x0.6m)
    # Object3: Large Block (1x1x0.9m)
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # CORRECT: Access the robot object
    robot = env.scene["robot"]
    # Access the required robot part(s) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define target relative offsets for stairs based on block dimensions (1m width)
    # Reward design plan specifies 1m in X and 0.5m in Y.
    target_x_offset = 1.0 # CORRECT: Hardcoded from reward design plan, based on block dimensions
    target_y_offset = 0.5 # CORRECT: Hardcoded from reward design plan, based on block dimensions

    # Calculate distances for Object2 relative to Object1
    # Reward for Object2 being correctly placed relative to Object1
    # Using absolute differences for continuous reward, penalizing deviation from target offsets.
    dist_obj2_obj1_x = object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0]
    dist_obj2_obj1_y = object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1]
    dist_obj2_obj1_z = object2.data.root_pos_w[:, 2] - object1.data.root_pos_w[:, 2]
    # CORRECT: Continuous reward based on relative distances
    reward_obj2_placement = -torch.abs(dist_obj2_obj1_x - target_x_offset) \
                            -torch.abs(dist_obj2_obj1_y - target_y_offset) \
                            -torch.abs(dist_obj2_obj1_z) # Z should be near 0 for blocks on ground

    # Calculate distances for Object3 relative to Object2
    # Reward for Object3 being correctly placed relative to Object2
    dist_obj3_obj2_x = object3.data.root_pos_w[:, 0] - object2.data.root_pos_w[:, 0]
    dist_obj3_obj2_y = object3.data.root_pos_w[:, 1] - object2.data.root_pos_w[:, 1]
    dist_obj3_obj2_z = object3.data.root_pos_w[:, 2] - object2.data.root_pos_w[:, 2]
    # CORRECT: Continuous reward based on relative distances
    reward_obj3_placement = -torch.abs(dist_obj3_obj2_x - target_x_offset) \
                            -torch.abs(dist_obj3_obj2_y - target_y_offset) \
                            -torch.abs(dist_obj3_obj2_z) # Z should be near 0 for blocks on ground

    # Reward for robot pelvis being at a stable height (0.7m)
    # This encourages a stable posture for the robot.
    # CORRECT: Uses absolute Z position for pelvis height, which is allowed sparingly for height.
    reward_pelvis_height = -torch.abs(pelvis_pos[:, 2] - 0.7) # CORRECT: Hardcoded target height from reward design plan

    # Reward for robot pelvis being near the base of the stairs (Object1)
    # This encourages the robot to finish near the start of the stairs for the next skill (climbing).
    # Robot should be slightly behind Object1 in Y and centered in X relative to Object1.
    # CORRECT: Robot target position is relative to Object1's position.
    robot_target_x_relative_to_obj1 = object1.data.root_pos_w[:, 0]
    robot_target_y_relative_to_obj1 = object1.data.root_pos_w[:, 1] - 1.0 # 1m behind Object1 (hardcoded from plan)
    # CORRECT: Continuous reward based on relative distances
    reward_robot_final_pos = -torch.abs(pelvis_pos[:, 0] - robot_target_x_relative_to_obj1) \
                             -torch.abs(pelvis_pos[:, 1] - robot_target_y_relative_to_obj1)

    # Combine all primary rewards
    reward = reward_obj2_placement + reward_obj3_placement + reward_pelvis_height + reward_robot_final_pos

    # Mandatory reward normalization
    # CORRECT: Normalization block
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_and_push_blocks_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_and_push_blocks_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach each block in sequence (Small, Medium, Large)
    and get close enough to push it. Provides a gradient for reducing the distance between the robot's
    pelvis and the current target block.
    """
    # Ensure RewNormalizer is initialized for the correct device
    global RewNormalizer
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # CORRECT: Access the robot object
    robot = env.scene["robot"]
    # Access the required robot part(s)
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define target relative offsets for stairs (same as primary reward for consistency)
    target_x_offset = 1.0 # CORRECT: Hardcoded from reward design plan
    target_y_offset = 0.5 # CORRECT: Hardcoded from reward design plan

    # Conditions for "in place" for each block, based on the primary reward's target relative positions.
    # These conditions are used to activate the sequential approach reward.
    # A tolerance of 0.5m is used for "roughly in place".
    # CORRECT: Condition for Object1 not placed, using relative distance from origin (robot starts at origin, blocks 4m away)
    # This is a proxy for "not yet pushed from initial triangle position".
    condition_obj1_not_placed = (torch.abs(object1.data.root_pos_w[:, 0]) > 1.0) | (torch.abs(object1.data.root_pos_w[:, 1] - 4.0) > 1.0)
    
    # Check if Object2 is not yet placed relative to Object1
    dist_obj2_obj1_x = object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0]
    dist_obj2_obj1_y = object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1]
    dist_obj2_obj1_z = object2.data.root_pos_w[:, 2] - object1.data.root_pos_w[:, 2]
    # CORRECT: Condition for Object2 not placed, using relative distances to Object1 and hardcoded tolerance
    condition_obj2_not_placed = (torch.abs(dist_obj2_obj1_x - target_x_offset) > 0.5) | \
                                (torch.abs(dist_obj2_obj1_y - target_y_offset) > 0.5) | \
                                (torch.abs(dist_obj2_obj1_z) > 0.3) # Z tolerance for being on the same plane

    # Check if Object3 is not yet placed relative to Object2
    dist_obj3_obj2_x = object3.data.root_pos_w[:, 0] - object2.data.root_pos_w[:, 0]
    dist_obj3_obj2_y = object3.data.root_pos_w[:, 1] - object2.data.root_pos_w[:, 1]
    dist_obj3_obj2_z = object3.data.root_pos_w[:, 2] - object2.data.root_pos_w[:, 2]
    # CORRECT: Condition for Object3 not placed, using relative distances to Object2 and hardcoded tolerance
    condition_obj3_not_placed = (torch.abs(dist_obj3_obj2_x - target_x_offset) > 0.5) | \
                                (torch.abs(dist_obj3_obj2_y - target_y_offset) > 0.5) | \
                                (torch.abs(dist_obj3_obj2_z) > 0.3) # Z tolerance for being on the same plane

    # Calculate XY distances from pelvis to each object
    # CORRECT: Uses torch.norm for XY distance, which is continuous.
    dist_pelvis_obj1_xy = torch.norm(pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2], dim=1)
    dist_pelvis_obj2_xy = torch.norm(pelvis_pos[:, :2] - object2.data.root_pos_w[:, :2], dim=1)
    dist_pelvis_obj3_xy = torch.norm(pelvis_pos[:, :2] - object3.data.root_pos_w[:, :2], dim=1)

    # Reward for approaching each object (negative distance, so closer is higher reward)
    reward_approach_obj1 = -dist_pelvis_obj1_xy
    reward_approach_obj2 = -dist_pelvis_obj2_xy
    reward_approach_obj3 = -dist_pelvis_obj3_xy

    # Combine approach rewards, prioritizing the current target block sequentially.
    # This ensures the robot focuses on one block at a time.
    reward = torch.zeros_like(dist_pelvis_obj1_xy) # Initialize reward tensor

    # CORRECT: Sequential activation of rewards using torch.where for batch processing
    # If Object1 is not placed, reward approaching Object1
    reward = torch.where(condition_obj1_not_placed, reward_approach_obj1, reward)
    # If Object1 is placed but Object2 is not, reward approaching Object2
    reward = torch.where(~condition_obj1_not_placed & condition_obj2_not_placed, reward_approach_obj2, reward)
    # If Object1 and Object2 are placed but Object3 is not, reward approaching Object3
    reward = torch.where(~condition_obj1_not_placed & ~condition_obj2_not_placed & condition_obj3_not_placed, reward_approach_obj3, reward)

    # Mandatory reward normalization
    # CORRECT: Normalization block
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to penalize collisions between the robot's body parts and the blocks,
    and also between the blocks themselves (undesired interpenetration).
    """
    # Ensure RewNormalizer is initialized for the correct device
    global RewNormalizer
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # CORRECT: Access the robot object
    robot = env.scene["robot"]
    # Access relevant robot parts for collision detection
    # Using multiple parts to cover common collision points.
    pelvis_idx = robot.body_names.index('pelvis')
    left_palm_idx = robot.body_names.index('left_palm_link')
    right_palm_idx = robot.body_names.index('right_palm_link')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Define collision distance thresholds.
    # Blocks are 1m x 1m base. Half size is 0.5m.
    # Robot-block collision: If robot part center is within 0.5m (block half-size) + 0.1m (robot part radius/buffer)
    collision_threshold_robot_block = 0.6 # meters (hardcoded from plan)
    # Block-block collision: If block centers are closer than 0.9m (1m block side - 0.1m buffer),
    # it indicates interpenetration or undesired closeness.
    min_block_dist_for_interpenetration = 0.9 # meters (hardcoded from plan)

    reward = torch.zeros(env.num_envs, device=env.device)

    # Penalize robot-block collisions
    # Concatenate all relevant robot part positions for batched distance calculation.
    robot_parts_pos = torch.cat([pelvis_pos.unsqueeze(1), left_palm_pos.unsqueeze(1), right_palm_pos.unsqueeze(1),
                                 left_foot_pos.unsqueeze(1), right_foot_pos.unsqueeze(1)], dim=1) # Shape: [num_envs, num_parts, 3]

    for obj in [object1, object2, object3]:
        obj_pos = obj.data.root_pos_w # Shape: [num_envs, 3]
        # Calculate Euclidean distance from each robot part to the center of the current block.
        # Unsqueeze obj_pos to allow broadcasting across robot_parts_pos.
        dist_to_obj = torch.norm(robot_parts_pos - obj_pos.unsqueeze(1), dim=2) # Shape: [num_envs, num_parts]
        
        # Apply a linear penalty if any part is too close to the block center.
        # The penalty is continuous: it's 0 if distance >= threshold, and increases linearly as distance decreases.
        # CORRECT: Continuous linear penalty for collisions
        collision_penalty = torch.sum(torch.where(dist_to_obj < collision_threshold_robot_block,
                                                  -1.0 * (collision_threshold_robot_block - dist_to_obj),
                                                  torch.zeros_like(dist_to_obj)), dim=1) # Sum penalties across all parts
        reward += collision_penalty

    # Penalize undesired block-block collisions (interpenetration)
    # Calculate distances between all pairs of blocks.
    # CORRECT: Relative distances between blocks
    dist_obj1_obj2 = torch.norm(object1.data.root_pos_w - object2.data.root_pos_w, dim=1)
    dist_obj1_obj3 = torch.norm(object1.data.root_pos_w - object3.data.root_pos_w, dim=1)
    dist_obj2_obj3 = torch.norm(object2.data.root_pos_w - object3.data.root_pos_w, dim=1)

    # Apply linear penalty if blocks are too close, indicating interpenetration.
    # This penalty is active even if they are forming stairs, as it targets interpenetration.
    reward_block_collision = torch.zeros_like(reward)
    # CORRECT: Continuous linear penalty for block interpenetration
    reward_block_collision += torch.where(dist_obj1_obj2 < min_block_dist_for_interpenetration,
                                          -1.0 * (min_block_dist_for_interpenetration - dist_obj1_obj2),
                                          torch.zeros_like(dist_obj1_obj2))
    reward_block_collision += torch.where(dist_obj1_obj3 < min_block_dist_for_interpenetration,
                                          -1.0 * (min_block_dist_for_interpenetration - dist_obj1_obj3),
                                          torch.zeros_like(dist_obj1_obj3))
    reward_block_collision += torch.where(dist_obj2_obj3 < min_block_dist_for_interpenetration,
                                          -1.0 * (min_block_dist_for_interpenetration - dist_obj2_obj3),
                                          torch.zeros_like(dist_obj2_obj3))
    reward += reward_block_collision

    # Mandatory reward normalization
    # CORRECT: Normalization block
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Primary reward for building the stairs and robot final position
    # CORRECT: Main reward with weight 1.0
    build_stairs_main_reward = RewTerm(func=build_stairs_main_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "build_stairs_main_reward"})

    # Shaping reward for approaching and pushing blocks sequentially
    # CORRECT: Shaping reward with lower weight (0.4)
    approach_and_push_blocks_reward = RewTerm(func=approach_and_push_blocks_reward, weight=0.4,
                                              params={"normalise": True, "normaliser_name": "approach_and_push_blocks_reward"})

    # Shaping reward for collision avoidance
    # CORRECT: Shaping reward with lower weight (0.2)
    collision_avoidance_reward = RewTerm(func=collision_avoidance_reward, weight=0.2,
                                         params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})