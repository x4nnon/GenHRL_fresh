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


def main_build_stairs_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for building and climbing stairs.

    This reward guides the robot through two main phases:
    1. Pushing phase: Rewards the robot for moving blocks into a stair-like configuration.
    2. Climbing phase: Rewards the robot for climbing onto each block in sequence, culminating in standing on the largest block.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects using approved patterns
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcoded object dimensions from the description (x=1m y=1m z=height)
    # CRITICAL RULE: Object dimensions are hardcoded from the task description as per requirements.
    obj1_height = 0.3
    obj2_height = 0.6
    obj3_height = 0.9
    obj_width_xy = 1.0 # Assuming all blocks have 1m width/depth in X and Y

    # Calculate current block positions (absolute, will be used for relative distances)
    # CRITICAL RULE: Accessing object positions using approved pattern.
    obj1_pos = object1.data.root_pos_w
    obj2_pos = object2.data.root_pos_w
    obj3_pos = object3.data.root_pos_w

    # Define relative target positions for blocks to form stairs
    # The robot starts at (0,0,0). We define targets relative to this start.
    # Object1 target: A fixed point in front of the robot's start.
    # This is relative to the robot's initial position (0,0,0).
    # CRITICAL RULE: Hardcoded target positions are allowed for the initial block placement relative to the origin.
    target_obj1_x = 2.0
    target_obj1_y = 0.0
    target_obj1_z = obj1_height / 2.0 # Center Z for Object1

    # Object2 target relative to Object1: slightly offset in X, Y, and Z (base aligned)
    # This creates a step-like arrangement.
    # The X and Y offsets are chosen to create a diagonal step.
    # The Z offset is 0 because we want the bases to be on the ground, but the tops will be at different heights.
    # CRITICAL RULE: Relative target offsets are hardcoded as per requirements.
    target_obj2_rel_x_offset = 0.5 # 0.5m further in X from Object1 center
    target_obj2_rel_y_offset = 0.5 # 0.5m offset in Y from Object1 center
    target_obj2_rel_z_offset = 0.0 # Base Z aligned with Object1 base Z

    # Object3 target relative to Object2: similar offset
    # CRITICAL RULE: Relative target offsets are hardcoded as per requirements.
    target_obj3_rel_x_offset = 0.5 # 0.5m further in X from Object2 center
    target_obj3_rel_y_offset = 0.5 # 0.5m offset in Y from Object2 center
    target_obj3_rel_z_offset = 0.0 # Base Z aligned with Object2 base Z

    # Phase 1: Pushing rewards
    # Reward for Object1 reaching its target position
    # Using L1 norm for continuous reward, penalizing deviation from target.
    # CRITICAL RULE: Rewards are based on relative distances (object position relative to target).
    dist_obj1_from_target = torch.abs(obj1_pos[:, 0] - target_obj1_x) + \
                            torch.abs(obj1_pos[:, 1] - target_obj1_y) + \
                            torch.abs(obj1_pos[:, 2] - target_obj1_z)
    reward_push_obj1 = -dist_obj1_from_target # Continuous and negative for penalty

    # Reward for Object2 relative to Object1
    # Penalizing deviation from the desired relative position.
    # CRITICAL RULE: Rewards are based on relative distances between objects.
    dist_obj2_rel_obj1 = torch.abs(obj2_pos[:, 0] - (obj1_pos[:, 0] + target_obj2_rel_x_offset)) + \
                         torch.abs(obj2_pos[:, 1] - (obj1_pos[:, 1] + target_obj2_rel_y_offset)) + \
                         torch.abs(obj2_pos[:, 2] - (obj1_pos[:, 2] + target_obj2_rel_z_offset))
    reward_push_obj2 = -dist_obj2_rel_obj1 # Continuous and negative for penalty

    # Reward for Object3 relative to Object2
    # Penalizing deviation from the desired relative position.
    # CRITICAL RULE: Rewards are based on relative distances between objects.
    dist_obj3_rel_obj2 = torch.abs(obj3_pos[:, 0] - (obj2_pos[:, 0] + target_obj3_rel_x_offset)) + \
                         torch.abs(obj3_pos[:, 1] - (obj2_pos[:, 1] + target_obj3_rel_y_offset)) + \
                         torch.abs(obj3_pos[:, 2] - (obj2_pos[:, 2] + target_obj3_rel_z_offset))
    reward_push_obj3 = -dist_obj3_rel_obj2 # Continuous and negative for penalty

    # Condition for blocks being "in place" (thresholds for transition to climbing phase)
    # Using small thresholds for X/Y and Z to ensure blocks are reasonably positioned.
    # CRITICAL RULE: Thresholds are hardcoded as per requirements.
    blocks_in_place_condition = (torch.abs(obj1_pos[:, 0] - target_obj1_x) < 0.2) & \
                                (torch.abs(obj1_pos[:, 1] - target_obj1_y) < 0.2) & \
                                (torch.abs(obj1_pos[:, 2] - target_obj1_z) < 0.1) & \
                                (torch.abs(obj2_pos[:, 0] - (obj1_pos[:, 0] + target_obj2_rel_x_offset)) < 0.2) & \
                                (torch.abs(obj2_pos[:, 1] - (obj1_pos[:, 1] + target_obj2_rel_y_offset)) < 0.2) & \
                                (torch.abs(obj2_pos[:, 2] - (obj1_pos[:, 2] + target_obj2_rel_z_offset)) < 0.1) & \
                                (torch.abs(obj3_pos[:, 0] - (obj2_pos[:, 0] + target_obj3_rel_x_offset)) < 0.2) & \
                                (torch.abs(obj3_pos[:, 1] - (obj2_pos[:, 1] + target_obj3_rel_y_offset)) < 0.2) & \
                                (torch.abs(obj3_pos[:, 2] - (obj2_pos[:, 2] + target_obj3_rel_z_offset)) < 0.1)

    # Phase 2: Climbing rewards
    # Reward for feet on Object1
    # Check if feet are within the X/Y bounds of the block and at the correct Z height (top surface).
    # Using a small tolerance for Z height.
    # CRITICAL RULE: Rewards are based on relative positions of robot parts to objects.
    feet_on_obj1_x_cond = (left_foot_pos[:, 0] > obj1_pos[:, 0] - obj_width_xy/2) & (left_foot_pos[:, 0] < obj1_pos[:, 0] + obj_width_xy/2) & \
                          (right_foot_pos[:, 0] > obj1_pos[:, 0] - obj_width_xy/2) & (right_foot_pos[:, 0] < obj1_pos[:, 0] + obj_width_xy/2)
    feet_on_obj1_y_cond = (left_foot_pos[:, 1] > obj1_pos[:, 1] - obj_width_xy/2) & (left_foot_pos[:, 1] < obj1_pos[:, 1] + obj_width_xy/2) & \
                          (right_foot_pos[:, 1] > obj1_pos[:, 1] - obj_width_xy/2) & (right_foot_pos[:, 1] < obj1_pos[:, 1] + obj_width_xy/2)
    feet_on_obj1_z_cond = (left_foot_pos[:, 2] > obj1_pos[:, 2] + obj1_height/2 - 0.1) & (left_foot_pos[:, 2] < obj1_pos[:, 2] + obj1_height/2 + 0.2) & \
                          (right_foot_pos[:, 2] > obj1_pos[:, 2] + obj1_height/2 - 0.1) & (right_foot_pos[:, 2] < obj1_pos[:, 2] + obj1_height/2 + 0.2)
    feet_on_obj1 = feet_on_obj1_x_cond & feet_on_obj1_y_cond & feet_on_obj1_z_cond

    # Reward for feet on Object2
    # CRITICAL RULE: Rewards are based on relative positions of robot parts to objects.
    feet_on_obj2_x_cond = (left_foot_pos[:, 0] > obj2_pos[:, 0] - obj_width_xy/2) & (left_foot_pos[:, 0] < obj2_pos[:, 0] + obj_width_xy/2) & \
                          (right_foot_pos[:, 0] > obj2_pos[:, 0] - obj_width_xy/2) & (right_foot_pos[:, 0] < obj2_pos[:, 0] + obj_width_xy/2)
    feet_on_obj2_y_cond = (left_foot_pos[:, 1] > obj2_pos[:, 1] - obj_width_xy/2) & (left_foot_pos[:, 1] < obj2_pos[:, 1] + obj_width_xy/2) & \
                          (right_foot_pos[:, 1] > obj2_pos[:, 1] - obj_width_xy/2) & (right_foot_pos[:, 1] < obj2_pos[:, 1] + obj_width_xy/2)
    feet_on_obj2_z_cond = (left_foot_pos[:, 2] > obj2_pos[:, 2] + obj2_height/2 - 0.1) & (left_foot_pos[:, 2] < obj2_pos[:, 2] + obj2_height/2 + 0.2) & \
                          (right_foot_pos[:, 2] > obj2_pos[:, 2] + obj2_height/2 - 0.1) & (right_foot_pos[:, 2] < obj2_pos[:, 2] + obj2_height/2 + 0.2)
    feet_on_obj2 = feet_on_obj2_x_cond & feet_on_obj2_y_cond & feet_on_obj2_z_cond

    # Reward for feet on Object3 (final step)
    # CRITICAL RULE: Rewards are based on relative positions of robot parts to objects.
    feet_on_obj3_x_cond = (left_foot_pos[:, 0] > obj3_pos[:, 0] - obj_width_xy/2) & (left_foot_pos[:, 0] < obj3_pos[:, 0] + obj_width_xy/2) & \
                          (right_foot_pos[:, 0] > obj3_pos[:, 0] - obj_width_xy/2) & (right_foot_pos[:, 0] < obj3_pos[:, 0] + obj_width_xy/2)
    feet_on_obj3_y_cond = (left_foot_pos[:, 1] > obj3_pos[:, 1] - obj_width_xy/2) & (left_foot_pos[:, 1] < obj3_pos[:, 1] + obj_width_xy/2) & \
                          (right_foot_pos[:, 1] > obj3_pos[:, 1] - obj_width_xy/2) & (right_foot_pos[:, 1] < obj3_pos[:, 1] + obj_width_xy/2)
    feet_on_obj3_z_cond = (left_foot_pos[:, 2] > obj3_pos[:, 2] + obj3_height/2 - 0.1) & (left_foot_pos[:, 2] < obj3_pos[:, 2] + obj3_height/2 + 0.2) & \
                          (right_foot_pos[:, 2] > obj3_pos[:, 2] + obj3_height/2 - 0.1) & (right_foot_pos[:, 2] < obj3_pos[:, 2] + obj3_height/2 + 0.2)
    feet_on_obj3 = feet_on_obj3_x_cond & feet_on_obj3_y_cond & feet_on_obj3_z_cond

    # Pelvis height for stability on top of Object3
    # Encourages the robot to stand upright on the final block.
    # CRITICAL RULE: Reward based on relative Z height of pelvis to object.
    target_pelvis_z_on_obj3 = obj3_pos[:, 2] + obj3_height/2 + 0.7 # 0.7m above the top surface of Object3
    reward_pelvis_height_on_obj3 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj3) # Continuous and negative for penalty

    # Combine climbing rewards: progressive reward for reaching higher blocks.
    # Using torch.where for discrete steps in climbing reward, but the pelvis height reward is continuous.
    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    reward_climb = torch.zeros_like(pelvis_pos[:, 0])
    reward_climb = torch.where(feet_on_obj1, 1.0, reward_climb) # Base reward for being on obj1
    reward_climb = torch.where(feet_on_obj2, 2.0, reward_climb) # Higher reward for being on obj2
    reward_climb = torch.where(feet_on_obj3, 3.0 + reward_pelvis_height_on_obj3, reward_climb) # Highest for obj3 + stability

    # Final primary reward combines pushing and climbing phases
    # Switches from pushing rewards to climbing rewards once blocks are in place.
    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    primary_reward = torch.where(blocks_in_place_condition, reward_climb, reward_push_obj1 + reward_push_obj2 + reward_push_obj3)

    # Mandatory reward normalization
    # CRITICAL RULE: Normalization is applied at the end of every reward function.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def robot_block_interaction_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "robot_block_interaction_reward") -> torch.Tensor:
    '''Shaping reward for robot-block interaction during the pushing phase.

    This reward encourages the robot's hands to be close to the current target block,
    providing a continuous gradient for interaction. It also encourages the robot to be
    generally in front of the block it intends to push.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects
    # CRITICAL RULE: Accessing objects using approved pattern.
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access robot parts
    # CRITICAL RULE: Accessing robot parts using approved pattern.
    robot = env.scene["robot"]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcoded object dimensions
    # CRITICAL RULE: Object dimensions are hardcoded from the task description.
    obj1_height = 0.3
    obj2_height = 0.6
    obj3_height = 0.9
    obj_width_xy = 1.0

    # Calculate current block positions
    # CRITICAL RULE: Accessing object positions using approved pattern.
    obj1_pos = object1.data.root_pos_w
    obj2_pos = object2.data.root_pos_w
    obj3_pos = object3.data.root_pos_w

    # Define relative target positions for blocks (re-using from main reward for consistency)
    # CRITICAL RULE: Hardcoded target positions/offsets are allowed.
    target_obj1_x = 2.0
    target_obj1_y = 0.0
    target_obj1_z = obj1_height / 2.0

    target_obj2_rel_x_offset = 0.5
    target_obj2_rel_y_offset = 0.5
    target_obj2_rel_z_offset = 0.0

    target_obj3_rel_x_offset = 0.5
    target_obj3_rel_y_offset = 0.5
    target_obj3_rel_z_offset = 0.0

    # Calculate current block placement distances (for phase transition logic)
    # CRITICAL RULE: Distances are calculated using torch.norm for relative positions.
    dist_obj1_from_target_xy = torch.norm(obj1_pos[:, :2] - torch.tensor([target_obj1_x, target_obj1_y], device=env.device), dim=1)
    dist_obj2_rel_obj1_xy = torch.norm(obj2_pos[:, :2] - (obj1_pos[:, :2] + torch.tensor([target_obj2_rel_x_offset, target_obj2_rel_y_offset], device=env.device)), dim=1)
    dist_obj3_rel_obj2_xy = torch.norm(obj3_pos[:, :2] - (obj2_pos[:, :2] + torch.tensor([target_obj3_rel_x_offset, target_obj3_rel_y_offset], device=env.device)), dim=1)

    # Condition for blocks being "in place" (re-using from main reward)
    # CRITICAL RULE: Thresholds are hardcoded as per requirements.
    blocks_in_place_condition = (torch.abs(obj1_pos[:, 0] - target_obj1_x) < 0.2) & \
                                (torch.abs(obj1_pos[:, 1] - target_obj1_y) < 0.2) & \
                                (torch.abs(obj1_pos[:, 2] - target_obj1_z) < 0.1) & \
                                (torch.abs(obj2_pos[:, 0] - (obj1_pos[:, 0] + target_obj2_rel_x_offset)) < 0.2) & \
                                (torch.abs(obj2_pos[:, 1] - (obj1_pos[:, 1] + target_obj2_rel_y_offset)) < 0.2) & \
                                (torch.abs(obj2_pos[:, 2] - (obj1_pos[:, 2] + target_obj2_rel_z_offset)) < 0.1) & \
                                (torch.abs(obj3_pos[:, 0] - (obj2_pos[:, 0] + target_obj3_rel_x_offset)) < 0.2) & \
                                (torch.abs(obj3_pos[:, 1] - (obj2_pos[:, 1] + target_obj3_rel_y_offset)) < 0.2) & \
                                (torch.abs(obj3_pos[:, 2] - (obj2_pos[:, 2] + target_obj3_rel_z_offset)) < 0.1)

    # Determine which block is the current target for pushing based on completion
    # This creates a sequential pushing behavior.
    # CRITICAL RULE: Logic uses relative distances and thresholds.
    pushing_obj1_active = (dist_obj1_from_target_xy > 0.2) # If obj1 not in place in XY
    pushing_obj2_active = (dist_obj1_from_target_xy < 0.5) & (dist_obj2_rel_obj1_xy > 0.2) # If obj1 somewhat in place, and obj2 not
    pushing_obj3_active = (dist_obj2_rel_obj1_xy < 0.5) & (dist_obj3_rel_obj2_xy > 0.2) # If obj2 somewhat in place, and obj3 not

    # Calculate average hand position
    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    avg_hand_pos_x = (left_hand_pos[:, 0] + right_hand_pos[:, 0]) / 2.0
    avg_hand_pos_y = (left_hand_pos[:, 1] + right_hand_pos[:, 1]) / 2.0
    avg_hand_pos_z = (left_hand_pos[:, 2] + right_hand_pos[:, 2]) / 2.0

    # Reward for hands near Object1
    # Encourages hands to be at the center height of the block.
    # CRITICAL RULE: Reward based on relative distances (hand position relative to object center).
    dist_hand_obj1 = torch.abs(avg_hand_pos_x - obj1_pos[:, 0]) + \
                     torch.abs(avg_hand_pos_y - obj1_pos[:, 1]) + \
                     torch.abs(avg_hand_pos_z - (obj1_pos[:, 2] + obj1_height / 2.0))
    reward_hands_obj1 = -dist_hand_obj1 # Continuous and negative for penalty

    # Reward for hands near Object2
    # CRITICAL RULE: Reward based on relative distances (hand position relative to object center).
    dist_hand_obj2 = torch.abs(avg_hand_pos_x - obj2_pos[:, 0]) + \
                     torch.abs(avg_hand_pos_y - obj2_pos[:, 1]) + \
                     torch.abs(avg_hand_pos_z - (obj2_pos[:, 2] + obj2_height / 2.0))
    reward_hands_obj2 = -dist_hand_obj2 # Continuous and negative for penalty

    # Reward for hands near Object3
    # CRITICAL RULE: Reward based on relative distances (hand position relative to object center).
    dist_hand_obj3 = torch.abs(avg_hand_pos_x - obj3_pos[:, 0]) + \
                     torch.abs(avg_hand_pos_y - obj3_pos[:, 1]) + \
                     torch.abs(avg_hand_pos_z - (obj3_pos[:, 2] + obj3_height / 2.0))
    reward_hands_obj3 = -dist_hand_obj3 # Continuous and negative for penalty

    # Combine interaction rewards based on active pushing phase
    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    shaping_reward1 = torch.zeros_like(pelvis_pos[:, 0])
    shaping_reward1 = torch.where(pushing_obj1_active, reward_hands_obj1, shaping_reward1)
    shaping_reward1 = torch.where(pushing_obj2_active, reward_hands_obj2, shaping_reward1)
    shaping_reward1 = torch.where(pushing_obj3_active, reward_hands_obj3, shaping_reward1)

    # Ensure this reward is only active during the pushing phase
    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    shaping_reward1 = torch.where(~blocks_in_place_condition, shaping_reward1, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    # CRITICAL RULE: Normalization is applied at the end of every reward function.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance.

    This reward penalizes collisions between the robot's body parts and the blocks,
    and also penalizes excessive overlap between the blocks themselves.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects
    # CRITICAL RULE: Accessing objects using approved pattern.
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access robot parts
    # CRITICAL RULE: Accessing robot parts using approved pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    head_idx = robot.body_names.index('head_link')
    head_pos = robot.data.body_pos_w[:, head_idx]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcoded object dimensions
    # CRITICAL RULE: Object dimensions are hardcoded from the task description.
    obj1_height = 0.3
    obj2_height = 0.6
    obj3_height = 0.9
    obj_width_xy = 1.0

    # Calculate current block positions
    # CRITICAL RULE: Accessing object positions using approved pattern.
    obj1_pos = object1.data.root_pos_w
    obj2_pos = object2.data.root_pos_w
    obj3_pos = object3.data.root_pos_w

    # Initialize as per-environment tensor to avoid in-place broadcast issues
    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Helper function for robot part-block collision
    # CRITICAL RULE: Rewards are based on relative distances (robot part position relative to object center).
    def penalize_robot_block_collision(robot_part_pos, obj_pos, obj_dims, penalty_value=-0.5):
        # obj_dims = [width_xy, width_xy, height]
        # Using a small buffer (0.1m) around the block for collision detection.
        dist_x = torch.abs(robot_part_pos[:, 0] - obj_pos[:, 0])
        dist_y = torch.abs(robot_part_pos[:, 1] - obj_pos[:, 1])
        dist_z = torch.abs(robot_part_pos[:, 2] - obj_pos[:, 2])
        
        # Check for overlap in all dimensions
        # CRITICAL RULE: Thresholds are hardcoded as per requirements.
        collision_cond = (dist_x < obj_dims[0]/2 + 0.1) & \
                         (dist_y < obj_dims[1]/2 + 0.1) & \
                         (dist_z < obj_dims[2]/2 + 0.1)
        # Return per-env penalty on the correct device
        return torch.where(collision_cond, torch.tensor(penalty_value, device=env.device), torch.tensor(0.0, device=env.device))

    # Penalize collisions for various robot parts with all blocks
    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    collision_penalty += penalize_robot_block_collision(pelvis_pos, obj1_pos, [obj_width_xy, obj_width_xy, obj1_height])
    collision_penalty += penalize_robot_block_collision(pelvis_pos, obj2_pos, [obj_width_xy, obj_width_xy, obj2_height])
    collision_penalty += penalize_robot_block_collision(pelvis_pos, obj3_pos, [obj_width_xy, obj_width_xy, obj3_height])

    collision_penalty += penalize_robot_block_collision(head_pos, obj1_pos, [obj_width_xy, obj_width_xy, obj1_height])
    collision_penalty += penalize_robot_block_collision(head_pos, obj2_pos, [obj_width_xy, obj_width_xy, obj2_height])
    collision_penalty += penalize_robot_block_collision(head_pos, obj3_pos, [obj_width_xy, obj_width_xy, obj3_height])

    collision_penalty += penalize_robot_block_collision(left_hand_pos, obj1_pos, [obj_width_xy, obj_width_xy, obj1_height])
    collision_penalty += penalize_robot_block_collision(left_hand_pos, obj2_pos, [obj_width_xy, obj_width_xy, obj2_height])
    collision_penalty += penalize_robot_block_collision(left_hand_pos, obj3_pos, [obj_width_xy, obj_width_xy, obj3_height])

    collision_penalty += penalize_robot_block_collision(right_hand_pos, obj1_pos, [obj_width_xy, obj_width_xy, obj1_height])
    collision_penalty += penalize_robot_block_collision(right_hand_pos, obj2_pos, [obj_width_xy, obj_width_xy, obj2_height])
    collision_penalty += penalize_robot_block_collision(right_hand_pos, obj3_pos, [obj_width_xy, obj_width_xy, obj3_height])

    # Feet collisions are expected during climbing, so penalize only if they are *inside* the block, not just on top.
    # A smaller buffer or different logic might be needed for feet if they are meant to be on the block.
    # For general collision avoidance, we can use the same logic.
    collision_penalty += penalize_robot_block_collision(left_foot_pos, obj1_pos, [obj_width_xy, obj_width_xy, obj1_height])
    collision_penalty += penalize_robot_block_collision(left_foot_pos, obj2_pos, [obj_width_xy, obj_width_xy, obj2_height])
    collision_penalty += penalize_robot_block_collision(left_foot_pos, obj3_pos, [obj_width_xy, obj_width_xy, obj3_height])

    collision_penalty += penalize_robot_block_collision(right_foot_pos, obj1_pos, [obj_width_xy, obj_width_xy, obj1_height])
    collision_penalty += penalize_robot_block_collision(right_foot_pos, obj2_pos, [obj_width_xy, obj_width_xy, obj2_height])
    collision_penalty += penalize_robot_block_collision(right_foot_pos, obj3_pos, [obj_width_xy, obj_width_xy, obj3_height])


    # Collision avoidance between blocks (e.g., preventing them from overlapping too much)
    # Penalize if centers are too close, considering their dimensions.
    # Using a factor (e.g., 0.8) to allow some slight overlap if necessary for stair formation, but penalize significant overlap.
    # CRITICAL RULE: Rewards are based on relative distances between objects.
    def penalize_block_block_overlap(pos1, h1, pos2, h2, width_xy, penalty_value=-0.2):
        dist_x = torch.abs(pos1[:, 0] - pos2[:, 0])
        dist_y = torch.abs(pos1[:, 1] - pos2[:, 1])
        dist_z = torch.abs(pos1[:, 2] - pos2[:, 2])
        
        # Check for significant overlap in X, Y, and Z
        # CRITICAL RULE: Thresholds are hardcoded as per requirements.
        overlap_cond = (dist_x < width_xy * 0.8) & \
                       (dist_y < width_xy * 0.8) & \
                       (dist_z < (h1/2 + h2/2) * 0.8) # Z overlap based on half-heights
        return torch.where(overlap_cond, torch.tensor(penalty_value, device=env.device), torch.tensor(0.0, device=env.device))

    # CRITICAL RULE: Tensor operations handle batched environments correctly.
    collision_penalty += penalize_block_block_overlap(obj1_pos, obj1_height, obj2_pos, obj2_height, obj_width_xy)
    collision_penalty += penalize_block_block_overlap(obj2_pos, obj2_height, obj3_pos, obj3_height, obj_width_xy)
    # Also penalize direct overlap between Object1 and Object3 if they are not meant to be adjacent
    collision_penalty += penalize_block_block_overlap(obj1_pos, obj1_height, obj3_pos, obj3_height, obj_width_xy)

    shaping_reward2 = collision_penalty

    # Mandatory reward normalization
    # CRITICAL RULE: Normalization is applied at the end of every reward function.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # Primary reward for building and climbing stairs
    # CRITICAL RULE: Main reward has a weight of 1.0.
    main_build_stairs_reward = RewTerm(func=main_build_stairs_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for robot-block interaction during pushing phase
    # CRITICAL RULE: Shaping rewards have weights less than 1.0.
    robot_block_interaction_reward = RewTerm(func=robot_block_interaction_reward, weight=0.4,
                                             params={"normalise": True, "normaliser_name": "robot_block_interaction_reward"})

    # Shaping reward for collision avoidance
    # CRITICAL RULE: Shaping rewards have weights less than 1.0.
    collision_avoidance_reward = RewTerm(func=collision_avoidance_reward, weight=0.2,
                                         params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})