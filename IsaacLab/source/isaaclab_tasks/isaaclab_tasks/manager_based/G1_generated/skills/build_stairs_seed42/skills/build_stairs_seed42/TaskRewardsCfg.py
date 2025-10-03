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
    """
    Main reward for the build_stairs_seed42 skill.
    This reward encourages the robot to first arrange the blocks into a stair-like structure
    and then to climb the largest block (Object3).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects based on the object configuration
    # Object1: Small Block (1m x 1m x 0.3m)
    # Object2: Medium Block (1m x 1m x 0.6m)
    # Object3: Large Block (1m x 1m x 0.9m)
    object1 = env.scene['Object1'] # Accessing object using approved pattern
    object2 = env.scene['Object2'] # Accessing object using approved pattern
    object3 = env.scene['Object3'] # Accessing object using approved pattern

    # Access required robot parts
    robot = env.scene["robot"] # Accessing robot using approved pattern
    left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing robot part index using approved pattern
    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing robot part index using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing robot part position using approved pattern
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing robot part position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern

    # Hardcoded block dimensions (from object configuration) - REQUIRED as per prompt
    obj1_height = 0.3
    obj2_height = 0.6
    obj3_height = 0.9
    block_half_size_xy = 0.5 # Half of 1m block size in X and Y

    # Define target relative positions for blocks to form stairs
    # These values are chosen to create a reasonable stair step.
    # Object2 relative to Object1: slightly behind (+Y), slightly offset in X
    target_obj2_rel_x = 0.5
    target_obj2_rel_y = 0.8
    # Object3 relative to Object2: slightly behind (+Y), slightly offset in X
    target_obj3_rel_x = 0.5
    target_obj3_rel_y = 0.8

    # Phase 1: Block Arrangement Rewards
    # Reward for Object2 being in position relative to Object1
    # Using relative distances between object root positions.
    dist_obj2_obj1_x = object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0] # Relative distance in X
    dist_obj2_obj1_y = object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1] # Relative distance in Y
    # Z-distance should be close to 0 if blocks are on the ground plane relative to each other.
    dist_obj2_obj1_z = object2.data.root_pos_w[:, 2] - object1.data.root_pos_w[:, 2] # Relative distance in Z

    # Reward is negative absolute difference, encouraging values to be close to target.
    # Continuous reward based on relative distances
    reward_obj2_pos = -torch.abs(dist_obj2_obj1_x - target_obj2_rel_x) \
                      -torch.abs(dist_obj2_obj1_y - target_obj2_rel_y) \
                      -torch.abs(dist_obj2_obj1_z) # Encourage blocks to be on the same ground plane initially

    # Reward for Object3 being in position relative to Object2
    dist_obj3_obj2_x = object3.data.root_pos_w[:, 0] - object2.data.root_pos_w[:, 0] # Relative distance in X
    dist_obj3_obj2_y = object3.data.root_pos_w[:, 1] - object2.data.root_pos_w[:, 1] # Relative distance in Y
    dist_obj3_obj2_z = object3.data.root_pos_w[:, 2] - object2.data.root_pos_w[:, 2] # Relative distance in Z

    # Continuous reward based on relative distances
    reward_obj3_pos = -torch.abs(dist_obj3_obj2_x - target_obj3_rel_x) \
                      -torch.abs(dist_obj3_obj2_y - target_obj3_rel_y) \
                      -torch.abs(dist_obj3_obj2_z)

    # Condition for blocks being considered "arranged"
    # Using small thresholds for relative positions to define completion of phase 1.
    blocks_arranged_cond = (torch.abs(dist_obj2_obj1_x - target_obj2_rel_x) < 0.2) & \
                           (torch.abs(dist_obj2_obj1_y - target_obj2_rel_y) < 0.2) & \
                           (torch.abs(dist_obj3_obj2_x - target_obj3_rel_x) < 0.2) & \
                           (torch.abs(dist_obj3_obj2_y - target_obj3_rel_y) < 0.2)

    # Phase 2: Climbing Rewards (activated once blocks are arranged)
    # Target Z-position for feet on top of Object3 (Large Block)
    # Object root_pos_w[:, 2] is the center of the block. Top surface is center + half_height.
    # Add a small offset (0.05m) for stability/clearance.
    # This is a relative target Z based on the block's current Z position.
    target_foot_z_on_obj3 = object3.data.root_pos_w[:, 2] + (obj3_height / 2) + 0.05

    # Reward for feet being at the target Z-height on Object3
    # Continuous reward based on relative Z distance
    reward_feet_on_obj3_z = -torch.abs(left_foot_pos[:, 2] - target_foot_z_on_obj3) \
                           -torch.abs(right_foot_pos[:, 2] - target_foot_z_on_obj3)

    # Reward for feet being horizontally centered on Object3
    obj3_center_x = object3.data.root_pos_w[:, 0]
    obj3_center_y = object3.data.root_pos_w[:, 1]

    # Reward is negative absolute difference from the center of Object3.
    # Continuous reward based on relative XY distances
    reward_feet_on_obj3_xy = -torch.abs(left_foot_pos[:, 0] - obj3_center_x) -torch.abs(left_foot_pos[:, 1] - obj3_center_y) \
                            -torch.abs(right_foot_pos[:, 0] - obj3_center_x) -torch.abs(right_foot_pos[:, 1] - obj3_center_y)

    # Condition for feet being "on" Object3 (within its horizontal bounds and at correct height)
    feet_on_obj3_x_cond = (left_foot_pos[:, 0] > obj3_center_x - block_half_size_xy) & (left_foot_pos[:, 0] < obj3_center_x + block_half_size_xy) & \
                          (right_foot_pos[:, 0] > obj3_center_x - block_half_size_xy) & (right_foot_pos[:, 0] < obj3_center_x + block_half_size_xy)
    feet_on_obj3_y_cond = (left_foot_pos[:, 1] > obj3_center_y - block_half_size_xy) & (left_foot_pos[:, 1] < obj3_center_y + block_half_size_xy) & \
                          (right_foot_pos[:, 1] > obj3_center_y - block_half_size_xy) & (right_foot_pos[:, 1] < obj3_center_y + block_half_size_xy)
    feet_on_obj3_z_approx_cond = (left_foot_pos[:, 2] > target_foot_z_on_obj3 - 0.1) & (right_foot_pos[:, 2] > target_foot_z_on_obj3 - 0.1)
    feet_on_obj3_overall_cond = feet_on_obj3_x_cond & feet_on_obj3_y_cond & feet_on_obj3_z_approx_cond

    # Reward for pelvis stability at target height on Object3
    # Target pelvis Z is above the feet, assuming a standing posture.
    # This is a relative target Z based on the block's current Z position.
    target_pelvis_z_on_obj3 = target_foot_z_on_obj3 + 0.7 # Approximately 0.7m above feet for standing
    # Continuous reward based on relative Z distance
    reward_pelvis_stability = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj3)

    # Combine rewards with conditional activation for phases
    # Initial focus on block arrangement.
    primary_reward = (reward_obj2_pos + reward_obj3_pos) * 0.5

    # Once blocks are arranged, add climbing rewards.
    # Uses torch.where for conditional reward activation, ensuring continuity.
    primary_reward = torch.where(blocks_arranged_cond, primary_reward + (reward_feet_on_obj3_z + reward_feet_on_obj3_xy) * 0.5, primary_reward)

    # Add pelvis stability reward only when feet are properly on Object3.
    # Uses torch.where for conditional reward activation, ensuring continuity.
    primary_reward = torch.where(feet_on_obj3_overall_cond, primary_reward + reward_pelvis_stability * 0.2, primary_reward)

    # Normalization - MANDATORY
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def shaping_approach_blocks_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_approach_blocks") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach and push blocks in sequence.
    It rewards reducing the distance between the robot's pelvis and the current target block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access required robot part
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define target relative positions for blocks (re-used from main reward for consistency)
    target_obj2_rel_x = 0.5
    target_obj2_rel_y = 0.8
    target_obj3_rel_x = 0.5
    target_obj3_rel_y = 0.8

    # Define conditions for sequential approach based on block placement progress
    # Condition for Object1 being "in place" (e.g., near the origin where it starts or pushed slightly)
    # Assuming Object1's initial target is near (0,0) for simplicity in this shaping reward.
    # This uses a hardcoded target for Object1's initial position, which is acceptable for a "shaping" reward
    # to guide the robot from its starting point towards the first block.
    obj1_target_x = 0.0
    obj1_target_y = 0.0
    # Relative distance calculation
    dist_obj1_to_initial_target = torch.sqrt(torch.square(object1.data.root_pos_w[:, 0] - obj1_target_x) +
                                             torch.square(object1.data.root_pos_w[:, 1] - obj1_target_y))
    obj1_in_place_shaping_cond = (dist_obj1_to_initial_target < 0.5) # Object1 is within 0.5m of its initial target

    # Condition for Object2 being "in place" relative to Object1
    dist_obj2_obj1_x = object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0] # Relative distance
    dist_obj2_obj1_y = object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1] # Relative distance
    obj2_in_place_shaping_cond = (torch.abs(dist_obj2_obj1_x - target_obj2_rel_x) < 0.2) & \
                                 (torch.abs(dist_obj2_obj1_y - target_obj2_rel_y) < 0.2)

    # Condition for Object3 being "in place" relative to Object2
    dist_obj3_obj2_x = object3.data.root_pos_w[:, 0] - object2.data.root_pos_w[:, 0] # Relative distance
    dist_obj3_obj2_y = object3.data.root_pos_w[:, 1] - object2.data.root_pos_w[:, 1] # Relative distance
    obj3_in_place_shaping_cond = (torch.abs(dist_obj3_obj2_x - target_obj3_rel_x) < 0.2) & \
                                 (torch.abs(dist_obj3_obj2_y - target_obj3_rel_y) < 0.2)

    # Calculate Euclidean distances from pelvis to each block (relative distances)
    dist_pelvis_obj1 = torch.norm(pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2], dim=1)
    dist_pelvis_obj2 = torch.norm(pelvis_pos[:, :2] - object2.data.root_pos_w[:, :2], dim=1)
    dist_pelvis_obj3 = torch.norm(pelvis_pos[:, :2] - object3.data.root_pos_w[:, :2], dim=1)

    # Reward for approaching the current target block (negative distance)
    # Default: approach Object1
    reward_approach = -dist_pelvis_obj1 # Continuous reward

    # If Object1 is in place, switch to approaching Object2
    # Uses torch.where for conditional reward activation, ensuring continuity.
    reward_approach = torch.where(obj1_in_place_shaping_cond & ~obj2_in_place_shaping_cond, -dist_pelvis_obj2, reward_approach)

    # If Object2 is in place, switch to approaching Object3
    # Uses torch.where for conditional reward activation, ensuring continuity.
    reward_approach = torch.where(obj1_in_place_shaping_cond & obj2_in_place_shaping_cond & ~obj3_in_place_shaping_cond, -dist_pelvis_obj3, reward_approach)

    # If all blocks are in place, the robot should be climbing, so this approach reward becomes less relevant or zero.
    # Uses torch.where for conditional reward activation, ensuring continuity.
    all_blocks_arranged_cond = obj1_in_place_shaping_cond & obj2_in_place_shaping_cond & obj3_in_place_shaping_cond
    reward_approach = torch.where(all_blocks_arranged_cond, torch.tensor(0.0, device=env.device), reward_approach)

    shaping_reward1 = reward_approach

    # Normalization - MANDATORY
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def shaping_posture_and_interaction_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_posture_interaction") -> torch.Tensor:
    """
    Shaping reward for maintaining stable posture, encouraging proximity for pushing,
    penalizing collisions, and rewarding increasing pelvis height during climbing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    object1 = env.scene['Object1'] # Small Block (1m x 1m x 0.3m)
    object2 = env.scene['Object2'] # Medium Block (1m x 1m x 0.6m)
    object3 = env.scene['Object3'] # Large Block (1m x 1m x 0.9m)

    # Access required robot parts
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Hardcoded block dimensions (from object configuration) - REQUIRED as per prompt
    obj1_height = 0.3
    obj2_height = 0.6
    obj3_height = 0.9
    block_half_size_xy = 0.5 # Half of 1m block size in X and Y

    # Pelvis height stability reward (when on ground or on top of blocks)
    # Target pelvis Z for standing on the ground. This is an absolute Z target.
    target_pelvis_z_ground = 0.7
    reward_pelvis_z_stability = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_ground) # Continuous reward

    # Collision avoidance and push proximity
    # Calculate 3D distance from pelvis to each block's center (relative distances)
    dist_pelvis_obj1_xyz = torch.norm(object1.data.root_pos_w - pelvis_pos, dim=1)
    dist_pelvis_obj2_xyz = torch.norm(object2.data.root_pos_w - pelvis_pos, dim=1)
    dist_pelvis_obj3_xyz = torch.norm(object3.data.root_pos_w - pelvis_pos, dim=1)

    # Collision threshold: pelvis is too close to the block.
    # Block half size (0.5m) + a small buffer (e.g., 0.1m for robot body radius) = 0.6m
    collision_threshold = block_half_size_xy + 0.1

    # Penalty for collision (being too close)
    # Uses torch.where for conditional reward, ensuring continuity by applying penalty only when condition met.
    penalty_collision = torch.zeros_like(dist_pelvis_obj1_xyz)
    penalty_collision = torch.where(dist_pelvis_obj1_xyz < collision_threshold, -1.0, penalty_collision)
    penalty_collision = torch.where(dist_pelvis_obj2_xyz < collision_threshold, -1.0, penalty_collision)
    penalty_collision = torch.where(dist_pelvis_obj3_xyz < collision_threshold, -1.0, penalty_collision)

    # Reward for being in a good range for pushing (positive reward for being within push range, but not colliding)
    push_range_min = collision_threshold # Start rewarding just outside collision
    push_range_max = 1.0 # Max distance for effective pushing

    reward_push_proximity = torch.zeros_like(dist_pelvis_obj1_xyz)

    # Reward for being in push range of Object1
    # Continuous reward based on distance within a range.
    reward_push_proximity = torch.where((dist_pelvis_obj1_xyz >= push_range_min) & (dist_pelvis_obj1_xyz < push_range_max),
                                        (push_range_max - dist_pelvis_obj1_xyz) * 0.1, reward_push_proximity)

    # Conditions for block placement (re-used from main reward for consistency)
    target_obj2_rel_x = 0.5
    target_obj2_rel_y = 0.8
    target_obj3_rel_x = 0.5
    target_obj3_rel_y = 0.8

    # Condition for Object1 being "in place" (e.g., near the origin where it starts or pushed slightly)
    # This uses a hardcoded target for Object1's initial position, which is acceptable for a "shaping" reward.
    obj1_in_place_shaping_cond = (torch.sqrt(torch.square(object1.data.root_pos_w[:, 0] - 0.0) + torch.square(object1.data.root_pos_w[:, 1] - 0.0)) < 0.5)
    dist_obj2_obj1_x = object2.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0]
    dist_obj2_obj1_y = object2.data.root_pos_w[:, 1] - object1.data.root_pos_w[:, 1]
    obj2_in_place_shaping_cond = (torch.abs(dist_obj2_obj1_x - target_obj2_rel_x) < 0.2) & \
                                 (torch.abs(dist_obj2_obj1_y - target_obj2_rel_y) < 0.2)

    # Add reward for Object2 if Object1 is in place
    # Uses torch.where for conditional reward, ensuring continuity.
    reward_push_proximity = torch.where(obj1_in_place_shaping_cond & (dist_pelvis_obj2_xyz >= push_range_min) & (dist_pelvis_obj2_xyz < push_range_max),
                                        reward_push_proximity + (push_range_max - dist_pelvis_obj2_xyz) * 0.1, reward_push_proximity)

    # Add reward for Object3 if Object2 is in place
    # Uses torch.where for conditional reward, ensuring continuity.
    reward_push_proximity = torch.where(obj1_in_place_shaping_cond & obj2_in_place_shaping_cond & (dist_pelvis_obj3_xyz >= push_range_min) & (dist_pelvis_obj3_xyz < push_range_max),
                                        reward_push_proximity + (push_range_max - dist_pelvis_obj3_xyz) * 0.1, reward_push_proximity)

    # Reward for increasing pelvis height during climbing phase
    # This reward is active only when blocks are arranged and robot is attempting to climb.
    # Re-using the `blocks_arranged_cond` from the main reward for consistency.
    dist_obj3_obj2_x = object3.data.root_pos_w[:, 0] - object2.data.root_pos_w[:, 0]
    dist_obj3_obj2_y = object3.data.root_pos_w[:, 1] - object2.data.root_pos_w[:, 1]
    blocks_arranged_cond = (torch.abs(dist_obj2_obj1_x - target_obj2_rel_x) < 0.2) & \
                           (torch.abs(dist_obj2_obj1_y - target_obj2_rel_y) < 0.2) & \
                           (torch.abs(dist_obj3_obj2_x - target_obj3_rel_x) < 0.2) & \
                           (torch.abs(dist_obj3_obj2_y - target_obj3_rel_y) < 0.2)

    # Target pelvis Z for standing on top of each block
    # Pelvis height is approximately 0.7m above the top surface of the block.
    # These are relative targets based on the block's current Z position.
    target_pelvis_z_on_obj1 = object1.data.root_pos_w[:, 2] + (obj1_height / 2) + 0.7
    target_pelvis_z_on_obj2 = object2.data.root_pos_w[:, 2] + (obj2_height / 2) + 0.7
    target_pelvis_z_on_obj3 = object3.data.root_pos_w[:, 2] + (obj3_height / 2) + 0.7

    # Reward for pelvis height increasing towards Object1 height
    reward_climb_obj1 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj1) # Continuous reward
    # Only active if blocks are arranged and robot is near Object1 (within 1m horizontal distance).
    # Uses torch.where for conditional reward, ensuring continuity.
    is_near_obj1 = torch.norm(pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2], dim=1) < 1.0
    reward_climb_obj1 = torch.where(blocks_arranged_cond & is_near_obj1, reward_climb_obj1, torch.tensor(0.0, device=env.device))

    # Reward for pelvis height increasing towards Object2 height
    reward_climb_obj2 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj2) # Continuous reward
    # Only active if blocks are arranged, robot is near Object2, and pelvis is already above Object1 height.
    # Uses torch.where for conditional reward, ensuring continuity.
    is_near_obj2 = torch.norm(pelvis_pos[:, :2] - object2.data.root_pos_w[:, :2], dim=1) < 1.0
    pelvis_above_obj1 = pelvis_pos[:, 2] > target_pelvis_z_on_obj1 - 0.1 # Small tolerance
    reward_climb_obj2 = torch.where(blocks_arranged_cond & is_near_obj2 & pelvis_above_obj1, reward_climb_obj2, torch.tensor(0.0, device=env.device))

    # Reward for pelvis height increasing towards Object3 height
    reward_climb_obj3 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_on_obj3) # Continuous reward
    # Only active if blocks are arranged, robot is near Object3, and pelvis is already above Object2 height.
    # Uses torch.where for conditional reward, ensuring continuity.
    is_near_obj3 = torch.norm(pelvis_pos[:, :2] - object3.data.root_pos_w[:, :2], dim=1) < 1.0
    pelvis_above_obj2 = pelvis_pos[:, 2] > target_pelvis_z_on_obj2 - 0.1
    reward_climb_obj3 = torch.where(blocks_arranged_cond & is_near_obj3 & pelvis_above_obj2, reward_climb_obj3, torch.tensor(0.0, device=env.device))

    # Combine all components for shaping reward 2
    shaping_reward2 = reward_pelvis_z_stability * 0.1 \
                      + penalty_collision * 5.0 \
                      + reward_push_proximity * 0.5 \
                      + reward_climb_obj1 * 0.2 \
                      + reward_climb_obj2 * 0.3 \
                      + reward_climb_obj3 * 0.4

    # Normalization - MANDATORY
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    """
    Reward terms for the build_stairs_seed42 skill.
    """
    # Main reward for arranging blocks and climbing them.
    MainBuildStairsReward = RewTerm(func=main_build_stairs_reward, weight=1.0, # Main reward with weight 1.0
                                    params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for guiding the robot to approach blocks in sequence for pushing.
    ShapingApproachBlocksReward = RewTerm(func=shaping_approach_blocks_reward, weight=0.4, # Shaping reward with lower weight
                                          params={"normalise": True, "normaliser_name": "shaping_approach_blocks"})

    # Shaping reward for maintaining posture, managing collisions, and rewarding climbing height.
    ShapingPostureAndInteractionReward = RewTerm(func=shaping_posture_and_interaction_reward, weight=0.3, # Shaping reward with lower weight
                                                 params={"normalise": True, "normaliser_name": "shaping_posture_interaction"})