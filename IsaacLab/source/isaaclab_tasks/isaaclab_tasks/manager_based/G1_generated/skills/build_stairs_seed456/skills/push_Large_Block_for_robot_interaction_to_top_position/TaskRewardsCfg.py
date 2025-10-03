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


def primary_push_align_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_reward") -> torch.Tensor:
    """
    Primary reward for pushing the Large Block (Object3) to its designated top stair position,
    adjacent to the Medium Block (Object2). This reward combines an approach phase and a push/alignment phase.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3'] # Large Block
    object2 = env.scene['Object2'] # Medium Block

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Object dimensions (hardcoded from description as per requirements)
    # Large Block: x=1m y=1m z=0.9m
    # Medium Block: x=1m y=1m z=0.6m
    large_block_size_x = 1.0
    large_block_size_y = 1.0
    large_block_size_z = 0.9
    medium_block_size_x = 1.0
    medium_block_size_y = 1.0
    medium_block_size_z = 0.6

    # Phase 1: Approach Object3 (Large Block)
    # Reward for reducing distance between pelvis and Object3.
    # Using absolute differences for each axis to encourage alignment.
    # This uses relative distances between robot pelvis and Object3.
    dist_pelvis_obj3_x = torch.abs(object3.data.root_pos_w[:, 0] - pelvis_pos_x)
    dist_pelvis_obj3_y = torch.abs(object3.data.root_pos_w[:, 1] - pelvis_pos_y)
    # Sum of distances, negated for reward (closer is better)
    # Only give approach reward when robot is further than 0.8m from cube
    total_dist = torch.sqrt(dist_pelvis_obj3_x**2 + dist_pelvis_obj3_y**2)
    approach_reward = torch.where(total_dist > 0.8, 
                                -(dist_pelvis_obj3_x + dist_pelvis_obj3_y),
                                torch.zeros_like(dist_pelvis_obj3_x))

    # Phase 2 & 3: Push Object3 towards Object2 and align
    # Target position for Object3 relative to Object2 for stair formation.
    # Assuming stairs are built along the Y-axis, with Object3 "behind" Object2 (in +Y direction).
    # Object3's X should align with Object2's X.
    # Object3's Y should be offset by Object2's Y dimension (or Object3's Y dimension if pushing from front).
    # Object3's Z should be on top of Object2, so its center is at Object2's center Z + (Object2_height/2 + Object3_height/2).
    # This uses relative target positions based on Object2's position and hardcoded object dimensions.
    target_obj3_x = object2.data.root_pos_w[:, 0] # Align X with Object2
    target_obj3_y = object2.data.root_pos_w[:, 1] 

    # Distance of Object3 from its target position relative to Object2.
    # Using absolute differences for each axis.
    # This uses relative distances between Object3 and its target relative to Object2.
    dist_obj3_target_x = torch.abs(object3.data.root_pos_w[:, 0] - target_obj3_x)
    dist_obj3_target_y = torch.abs(object3.data.root_pos_w[:, 1] - target_obj3_y)

    # Reward for Object3 reaching its target position. Negated for reward.
    push_align_reward = - (dist_obj3_target_x + dist_obj3_target_y)


    # Use torch.where to switch between rewards based on proximity.
    # This ensures a continuous transition between approach and push/align phases.
    reward = push_align_reward + 0.2 * approach_reward

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_posture_hand_position_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    """
    Shaping reward to encourage stable standing posture (pelvis height) and appropriate hand positioning
    for pushing Object3.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    right_hand_pos_y = right_hand_pos[:, 1]
    right_hand_pos_z = right_hand_pos[:, 2]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    left_hand_pos_y = left_hand_pos[:, 1]
    left_hand_pos_z = left_hand_pos[:, 2]

    # Object dimensions (hardcoded from description)
    large_block_size_y = 1.0
    large_block_size_z = 0.9

    # Pelvis stability reward: encourages pelvis to be at a default stable height.
    # This uses an absolute Z-position for the pelvis, which is allowed sparingly for height.
    pelvis_target_z = 0.7 # Default stable pelvis height, relative to ground (Z=0)
    pelvis_stability_reward = -torch.abs(pelvis_pos_z - pelvis_target_z)

    # Hand positioning for pushing Object3.
    # Target hand position relative to Object3's center for pushing.
    # Assuming robot pushes from -Y side of Object3 towards +Y.
    # Hand Y should be slightly behind the block's edge.
    # Hand Z should be at the block's center height for effective pushing.
    # This uses relative target positions based on Object3's position and hardcoded object dimensions.
    target_hand_y = object3.data.root_pos_w[:, 1] - (large_block_size_y / 2.0) - 0.1 # Slightly behind the block's edge
    target_hand_z = object3.data.root_pos_w[:, 2] # At the block's center height

    # Calculate distances for each hand to the target pushing point.
    # This uses relative distances between robot hands and the target pushing point on Object3.
    dist_right_hand_obj3_y = torch.abs(right_hand_pos_y - target_hand_y)
    dist_right_hand_obj3_z = torch.abs(right_hand_pos_z - target_hand_z)
    dist_left_hand_obj3_y = torch.abs(left_hand_pos_y - target_hand_y)
    dist_left_hand_obj3_z = torch.abs(left_hand_pos_z - target_hand_z)

    # Combine hand positioning rewards. Negated for reward.
    hand_position_reward = - (dist_right_hand_obj3_y + dist_right_hand_obj3_z + dist_left_hand_obj3_y + dist_left_hand_obj3_z)

    # Activation condition: only reward hand positioning when robot is close to Object3.
    # Re-using approach_threshold logic from primary reward for consistency.
    # This uses relative distances between robot pelvis and Object3.
    dist_pelvis_obj3_x = torch.abs(object3.data.root_pos_w[:, 0] - pelvis_pos[:, 0])
    dist_pelvis_obj3_y = torch.abs(object3.data.root_pos_w[:, 1] - pelvis_pos[:, 1])
    dist_pelvis_obj3_z = torch.abs(object3.data.root_pos_w[:, 2] - pelvis_pos[:, 2])
    
    # Thresholds are hardcoded as per prompt's allowance for "arbitrary thresholds" in the plan.
    approach_threshold_x = 1.0
    approach_threshold_y = 1.0
    approach_threshold_z = 1.0
    is_close_to_obj3 = (dist_pelvis_obj3_x < approach_threshold_x) & \
                       (dist_pelvis_obj3_y < approach_threshold_y) & \
                       (dist_pelvis_obj3_z < approach_threshold_z)

    # Combine pelvis stability (always active) with conditional hand positioning reward.
    # This ensures a continuous reward by adding 0.0 when not active.
    reward = pelvis_stability_reward + torch.where(is_close_to_obj3, hand_position_reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_collision_retreat_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    """
    Shaping reward to encourage collision avoidance between robot body parts and blocks,
    and to encourage the robot to retreat to a safe position after the push is complete.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    right_knee_idx = robot.body_names.index('right_knee_link')
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]

    left_knee_idx = robot.body_names.index('left_knee_link')
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]

    # Object dimensions (hardcoded from description)
    large_block_size_x = 1.0
    large_block_size_y = 1.0
    large_block_size_z = 0.9
    medium_block_size_x = 1.0
    medium_block_size_y = 1.0
    medium_block_size_z = 0.6

    # Define a safe distance for collision avoidance.
    # This is relative to the block's half-size, plus a small clearance.
    # These are not hard-coded positions, but relative clearances.
    # Clearance is a hardcoded value, but it's a buffer, not a target position.
    clearance = 0.1 # Small buffer
    safe_dist_x_obj3 = (large_block_size_x / 2.0) + clearance
    safe_dist_y_obj3 = (large_block_size_y / 2.0) + clearance
    safe_dist_z_obj3 = (large_block_size_z / 2.0) + clearance

    safe_dist_x_obj2 = (medium_block_size_x / 2.0) + clearance
    safe_dist_y_obj2 = (medium_block_size_y / 2.0) + clearance
    safe_dist_z_obj2 = (medium_block_size_z / 2.0) + clearance

    # Collision avoidance for pelvis with Object3
    # This uses relative distances for collision detection.
    dist_pelvis_obj3_x = torch.abs(pelvis_pos[:, 0] - object3.data.root_pos_w[:, 0])
    dist_pelvis_obj3_y = torch.abs(pelvis_pos[:, 1] - object3.data.root_pos_w[:, 1])
    dist_pelvis_obj3_z = torch.abs(pelvis_pos[:, 2] - object3.data.root_pos_w[:, 2])

    # Penalty is a discrete value (-1.0), which is acceptable for collision penalties.
    collision_pelvis_obj3_reward = torch.where(
        (dist_pelvis_obj3_x < safe_dist_x_obj3) & (dist_pelvis_obj3_y < safe_dist_y_obj3) & (dist_pelvis_obj3_z < safe_dist_z_obj3),
        -1.0, # Large penalty for collision
        0.0
    )

    # Collision avoidance for pelvis with Object2
    dist_pelvis_obj2_x = torch.abs(pelvis_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_pelvis_obj2_y = torch.abs(pelvis_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_pelvis_obj2_z = torch.abs(pelvis_pos[:, 2] - object2.data.root_pos_w[:, 2])

    collision_pelvis_obj2_reward = torch.where(
        (dist_pelvis_obj2_x < safe_dist_x_obj2) & (dist_pelvis_obj2_y < safe_dist_y_obj2) & (dist_pelvis_obj2_z < safe_dist_z_obj2),
        -1.0, # Large penalty for collision
        0.0
    )

    # Collision avoidance for right knee with Object3
    dist_r_knee_obj3_x = torch.abs(right_knee_pos[:, 0] - object3.data.root_pos_w[:, 0])
    dist_r_knee_obj3_y = torch.abs(right_knee_pos[:, 1] - object3.data.root_pos_w[:, 1])
    dist_r_knee_obj3_z = torch.abs(right_knee_pos[:, 2] - object3.data.root_pos_w[:, 2])

    collision_r_knee_obj3_reward = torch.where(
        (dist_r_knee_obj3_x < safe_dist_x_obj3) & (dist_r_knee_obj3_y < safe_dist_y_obj3) & (dist_r_knee_obj3_z < safe_dist_z_obj3),
        -1.0,
        0.0
    )

    # Collision avoidance for left knee with Object3
    dist_l_knee_obj3_x = torch.abs(left_knee_pos[:, 0] - object3.data.root_pos_w[:, 0])
    dist_l_knee_obj3_y = torch.abs(left_knee_pos[:, 1] - object3.data.root_pos_w[:, 1])
    dist_l_knee_obj3_z = torch.abs(left_knee_pos[:, 2] - object3.data.root_pos_w[:, 2])

    collision_l_knee_obj3_reward = torch.where(
        (dist_l_knee_obj3_x < safe_dist_x_obj3) & (dist_l_knee_obj3_y < safe_dist_y_obj3) & (dist_l_knee_obj3_z < safe_dist_z_obj3),
        -1.0,
        0.0
    )

    # Collision avoidance for right knee with Object2
    dist_r_knee_obj2_x = torch.abs(right_knee_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_r_knee_obj2_y = torch.abs(right_knee_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_r_knee_obj2_z = torch.abs(right_knee_pos[:, 2] - object2.data.root_pos_w[:, 2])

    collision_r_knee_obj2_reward = torch.where(
        (dist_r_knee_obj2_x < safe_dist_x_obj2) & (dist_r_knee_obj2_y < safe_dist_y_obj2) & (dist_r_knee_obj2_z < safe_dist_z_obj2),
        -1.0,
        0.0
    )

    # Collision avoidance for left knee with Object2
    dist_l_knee_obj2_x = torch.abs(left_knee_pos[:, 0] - object2.data.root_pos_w[:, 0])
    dist_l_knee_obj2_y = torch.abs(left_knee_pos[:, 1] - object2.data.root_pos_w[:, 1])
    dist_l_knee_obj2_z = torch.abs(left_knee_pos[:, 2] - object2.data.root_pos_w[:, 2])

    collision_l_knee_obj2_reward = torch.where(
        (dist_l_knee_obj2_x < safe_dist_x_obj2) & (dist_l_knee_obj2_y < safe_dist_y_obj2) & (dist_l_knee_obj2_z < safe_dist_z_obj2),
        -1.0,
        0.0
    )

    # Reward for moving away from the blocks after push is complete.
    # First, check if Object3 is already in its target position (re-using logic from primary reward).
    # This uses relative target positions based on Object2's position and hardcoded object dimensions.
    target_obj3_x = object2.data.root_pos_w[:, 0]
    target_obj3_y = object2.data.root_pos_w[:, 1] + medium_block_size_y
    target_obj3_z = object2.data.root_pos_w[:, 2] + (medium_block_size_z / 2.0) + (large_block_size_z / 2.0)

    # This uses relative distances for checking if Object3 is in place.
    dist_obj3_target_x = torch.abs(object3.data.root_pos_w[:, 0] - target_obj3_x)
    dist_obj3_target_y = torch.abs(object3.data.root_pos_w[:, 1] - target_obj3_y)
    dist_obj3_target_z = torch.abs(object3.data.root_pos_w[:, 2] - target_obj3_z)

    # Define a small tolerance for "in place"
    # Tolerance is a hardcoded value, but it's a threshold for "in place", not a target position.
    in_place_tolerance = 0.1
    is_obj3_in_place = (dist_obj3_target_x < in_place_tolerance) & \
                       (dist_obj3_target_y < in_place_tolerance) & \
                       (dist_obj3_target_z < in_place_tolerance)

    # Reward for increasing distance from pelvis to the blocks (e.g., Object3)
    # but only if Object3 is in place and robot is not too far already.
    # Target retreat position: 1.0m behind Object3 along the Y-axis (assuming robot pushed from -Y).
    # This uses a relative target position for retreat.
    retreat_target_y = object3.data.root_pos_w[:, 1] - 1.0
    dist_pelvis_retreat_y = torch.abs(pelvis_pos[:, 1] - retreat_target_y)

    # Reward is negative distance, so smaller distance to target retreat is better.
    # Cap the retreat reward to prevent overshooting and encourage being at the specific spot.
    # Retreat distance threshold is a hardcoded value, but it's a range, not a target position.
    retreat_dist_threshold = 2.0 # meters, max distance to reward retreat
    retreat_reward_raw = -dist_pelvis_retreat_y
    
    # Only apply retreat reward if Object3 is in place and pelvis is within a reasonable range of the retreat target.
    # This ensures the reward is continuous and only active when relevant.
    retreat_reward = torch.where(
        is_obj3_in_place & (dist_pelvis_retreat_y < retreat_dist_threshold),
        retreat_reward_raw,
        torch.tensor(0.0, device=env.device)
    )

    # Sum all collision penalties and the retreat reward.
    reward = collision_pelvis_obj3_reward + collision_pelvis_obj2_reward + \
             collision_r_knee_obj3_reward + collision_l_knee_obj3_reward + \
             collision_r_knee_obj2_reward + collision_l_knee_obj2_reward + \
             retreat_reward

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
    # Primary reward for pushing and aligning the large block.
    # Weight 1.0 as it's the main objective.
    PrimaryPushAlignReward = RewTerm(func=primary_push_align_reward, weight=1.0,
                                     params={"normalise": True, "normaliser_name": "primary_push_align_reward"})

    # Shaping reward for maintaining posture and correct hand positioning.
    # Weight 0.4 to guide behavior without overpowering the main goal.
    ShapingPostureHandPositionReward = RewTerm(func=shaping_posture_hand_position_reward, weight=0.0,
                                               params={"normalise": True, "normaliser_name": "shaping_posture_hand_position_reward"})

    # Shaping reward for collision avoidance and retreating after the push.
    # Weight 0.3 to penalize undesirable collisions and encourage final positioning.
    ShapingCollisionRetreatReward = RewTerm(func=shaping_collision_retreat_reward, weight=0.0,
                                            params={"normalise": True, "normaliser_name": "shaping_collision_retreat_reward"})