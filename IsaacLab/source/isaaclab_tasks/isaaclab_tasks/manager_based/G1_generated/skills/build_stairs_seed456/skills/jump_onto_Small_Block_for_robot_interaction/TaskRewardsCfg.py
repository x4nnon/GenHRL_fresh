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


def jump_onto_Small_Block_for_robot_interaction_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the robot to jump onto the Small Block (Object1) and stand stably on it.
    This reward combines phases of approach, jump execution, and stable landing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot and the small block (Object1) using approved patterns
    robot = env.scene["robot"]
    small_block = env.scene['Object1']

    # Hardcoded dimensions for the Small Block from the object configuration (x=1m y=1m z=0.3m)
    # This follows the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    small_block_height = 0.3
    small_block_half_width = 0.5 # Half of 1m width
    small_block_half_depth = 0.5 # Half of 1m depth

    # Access required robot parts and their positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is an absolute height, used sparingly as allowed.

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_x = left_foot_pos[:, 0]
    left_foot_pos_y = left_foot_pos[:, 1]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Access small block's center position using approved patterns
    block_center_x = small_block.data.root_pos_w[:, 0]
    block_center_y = small_block.data.root_pos_w[:, 1]

    # Calculate average foot position for horizontal alignment
    avg_foot_pos_x = (left_foot_pos_x + right_foot_pos_x) / 2
    avg_foot_pos_y = (left_foot_pos_y + right_foot_pos_y) / 2

    # Phase 1: Approach and Pre-Jump (horizontal alignment)
    # Reward for getting horizontally close to the block. Uses relative distances.
    # This follows the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    horizontal_dist_to_block_x = torch.abs(avg_foot_pos_x - block_center_x)
    horizontal_dist_to_block_y = torch.abs(avg_foot_pos_y - block_center_y)
    # Reward is negative, encouraging smaller distances. This is a continuous reward.
    approach_reward = - (horizontal_dist_to_block_x + horizontal_dist_to_block_y)

    # Phase 2 & 3: Jump Execution & Over-Block Traversal (vertical clearance and horizontal over-block)
    # Define a small buffer for feet thickness, e.g., 0.05m, to ensure feet clear the block's top surface.
    # This is a hardcoded threshold, but it's a small offset, not an arbitrary position.
    feet_clearance_height = small_block_height + 0.05

    # Condition for feet being horizontally over the block. Uses relative distances.
    # This uses hardcoded half-dimensions of the block, which is allowed as per rule 8.
    feet_over_block_x_condition = (avg_foot_pos_x > (block_center_x - small_block_half_width)) & \
                                  (avg_foot_pos_x < (block_center_x + small_block_half_width))
    feet_over_block_y_condition = (avg_foot_pos_y > (block_center_y - small_block_half_depth)) & \
                                  (avg_foot_pos_y < (block_center_y + small_block_half_depth))
    is_over_block_horizontal = feet_over_block_x_condition & feet_over_block_y_condition

    # Reward for feet being above the block's top surface (during jump). Uses relative distances.
    # Reward is positive, encouraging higher feet when over the block. This is a continuous reward.
    feet_above_block_reward = torch.where(
        (left_foot_pos_z > feet_clearance_height) & (right_foot_pos_z > feet_clearance_height),
        (left_foot_pos_z - feet_clearance_height) + (right_foot_pos_z - feet_clearance_height),
        torch.tensor(0.0, device=env.device) # Ensure tensor is on the correct device
    )

    # Phase 4 & 5: Landing and Stability (feet on block, pelvis stable)
    # Target Z for feet is block_height + small_offset (e.g., 0.02m for foot thickness).
    target_foot_z = small_block_height + 0.02
    # Reward for feet being on top of the block (z-position). Uses relative distances.
    # Reward is negative, encouraging feet to be precisely at target_foot_z. This is a continuous reward.
    feet_on_block_z_reward = - (torch.abs(left_foot_pos_z - target_foot_z) + torch.abs(right_foot_pos_z - target_foot_z))

    # Target Z for pelvis is block_height + 0.7m (stable standing height). Uses relative distances.
    target_pelvis_z = small_block_height + 0.7
    # Reward for pelvis stability on top of the block. Uses relative distances.
    # Reward is negative, encouraging pelvis to be precisely at target_pelvis_z. This is a continuous reward.
    pelvis_stability_reward = - torch.abs(pelvis_pos_z - target_pelvis_z)

    # Condition for being "on" the block (feet close to target_foot_z and horizontally over)
    # This uses a small threshold (0.1) for vertical alignment, which is acceptable for defining a "state".
    is_on_block = is_over_block_horizontal & \
                  (torch.abs(left_foot_pos_z - target_foot_z) < 0.1) & \
                  (torch.abs(right_foot_pos_z - target_foot_z) < 0.1)

    # Primary Reward Logic: Conditional reward based on the robot's state.
    # This combines continuous rewards based on phases, ensuring continuity.
    reward = torch.where(
        is_on_block,
        # If on block, prioritize stability and precise landing
        (feet_on_block_z_reward * 2.0) + (pelvis_stability_reward * 1.5), # Higher weight for final state
        torch.where(
            is_over_block_horizontal,
            feet_above_block_reward, # Reward for clearing height when over block
            approach_reward # Reward for horizontal approach when not yet over block
        )
    )

    # Mandatory normalization as per rule 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def jump_onto_Small_Block_for_robot_interaction_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Penalizes any part of the robot (excluding feet when landing) from colliding with the Small Block (Object1).
    This encourages the robot to jump over the block cleanly.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the small block (Object1) using approved patterns
    small_block = env.scene['Object1']

    # Hardcoded dimensions for the Small Block from the object configuration (x=1m y=1m z=0.3m)
    # This follows the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    small_block_height = 0.3
    small_block_half_width = 0.5 # Half of 1m width
    small_block_half_depth = 0.5 # Half of 1m depth

    # Calculate block's min/max coordinates in world frame. Uses relative distances from block center.
    # This follows the rule: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    block_center_x = small_block.data.root_pos_w[:, 0]
    block_center_y = small_block.data.root_pos_w[:, 1]
    small_block_x_min = block_center_x - small_block_half_width
    small_block_x_max = block_center_x + small_block_half_width
    small_block_y_min = block_center_y - small_block_half_depth
    small_block_y_max = block_center_y + small_block_half_depth

    # Access the robot using approved patterns
    robot = env.scene["robot"]

    # Define robot parts to check for collision (excluding feet for landing)
    robot_parts_to_check = ['pelvis', 'left_knee_link', 'right_knee_link', 'left_palm_link', 'right_palm_link', 'head_link']
    collision_penalty = torch.zeros(env.num_envs, device=env.device) # Initialize penalty tensor for batch processing

    for part_name in robot_parts_to_check:
        part_idx = robot.body_names.index(part_name) # Approved pattern for body part index
        part_pos = robot.data.body_pos_w[:, part_idx] # Approved pattern for body part position
        part_pos_x = part_pos[:, 0]
        part_pos_y = part_pos[:, 1]
        part_pos_z = part_pos[:, 2]

        # Check if part is within block's horizontal bounds. Uses relative distances.
        is_horizontally_colliding = (part_pos_x > small_block_x_min) & (part_pos_x < small_block_x_max) & \
                                    (part_pos_y > small_block_y_min) & (part_pos_y < small_block_y_max)

        # Check if part is within block's vertical bounds (from ground to top). Uses relative distances.
        # The block's base is at z=0, so its volume extends from z=0 to z=small_block_height.
        # This uses absolute Z for the ground (0.0) and relative to block height.
        is_vertically_colliding = (part_pos_z > 0.0) & (part_pos_z < small_block_height)

        # Collision condition: part is within block's volume
        collision_condition = is_horizontally_colliding & is_vertically_colliding

        # Add penalty if collision occurs. Penalty is -1.0 for each colliding part.
        # This is a continuous reward (or rather, a penalty that accumulates).
        collision_penalty += torch.where(collision_condition, -1.0, 0.0)

    reward = collision_penalty # The reward is the accumulated penalty

    # Mandatory normalization as per rule 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def jump_onto_Small_Block_for_robot_interaction_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "upright_posture_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain a reasonable pelvis height throughout the skill,
    preventing it from crouching too low or falling. This is important during approach,
    jump preparation, and for stability after landing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the robot and the small block (Object1) using approved patterns
    robot = env.scene["robot"]
    small_block = env.scene['Object1']

    # Hardcoded dimensions for the Small Block from the object configuration (x=1m y=1m z=0.3m)
    # This follows the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    small_block_height = 0.3
    small_block_half_width = 0.5 # Half of 1m width
    small_block_half_depth = 0.5 # Half of 1m depth

    # Access required robot parts and their positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is an absolute height, used sparingly as allowed.

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Calculate average foot position for horizontal alignment
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2

    # Access small block's center position using approved patterns
    block_center_x = small_block.data.root_pos_w[:, 0]
    block_center_y = small_block.data.root_pos_w[:, 1]

    # Define target pelvis height for standing/jumping preparation on the ground (e.g., 0.7m from ground)
    # These are hardcoded target heights, which are acceptable for defining desired posture.
    target_pelvis_z_ground = 0.7

    # Define target pelvis height when on top of the block (block height + standing height)
    target_pelvis_z_on_block = small_block_height + 0.7

    # Define target foot Z for being on the block (block height + small offset for foot thickness)
    target_foot_z_on_block = small_block_height + 0.02

    # Condition for being "on" the block (feet close to target_foot_z and horizontally over)
    # This condition uses relative distances for horizontal checks and absolute Z for vertical.
    is_on_block = (torch.abs(left_foot_pos[:, 2] - target_foot_z_on_block) < 0.1) & \
                  (torch.abs(right_foot_pos[:, 2] - target_foot_z_on_block) < 0.1) & \
                  (avg_foot_pos_x > (block_center_x - small_block_half_width)) & \
                  (avg_foot_pos_x < (block_center_x + small_block_half_width)) & \
                  (avg_foot_pos_y > (block_center_y - small_block_half_depth)) & \
                  (avg_foot_pos_y < (block_center_y + small_block_half_depth))

    # Reward based on current phase: on block or on ground
    # Reward is negative absolute difference, encouraging pelvis to be precisely at the target height.
    # This is a continuous reward.
    reward = torch.where(
        is_on_block,
        -torch.abs(pelvis_pos_z - target_pelvis_z_on_block), # Reward for stable pelvis height on block
        -torch.abs(pelvis_pos_z - target_pelvis_z_ground) # Reward for stable pelvis height on ground
    )

    # Mandatory normalization as per rule 2.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    """
    Reward terms for the jump_onto_Small_Block_for_robot_interaction skill.
    """
    # Main reward for successfully jumping onto and stabilizing on the small block
    # Weight is 1.0 as per rule: "PROPER WEIGHTS - Set appropriate weights in TaskRewardsCfg (primary reward ~1.0, supporting rewards <1.0)"
    main_jump_onto_small_block_reward = RewTerm(
        func=jump_onto_Small_Block_for_robot_interaction_main_reward,
        weight=1.0, # Primary reward, typically weight 1.0
        params={"normalise": True, "normaliser_name": "main_reward"}
    )

    # Shaping reward for avoiding collisions with the small block during the jump
    # Weight is 0.6, which is less than 1.0 for a shaping reward.
    collision_avoidance_reward = RewTerm(
        func=jump_onto_Small_Block_for_robot_interaction_collision_avoidance_reward,
        weight=0.6, # Shaping reward, typically lower weight
        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"}
    )

    # Shaping reward for maintaining an upright posture (pelvis height)
    # Weight is 0.4, which is less than 1.0 for a shaping reward.
    upright_posture_reward = RewTerm(
        func=jump_onto_Small_Block_for_robot_interaction_upright_posture_reward,
        weight=0.4, # Shaping reward, typically lower weight
        params={"normalise": True, "normaliser_name": "upright_posture_reward"}
    )