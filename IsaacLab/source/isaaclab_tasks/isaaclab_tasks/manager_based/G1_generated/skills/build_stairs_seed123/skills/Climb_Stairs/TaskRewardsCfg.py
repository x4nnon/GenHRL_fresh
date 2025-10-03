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


def primary_climb_stairs_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_climb_stairs_reward") -> torch.Tensor:
    """
    Primary reward for the Climb_Stairs skill.
    Guides the robot through the sequence of climbing each block (Object1, Object2, Object3)
    and stabilizing on the final block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Block dimensions (hardcoded from object configuration, as per requirements)
    block1_height = 0.3
    block2_height = 0.6
    block3_height = 0.9
    block_xy_size = 1.0 # Assuming 1m x 1m for all blocks

    # Target pelvis height relative to block top (relative distance, not hard-coded absolute)
    target_pelvis_z_offset = 0.7

    # Tolerance for being on top of a block (small buffer for Z-height)
    on_block_z_tolerance = 0.05

    # Conditions for being on a block: both feet within XY bounds and above Z height
    # All positions are relative to object root positions.
    on_block1_condition = (
        (left_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (left_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (left_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (left_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (left_foot_pos[:, 2] > object1.data.root_pos_w[:, 2] + block1_height - on_block_z_tolerance) &
        (right_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (right_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (right_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (right_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (right_foot_pos[:, 2] > object1.data.root_pos_w[:, 2] + block1_height - on_block_z_tolerance)
    )

    on_block2_condition = (
        (left_foot_pos[:, 0] > object2.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (left_foot_pos[:, 0] < object2.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (left_foot_pos[:, 1] > object2.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (left_foot_pos[:, 1] < object2.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (left_foot_pos[:, 2] > object2.data.root_pos_w[:, 2] + block2_height - on_block_z_tolerance) &
        (right_foot_pos[:, 0] > object2.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (right_foot_pos[:, 0] < object2.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (right_foot_pos[:, 1] > object2.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (right_foot_pos[:, 1] < object2.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (right_foot_pos[:, 2] > object2.data.root_pos_w[:, 2] + block2_height - on_block_z_tolerance)
    )

    on_block3_condition = (
        (left_foot_pos[:, 0] > object3.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (left_foot_pos[:, 0] < object3.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (left_foot_pos[:, 1] > object3.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (left_foot_pos[:, 1] < object3.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (left_foot_pos[:, 2] > object3.data.root_pos_w[:, 2] + block3_height - on_block_z_tolerance) &
        (right_foot_pos[:, 0] > object3.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (right_foot_pos[:, 0] < object3.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (right_foot_pos[:, 1] > object3.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (right_foot_pos[:, 1] < object3.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (right_foot_pos[:, 2] > object3.data.root_pos_w[:, 2] + block3_height - on_block_z_tolerance)
    )

    # Reward for approaching and landing on Object1 (relative distances)
    # Average feet position relative to the center and top surface of Object1
    avg_feet_pos = (left_foot_pos + right_foot_pos) / 2
    dist_feet_to_obj1_x = avg_feet_pos[:, 0] - object1.data.root_pos_w[:, 0]
    dist_feet_to_obj1_y = avg_feet_pos[:, 1] - object1.data.root_pos_w[:, 1]
    dist_feet_to_obj1_z = avg_feet_pos[:, 2] - (object1.data.root_pos_w[:, 2] + block1_height)
    reward_approach_obj1 = -torch.abs(dist_feet_to_obj1_x) - torch.abs(dist_feet_to_obj1_y) - torch.abs(dist_feet_to_obj1_z)

    # Reward for stabilizing pelvis on Object1 (relative distance)
    pelvis_z_on_obj1_target = object1.data.root_pos_w[:, 2] + block1_height + target_pelvis_z_offset
    reward_stabilize_obj1 = -torch.abs(pelvis_pos[:, 2] - pelvis_z_on_obj1_target)

    # Reward for approaching and landing on Object2 (relative distances)
    dist_feet_to_obj2_x = avg_feet_pos[:, 0] - object2.data.root_pos_w[:, 0]
    dist_feet_to_obj2_y = avg_feet_pos[:, 1] - object2.data.root_pos_w[:, 1]
    dist_feet_to_obj2_z = avg_feet_pos[:, 2] - (object2.data.root_pos_w[:, 2] + block2_height)
    reward_approach_obj2 = -torch.abs(dist_feet_to_obj2_x) - torch.abs(dist_feet_to_obj2_y) - torch.abs(dist_feet_to_obj2_z)

    # Reward for stabilizing pelvis on Object2 (relative distance)
    pelvis_z_on_obj2_target = object2.data.root_pos_w[:, 2] + block2_height + target_pelvis_z_offset
    reward_stabilize_obj2 = -torch.abs(pelvis_pos[:, 2] - pelvis_z_on_obj2_target)

    # Reward for approaching and landing on Object3 (relative distances)
    dist_feet_to_obj3_x = avg_feet_pos[:, 0] - object3.data.root_pos_w[:, 0]
    dist_feet_to_obj3_y = avg_feet_pos[:, 1] - object3.data.root_pos_w[:, 1]
    dist_feet_to_obj3_z = avg_feet_pos[:, 2] - (object3.data.root_pos_w[:, 2] + block3_height)
    reward_approach_obj3 = -torch.abs(dist_feet_to_obj3_x) - torch.abs(dist_feet_to_obj3_y) - torch.abs(dist_feet_to_obj3_z)

    # Reward for stabilizing pelvis on Object3 (final goal, relative distance)
    pelvis_z_on_obj3_target = object3.data.root_pos_w[:, 2] + block3_height + target_pelvis_z_offset
    reward_stabilize_obj3 = -torch.abs(pelvis_pos[:, 2] - pelvis_z_on_obj3_target)

    # Combine rewards based on progression using torch.where for batch compatibility
    primary_reward = torch.zeros_like(reward_approach_obj1)

    # Phase 4: Stabilize on Object3 (final goal)
    primary_reward = torch.where(on_block3_condition, reward_stabilize_obj3, primary_reward)
    # Phase 3: Stabilize on Object2, then approach and land on Object3
    primary_reward = torch.where(on_block2_condition & ~on_block3_condition, reward_stabilize_obj2 + reward_approach_obj3, primary_reward)
    # Phase 2: Stabilize on Object1, then approach and land on Object2
    primary_reward = torch.where(on_block1_condition & ~on_block2_condition, reward_stabilize_obj1 + reward_approach_obj2, primary_reward)
    # Phase 1: Approach and land on Object1 (initial phase)
    primary_reward = torch.where(~on_block1_condition, reward_approach_obj1, primary_reward)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def shaping_pelvis_height_jump_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_pelvis_height_jump_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to lift its pelvis sufficiently high when attempting a jump.
    Active when the robot is not yet on the next block but has left the previous one.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Block dimensions (hardcoded from object configuration)
    block1_height = 0.3
    block2_height = 0.6
    block3_height = 0.9
    block_xy_size = 1.0

    # Tolerance for being on top of a block (re-used for consistency)
    on_block_z_tolerance = 0.05

    # Conditions for being on a block (re-used from primary reward for consistency)
    on_block1_condition = (
        (left_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (left_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (left_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (left_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (left_foot_pos[:, 2] > object1.data.root_pos_w[:, 2] + block1_height - on_block_z_tolerance) &
        (right_foot_pos[:, 0] > object1.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (right_foot_pos[:, 0] < object1.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (right_foot_pos[:, 1] > object1.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (right_foot_pos[:, 1] < object1.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (right_foot_pos[:, 2] > object1.data.root_pos_w[:, 2] + block1_height - on_block_z_tolerance)
    )

    on_block2_condition = (
        (left_foot_pos[:, 0] > object2.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (left_foot_pos[:, 0] < object2.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (left_foot_pos[:, 1] > object2.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (left_foot_pos[:, 1] < object2.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (left_foot_pos[:, 2] > object2.data.root_pos_w[:, 2] + block2_height - on_block_z_tolerance) &
        (right_foot_pos[:, 0] > object2.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (right_foot_pos[:, 0] < object2.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (right_foot_pos[:, 1] > object2.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (right_foot_pos[:, 1] < object2.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (right_foot_pos[:, 2] > object2.data.root_pos_w[:, 2] + block2_height - on_block_z_tolerance)
    )

    on_block3_condition = (
        (left_foot_pos[:, 0] > object3.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (left_foot_pos[:, 0] < object3.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (left_foot_pos[:, 1] > object3.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (left_foot_pos[:, 1] < object3.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (left_foot_pos[:, 2] > object3.data.root_pos_w[:, 2] + block3_height - on_block_z_tolerance) &
        (right_foot_pos[:, 0] > object3.data.root_pos_w[:, 0] - block_xy_size / 2) &
        (right_foot_pos[:, 0] < object3.data.root_pos_w[:, 0] + block_xy_size / 2) &
        (right_foot_pos[:, 1] > object3.data.root_pos_w[:, 1] - block_xy_size / 2) &
        (right_foot_pos[:, 1] < object3.data.root_pos_w[:, 1] + block_xy_size / 2) &
        (right_foot_pos[:, 2] > object3.data.root_pos_w[:, 2] + block3_height - on_block_z_tolerance)
    )

    # Target jump height for pelvis (relative to current block's top + clearance)
    jump_clearance = 0.2 # 20cm clearance above the next block's top

    # Reward for jumping from ground to Object1
    # Target pelvis Z is relative to Object1's top surface + clearance
    target_pelvis_z_jump1 = object1.data.root_pos_w[:, 2] + block1_height + jump_clearance
    reward_jump_obj1 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_jump1)

    # Reward for jumping from Object1 to Object2
    # Target pelvis Z is relative to Object2's top surface + clearance
    target_pelvis_z_jump2 = object2.data.root_pos_w[:, 2] + block2_height + jump_clearance
    reward_jump_obj2 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_jump2)

    # Reward for jumping from Object2 to Object3
    # Target pelvis Z is relative to Object3's top surface + clearance
    target_pelvis_z_jump3 = object3.data.root_pos_w[:, 2] + block3_height + jump_clearance
    reward_jump_obj3 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z_jump3)

    shaping_reward1 = torch.zeros_like(reward_jump_obj1)

    # Active when not on Object1 yet, but pelvis is above ground (indicating a jump attempt)
    # Pelvis Z > object1.data.root_pos_w[:, 2] + 0.1 (0.1m above ground level of Object1)
    shaping_reward1 = torch.where(~on_block1_condition & (pelvis_pos[:, 2] > object1.data.root_pos_w[:, 2] + 0.1), reward_jump_obj1, shaping_reward1)

    # Active when on Object1 but not yet on Object2, and pelvis is above Object1's top (indicating a jump attempt)
    # Pelvis Z > object1.data.root_pos_w[:, 2] + block1_height + 0.1 (0.1m above Object1's top)
    shaping_reward1 = torch.where(on_block1_condition & ~on_block2_condition & (pelvis_pos[:, 2] > object1.data.root_pos_w[:, 2] + block1_height + 0.1), reward_jump_obj2, shaping_reward1)

    # Active when on Object2 but not yet on Object3, and pelvis is above Object2's top (indicating a jump attempt)
    # Pelvis Z > object2.data.root_pos_w[:, 2] + block2_height + 0.1 (0.1m above Object2's top)
    shaping_reward1 = torch.where(on_block2_condition & ~on_block3_condition & (pelvis_pos[:, 2] > object2.data.root_pos_w[:, 2] + block2_height + 0.1), reward_jump_obj3, shaping_reward1)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to penalize collisions between robot body parts and the sides/bottoms of the blocks.
    Encourages clear jumps and precise landings.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access the required robot part(s)
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')
    left_knee_idx = robot.body_names.index('left_knee_link')
    right_knee_idx = robot.body_names.index('right_knee_link')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]

    # Block dimensions (hardcoded from object configuration)
    block1_height = 0.3
    block2_height = 0.6
    block3_height = 0.9
    block_xy_size = 1.0

    # Define a small buffer for collision detection around block sides
    collision_buffer = 0.1 # 10cm buffer

    # Penalty value for a collision
    collision_penalty_value = -0.5

    # Function to calculate collision penalty for a robot part with a block
    # Uses relative positions to check if a robot part is "colliding" with the block's sides/bottom
    def calculate_block_collision_penalty(robot_part_pos, block_obj, block_height):
        block_center_x = block_obj.data.root_pos_w[:, 0]
        block_center_y = block_obj.data.root_pos_w[:, 1]
        block_bottom_z = block_obj.data.root_pos_w[:, 2]
        block_top_z = block_bottom_z + block_height

        # Check if robot part is within the X/Y bounds of the block (with buffer)
        within_xy = (robot_part_pos[:, 0] > block_center_x - block_xy_size / 2 - collision_buffer) & \
                    (robot_part_pos[:, 0] < block_center_x + block_xy_size / 2 + collision_buffer) & \
                    (robot_part_pos[:, 1] > block_center_y - block_xy_size / 2 - collision_buffer) & \
                    (robot_part_pos[:, 1] < block_center_y + block_xy_size / 2 + collision_buffer)

        # Check if robot part is at or below the top surface of the block (with buffer)
        # This focuses on side/bottom collisions, not standing on top.
        below_top_surface = (robot_part_pos[:, 2] < block_top_z + collision_buffer)

        # Check if robot part is above the ground (to avoid penalizing ground contact when not near a block)
        # This is relative to the block's bottom Z, ensuring it's not just on the floor far away.
        above_block_bottom = (robot_part_pos[:, 2] > block_bottom_z - 0.1) # Small buffer below block bottom

        # Collision occurs if within XY bounds, below top surface, and above block's bottom
        collision_condition = within_xy & below_top_surface & above_block_bottom

        # Penalty is a constant negative value if collision occurs
        penalty = torch.where(collision_condition, torch.tensor(collision_penalty_value, device=env.device), torch.tensor(0.0, device=env.device))
        return penalty

    # Calculate penalties for each relevant robot part with each block
    penalty_left_foot_obj1 = calculate_block_collision_penalty(left_foot_pos, object1, block1_height)
    penalty_right_foot_obj1 = calculate_block_collision_penalty(right_foot_pos, object1, block1_height)
    penalty_left_knee_obj1 = calculate_block_collision_penalty(left_knee_pos, object1, block1_height)
    penalty_right_knee_obj1 = calculate_block_collision_penalty(right_knee_pos, object1, block1_height)
    penalty_pelvis_obj1 = calculate_block_collision_penalty(pelvis_pos, object1, block1_height)

    penalty_left_foot_obj2 = calculate_block_collision_penalty(left_foot_pos, object2, block2_height)
    penalty_right_foot_obj2 = calculate_block_collision_penalty(right_foot_pos, object2, block2_height)
    penalty_left_knee_obj2 = calculate_block_collision_penalty(left_knee_pos, object2, block2_height)
    penalty_right_knee_obj2 = calculate_block_collision_penalty(right_knee_pos, object2, block2_height)
    penalty_pelvis_obj2 = calculate_block_collision_penalty(pelvis_pos, object2, block2_height)

    penalty_left_foot_obj3 = calculate_block_collision_penalty(left_foot_pos, object3, block3_height)
    penalty_right_foot_obj3 = calculate_block_collision_penalty(right_foot_pos, object3, block3_height)
    penalty_left_knee_obj3 = calculate_block_collision_penalty(left_knee_pos, object3, block3_height)
    penalty_right_knee_obj3 = calculate_block_collision_penalty(right_knee_pos, object3, block3_height)
    penalty_pelvis_obj3 = calculate_block_collision_penalty(pelvis_pos, object3, block3_height)

    # Sum all penalties
    shaping_reward2 = (
        penalty_left_foot_obj1 + penalty_right_foot_obj1 + penalty_left_knee_obj1 + penalty_right_knee_obj1 + penalty_pelvis_obj1 +
        penalty_left_foot_obj2 + penalty_right_foot_obj2 + penalty_left_knee_obj2 + penalty_right_knee_obj2 + penalty_pelvis_obj2 +
        penalty_left_foot_obj3 + penalty_right_foot_obj3 + penalty_left_knee_obj3 + penalty_right_knee_obj3 + penalty_pelvis_obj3
    )

    # Mandatory reward normalization
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
    Reward terms for the Climb_Stairs skill.
    """
    # Primary reward for progressing through the climbing sequence
    PrimaryClimbStairsReward = RewTerm(func=primary_climb_stairs_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "primary_climb_stairs_reward"})

    # Shaping reward for encouraging proper pelvis height during jumps
    ShapingPelvisHeightJumpReward = RewTerm(func=shaping_pelvis_height_jump_reward, weight=0.4,
                                            params={"normalise": True, "normaliser_name": "shaping_pelvis_height_jump_reward"})

    # Shaping reward for penalizing collisions with blocks
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.3,
                                              params={"normalise": True, "normaliser_name": "shaping_collision_avoidance_reward"})