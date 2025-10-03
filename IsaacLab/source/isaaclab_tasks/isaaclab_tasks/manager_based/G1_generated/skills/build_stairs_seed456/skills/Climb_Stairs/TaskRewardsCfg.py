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


def climb_stairs_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "climb_stairs_primary_reward") -> torch.Tensor:
    """
    Primary reward for the Climb_Stairs skill.
    Encourages the robot to sequentially jump onto Object1, then Object2, then Object3, and finally stabilize on Object3.
    Rewards horizontal proximity to the target block and vertical alignment with its top surface.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Hardcode object dimensions from the object configuration (x=1m, y=1m, z=height)
    # This is a critical requirement as object dimensions cannot be accessed from RigidObject.
    object1_height = 0.3
    object2_height = 0.6
    object3_height = 0.9
    block_width = 1.0 # Assuming 1m x 1m for all blocks' horizontal dimensions

    # Access the required robot parts using approved patterns
    robot = env.scene["robot"]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Calculate average foot position for robust tracking, handling batched environments
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    # Define conditions for feet being on each block
    # Feet are considered "on" a block if they are horizontally within its bounds (block_width/2 from center)
    # and vertically close to its top surface (within 0.1m tolerance).
    # This uses relative distances between foot position and block position.
    feet_on_object1_x = torch.abs(avg_foot_pos_x - object1.data.root_pos_w[:, 0]) < block_width / 2
    feet_on_object1_y = torch.abs(avg_foot_pos_y - object1.data.root_pos_w[:, 1]) < block_width / 2
    feet_on_object1_z = torch.abs(avg_foot_pos_z - (object1.data.root_pos_w[:, 2] + object1_height / 2)) < 0.1
    feet_on_object1 = feet_on_object1_x & feet_on_object1_y & feet_on_object1_z

    feet_on_object2_x = torch.abs(avg_foot_pos_x - object2.data.root_pos_w[:, 0]) < block_width / 2
    feet_on_object2_y = torch.abs(avg_foot_pos_y - object2.data.root_pos_w[:, 1]) < block_width / 2
    feet_on_object2_z = torch.abs(avg_foot_pos_z - (object2.data.root_pos_w[:, 2] + object2_height / 2)) < 0.1
    feet_on_object2 = feet_on_object2_x & feet_on_object2_y & feet_on_object2_z

    feet_on_object3_x = torch.abs(avg_foot_pos_x - object3.data.root_pos_w[:, 0]) < block_width / 2
    feet_on_object3_y = torch.abs(avg_foot_pos_y - object3.data.root_pos_w[:, 1]) < block_width / 2
    feet_on_object3_z = torch.abs(avg_foot_pos_z - (object3.data.root_pos_w[:, 2] + object3_height / 2)) < 0.1
    feet_on_object3 = feet_on_object3_x & feet_on_object3_y & feet_on_object3_z

    # Determine the current target block based on progression
    # Initialize target to Object1
    target_pos_x = object1.data.root_pos_w[:, 0]
    target_pos_y = object1.data.root_pos_w[:, 1]
    target_pos_z = object1.data.root_pos_w[:, 2]
    target_height = torch.tensor(object1_height, device=env.device)

    # If feet are on Object1, the next target is Object2
    target_pos_x = torch.where(feet_on_object1, object2.data.root_pos_w[:, 0], target_pos_x)
    target_pos_y = torch.where(feet_on_object1, object2.data.root_pos_w[:, 1], target_pos_y)
    target_pos_z = torch.where(feet_on_object1, object2.data.root_pos_w[:, 2], target_pos_z)
    target_height = torch.where(feet_on_object1, torch.tensor(object2_height, device=env.device), target_height)

    # If feet are on Object2, the next target is Object3
    target_pos_x = torch.where(feet_on_object2, object3.data.root_pos_w[:, 0], target_pos_x)
    target_pos_y = torch.where(feet_on_object2, object3.data.root_pos_w[:, 1], target_pos_y)
    target_pos_z = torch.where(feet_on_object2, object3.data.root_pos_w[:, 2], target_pos_z)
    target_height = torch.where(feet_on_object2, torch.tensor(object3_height, device=env.device), target_height)

    # Reward for horizontal distance to target block (negative distance, so closer is higher reward)
    # This uses relative distances between average foot position and target block center.
    dist_x_to_target = target_pos_x - avg_foot_pos_x
    dist_y_to_target = target_pos_y - avg_foot_pos_y
    horizontal_distance_to_target = torch.sqrt(dist_x_to_target**2 + dist_y_to_target**2)
    horizontal_distance_reward = -horizontal_distance_to_target # Continuous, negative reward

    # Reward for vertical alignment with target block's top surface
    # This encourages landing on top of the block.
    # Uses relative distance between average foot Z and target block's top surface Z.
    target_surface_z = target_pos_z + target_height / 2
    vertical_alignment_reward = -torch.abs(avg_foot_pos_z - target_surface_z) # Continuous, negative reward

    # Combine rewards: vertical alignment is more critical when horizontally close.
    # Use an exponential factor to increase the weight of vertical alignment as horizontal distance decreases.
    horizontal_proximity_threshold = 0.5 # meters: within this distance, vertical alignment becomes more important
    # The factor increases from 0 towards 1 as horizontal_distance_to_target approaches 0.
    horizontal_proximity_factor = torch.exp(-5 * (horizontal_distance_to_target - horizontal_proximity_threshold).clamp(min=0))

    primary_reward = horizontal_distance_reward + (vertical_alignment_reward * horizontal_proximity_factor)

    # Final stability reward for being stable on Object3
    # This reward is applied only when the robot has successfully reached Object3.
    # It encourages the pelvis to be at a stable height above Object3's surface and feet to be on Object3.
    pelvis_stable_z_target_on_obj3 = object3.data.root_pos_w[:, 2] + object3_height / 2 + 0.7 # 0.7m above block surface
    # Penalize deviation of pelvis Z from target stable height
    pelvis_stability_on_obj3_reward = -torch.abs(pelvis_pos[:, 2] - pelvis_stable_z_target_on_obj3)
    # Penalize deviation of feet Z from Object3's top surface
    feet_on_obj3_vertical_reward = -torch.abs(avg_foot_pos_z - (object3.data.root_pos_w[:, 2] + object3_height / 2))
    final_stability_reward = pelvis_stability_on_obj3_reward + feet_on_obj3_vertical_reward

    # Apply the final stability reward only when feet are on Object3
    # This ensures a clear progression: first reach Object1, then Object2, then Object3 and stabilize.
    reward = torch.where(feet_on_object3, final_stability_reward, primary_reward)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_for_jumping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to raise its pelvis height when horizontally close to the target block
    but not yet on it, facilitating a jump. Discourages excessive height when far away or already on a block.
    Includes a general penalty for pelvis being too low when not on a block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts (re-using logic from primary reward for consistency)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    avg_foot_pos_x = (left_foot_pos[:, 0] + right_foot_pos[:, 0]) / 2
    avg_foot_pos_y = (left_foot_pos[:, 1] + right_foot_pos[:, 1]) / 2
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2

    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    object1_height = 0.3
    object2_height = 0.6
    object3_height = 0.9
    block_width = 1.0

    # Determine current target block and its properties (same logic as primary reward)
    feet_on_object1_x = torch.abs(avg_foot_pos_x - object1.data.root_pos_w[:, 0]) < block_width / 2
    feet_on_object1_y = torch.abs(avg_foot_pos_y - object1.data.root_pos_w[:, 1]) < block_width / 2
    feet_on_object1_z = torch.abs(avg_foot_pos_z - (object1.data.root_pos_w[:, 2] + object1_height / 2)) < 0.1
    feet_on_object1 = feet_on_object1_x & feet_on_object1_y & feet_on_object1_z

    feet_on_object2_x = torch.abs(avg_foot_pos_x - object2.data.root_pos_w[:, 0]) < block_width / 2
    feet_on_object2_y = torch.abs(avg_foot_pos_y - object2.data.root_pos_w[:, 1]) < block_width / 2
    feet_on_object2_z = torch.abs(avg_foot_pos_z - (object2.data.root_pos_w[:, 2] + object2_height / 2)) < 0.1
    feet_on_object2 = feet_on_object2_x & feet_on_object2_y & feet_on_object2_z

    feet_on_object3_x = torch.abs(avg_foot_pos_x - object3.data.root_pos_w[:, 0]) < block_width / 2
    feet_on_object3_y = torch.abs(avg_foot_pos_y - object3.data.root_pos_w[:, 1]) < block_width / 2
    feet_on_object3_z = torch.abs(avg_foot_pos_z - (object3.data.root_pos_w[:, 2] + object3_height / 2)) < 0.1
    feet_on_object3 = feet_on_object3_x & feet_on_object3_y & feet_on_object3_z

    target_pos_x = object1.data.root_pos_w[:, 0]
    target_pos_y = object1.data.root_pos_w[:, 1]
    target_pos_z = object1.data.root_pos_w[:, 2]
    target_height = torch.tensor(object1_height, device=env.device)

    target_pos_x = torch.where(feet_on_object1, object2.data.root_pos_w[:, 0], target_pos_x)
    target_pos_y = torch.where(feet_on_object1, object2.data.root_pos_w[:, 1], target_pos_y)
    target_pos_z = torch.where(feet_on_object1, object2.data.root_pos_w[:, 2], target_pos_z)
    target_height = torch.where(feet_on_object1, torch.tensor(object2_height, device=env.device), target_height)

    target_pos_x = torch.where(feet_on_object2, object3.data.root_pos_w[:, 0], target_pos_x)
    target_pos_y = torch.where(feet_on_object2, object3.data.root_pos_w[:, 1], target_pos_y)
    target_pos_z = torch.where(feet_on_object2, object3.data.root_pos_w[:, 2], target_pos_z)
    target_height = torch.where(feet_on_object2, torch.tensor(object3_height, device=env.device), target_height)

    # Calculate horizontal distance from pelvis to current target block
    # Uses relative distance between pelvis position and target block center.
    dist_x_pelvis = target_pos_x - pelvis_pos[:, 0]
    dist_y_pelvis = target_pos_y - pelvis_pos[:, 1]
    horizontal_dist_pelvis = torch.sqrt(dist_x_pelvis**2 + dist_y_pelvis**2)

    # Target pelvis height for jumping: slightly above the target block's top surface
    # This encourages the robot to lift its center of mass for a jump.
    target_pelvis_jump_height = target_pos_z + target_height / 2 + 0.3 # 0.3m clearance above block surface

    # Reward for increasing pelvis height when approaching the block
    # This reward is active when the robot is horizontally close to the block (e.g., within 1.5m)
    # and its feet are not yet on the block it's targeting.
    approach_threshold = 1.5 # meters

    # Initialize reward to 0.0
    pelvis_height_reward = torch.tensor(0.0, device=env.device)

    # Condition for approaching Object1 (not on Object1 yet)
    condition_approach_obj1 = (horizontal_dist_pelvis < approach_threshold) & (~feet_on_object1)
    pelvis_height_reward_obj1 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_jump_height)
    pelvis_height_reward = torch.where(condition_approach_obj1, pelvis_height_reward_obj1, pelvis_height_reward)

    # Condition for approaching Object2 (feet are on Object1, but not on Object2 yet)
    condition_approach_obj2 = (horizontal_dist_pelvis < approach_threshold) & feet_on_object1 & (~feet_on_object2)
    pelvis_height_reward_obj2 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_jump_height)
    pelvis_height_reward = torch.where(condition_approach_obj2, pelvis_height_reward_obj2, pelvis_height_reward)

    # Condition for approaching Object3 (feet are on Object2, but not on Object3 yet)
    condition_approach_obj3 = (horizontal_dist_pelvis < approach_threshold) & feet_on_object2 & (~feet_on_object3)
    pelvis_height_reward_obj3 = -torch.abs(pelvis_pos[:, 2] - target_pelvis_jump_height)
    pelvis_height_reward = torch.where(condition_approach_obj3, pelvis_height_reward_obj3, pelvis_height_reward)

    # Add a small penalty for pelvis being too low when not on a block, to encourage standing.
    # This helps maintain general stability.
    pelvis_standing_height = 0.7 # default standing height for the robot
    pelvis_too_low_penalty = -torch.abs(pelvis_pos[:, 2] - pelvis_standing_height)
    # Only apply if not on any block and pelvis is below standing height
    not_on_any_block = ~(feet_on_object1 | feet_on_object2 | feet_on_object3)
    pelvis_height_reward += torch.where(not_on_any_block & (pelvis_pos[:, 2] < pelvis_standing_height), pelvis_too_low_penalty * 0.1, torch.tensor(0.0, device=env.device))

    shaping_reward1 = pelvis_height_reward

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 2: Penalizes the robot if its hands or feet collide with the sides of the blocks
    during the jump or approach. Encourages clearing the blocks.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Hardcode object dimensions
    object1_height = 0.3
    object2_height = 0.6
    object3_height = 0.9
    block_width = 1.0 # Assuming 1m x 1m for all blocks

    # Access the required robot parts
    robot = env.scene["robot"]
    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Define a small clearance margin around the block for collision detection
    clearance_margin = 0.1 # 10 cm around the block's horizontal and vertical sides

    # Initialize as per-environment tensor to avoid broadcasting issues
    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Helper function to calculate penalty for a given part and block
    # This function uses relative distances to determine if a part is within a block's extended bounding box.
    def calculate_block_collision_penalty(part_pos, block_obj, block_height):
        block_center_x = block_obj.data.root_pos_w[:, 0]
        block_center_y = block_obj.data.root_pos_w[:, 1]
        block_bottom_z = block_obj.data.root_pos_w[:, 2] - block_height / 2
        block_top_z = block_obj.data.root_pos_w[:, 2] + block_height / 2

        # Check if part is within the horizontal bounds of the block + margin
        # Uses relative distances for x and y.
        within_x = (part_pos[:, 0] > block_center_x - block_width / 2 - clearance_margin) & \
                   (part_pos[:, 0] < block_center_x + block_width / 2 + clearance_margin)
        within_y = (part_pos[:, 1] > block_center_y - block_width / 2 - clearance_margin) & \
                   (part_pos[:, 1] < block_center_y + block_width / 2 + clearance_margin)

        # Check if part is within the vertical bounds of the block + margin for side collision
        # Uses relative distances for z.
        within_z_vertical_collision = (part_pos[:, 2] > block_bottom_z - clearance_margin) & \
                                      (part_pos[:, 2] < block_top_z + clearance_margin)

        # Penalty for hands colliding with any part of the block (sides or top/bottom edges)
        # Hands should generally not touch the blocks.
        hand_collision_condition = within_x & within_y & within_z_vertical_collision
        hand_penalty = torch.where(hand_collision_condition, torch.tensor(-0.5, device=env.device), torch.tensor(0.0, device=env.device))

        # Penalty for feet colliding with the sides of the block (not landing on top)
        # Feet are considered colliding if they are within the block's horizontal bounds
        # AND their Z is significantly below the top surface (i.e., hitting the side or bottom edge).
        # A small tolerance (0.05m) is given for being on top.
        foot_side_collision_condition = within_x & within_y & (part_pos[:, 2] < block_top_z - 0.05)
        foot_penalty = torch.where(foot_side_collision_condition, torch.tensor(-0.5, device=env.device), torch.tensor(0.0, device=env.device))

        return hand_penalty, foot_penalty

    # Accumulate penalties for all relevant body parts and blocks
    for block_obj, block_height in [(object1, object1_height), (object2, object2_height), (object3, object3_height)]:
        hand_pen_l, _ = calculate_block_collision_penalty(left_hand_pos, block_obj, block_height)
        hand_pen_r, _ = calculate_block_collision_penalty(right_hand_pos, block_obj, block_height)
        _, foot_pen_l = calculate_block_collision_penalty(left_foot_pos, block_obj, block_height) # Only care about foot side collision
        _, foot_pen_r = calculate_block_collision_penalty(right_foot_pos, block_obj, block_height)

        collision_penalty += hand_pen_l + hand_pen_r + foot_pen_l + foot_pen_r

    shaping_reward2 = collision_penalty

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


def maintain_pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_reward") -> torch.Tensor:
    """
    Shaping reward 3: Encourages the robot to keep its pelvis at a stable height (around 0.7m)
    when it is not actively jumping or moving between blocks. This helps prevent the robot from
    falling or crouching unnecessarily.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot parts
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Define the target stable pelvis height (absolute Z position)
    pelvis_stable_height = 0.7 # meters, typical standing height for this robot

    # Condition: Robot is not actively jumping high.
    # This is a general stability reward, so it should be active most of the time,
    # but less critical when the primary reward is driving a jump.
    # A simple condition is if the average foot height is relatively low (e.g., below 0.5m, indicating not mid-jump).
    # This uses the absolute Z position of the feet.
    avg_foot_pos_z = (left_foot_pos[:, 2] + right_foot_pos[:, 2]) / 2
    is_not_jumping_high = (avg_foot_pos_z < 0.5) # Arbitrary threshold, adjust as needed

    # Reward for keeping pelvis close to the stable height
    # Uses relative distance between pelvis Z and target stable height.
    pelvis_stability_reward = -torch.abs(pelvis_pos[:, 2] - pelvis_stable_height) # Continuous, negative reward

    # Apply the reward when not actively jumping high
    shaping_reward3 = torch.where(is_not_jumping_high, pelvis_stability_reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward3)
        RewNormalizer.update_stats(normaliser_name, shaping_reward3)
        return scaled_reward
    return shaping_reward3


@configclass
class TaskRewardsCfg:
    """
    Configuration for the rewards used in the Climb_Stairs task.
    Defines the primary reward and all shaping rewards with their respective weights.
    """
    # Primary reward for climbing the stairs sequentially
    ClimbStairsPrimaryReward = RewTerm(func=climb_stairs_primary_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "climb_stairs_primary_reward"})

    # Shaping reward for encouraging pelvis height for jumping
    PelvisHeightForJumpingReward = RewTerm(func=pelvis_height_for_jumping_reward, weight=0.4,
                                           params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # Shaping reward for penalizing collisions with blocks
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining general pelvis stability
    MaintainPelvisStabilityReward = RewTerm(func=maintain_pelvis_stability_reward, weight=0.2,
                                            params={"normalise": True, "normaliser_name": "pelvis_stability_reward"})