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

# Object dimensions (hardcoded from task description as per requirements)
# Object3: Low wall (5m in y, 0.5m in z, 0.3m in x)
LOW_WALL_HEIGHT = 0.5
LOW_WALL_DEPTH_X = 0.3
LOW_WALL_WIDTH_Y = 5.0

# Object1: Large sphere (1m radius)
LARGE_SPHERE_RADIUS = 1.0

# Object4: High wall (5m in y, 1m in z, 0.3m in x)
HIGH_WALL_HEIGHT = 1.0
HIGH_WALL_DEPTH_X = 0.3
HIGH_WALL_WIDTH_Y = 5.0

# Object2: Small sphere (0.2m radius)
SMALL_SPHERE_RADIUS = 0.2

# Object5: Block cube (0.5m cubed)
BLOCK_SIZE = 0.5
BLOCK_HEIGHT = 0.5 # Same as BLOCK_SIZE for a cube

def main_obstacle_course_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for completing the obstacle course.
    This reward guides the robot through sequential phases:
    1. Jumping over the low wall.
    2. Pushing the large sphere into the high wall.
    3. Kicking the small sphere away.
    4. Jumping on top of the block.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (MANDATORY: direct access, ObjectN names)
    low_wall = env.scene['Object3']
    large_sphere = env.scene['Object1']
    high_wall = env.scene['Object4']
    small_sphere = env.scene['Object2']
    block = env.scene['Object5']

    # Access required robot parts (MANDATORY: using body_names.index)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # MANDATORY: relative positions
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Calculate object positions (MANDATORY: root_pos_w)
    low_wall_pos = low_wall.data.root_pos_w
    large_sphere_pos = large_sphere.data.root_pos_w
    high_wall_pos = high_wall.data.root_pos_w
    small_sphere_pos = small_sphere.data.root_pos_w
    block_pos = block.data.root_pos_w

    total_reward = torch.zeros_like(pelvis_pos_x)

    # --- Phase 1: Walk to low wall and jump over it ---
    # Target x-position for jumping over low wall: center of low wall + half its depth
    low_wall_jump_x_target = low_wall_pos[:, 0] + LOW_WALL_DEPTH_X / 2.0
    # Target x-position after clearing low wall: center of low wall + half its depth + small clearance
    low_wall_clear_x_target = low_wall_pos[:, 0] + LOW_WALL_DEPTH_X / 2.0 + 0.5

    # Reward for approaching low wall (MANDATORY: relative distance, continuous)
    # Only consider x-distance for approach, y-distance can be lenient (wall width)
    reward_approach_low_wall = -torch.abs(pelvis_pos_x - low_wall_jump_x_target)
    
    # Reward for jumping over low wall (pelvis height above wall + clearance)
    # This reward is active when pelvis is near the wall's x-center
    pelvis_over_low_wall_z_target = low_wall_pos[:, 2] + LOW_WALL_HEIGHT + 0.3 # 0.3m clearance
    # Only reward height when robot is within the x-range of the wall
    is_over_low_wall_x = (pelvis_pos_x > low_wall_pos[:, 0] - LOW_WALL_DEPTH_X / 2.0) & \
                         (pelvis_pos_x < low_wall_pos[:, 0] + LOW_WALL_DEPTH_X / 2.0)
    reward_jump_low_wall_height = torch.where(is_over_low_wall_x, -torch.abs(pelvis_pos_z - pelvis_over_low_wall_z_target), 0.0)
    
    # Reward for clearing low wall (pelvis x past the wall)
    reward_clear_low_wall = (pelvis_pos_x - low_wall_clear_x_target) * 2.0 # Positive reward for moving past

    # Phase 1 activation: active until robot clears the low wall
    phase1_active = pelvis_pos_x < low_wall_clear_x_target
    total_reward += torch.where(phase1_active, reward_approach_low_wall * 0.5, 0.0)
    total_reward += torch.where(phase1_active, reward_jump_low_wall_height * 0.5, 0.0)
    total_reward += torch.where(phase1_active, reward_clear_low_wall * 1.0, 0.0)

    # --- Phase 2: Push large sphere into high wall ---
    # Target x-position for pushing large sphere: center of large sphere - its radius
    large_sphere_push_x_target = large_sphere_pos[:, 0] - LARGE_SPHERE_RADIUS
    # Target x-position after high wall falls: center of high wall + half its depth + small clearance
    high_wall_clear_x_target = high_wall_pos[:, 0] + HIGH_WALL_DEPTH_X / 2.0 + 0.5

    # Reward for approaching large sphere (MANDATORY: relative distance, continuous)
    reward_approach_large_sphere = -torch.abs(pelvis_pos_x - large_sphere_push_x_target)
    
    # Reward for pushing large sphere (distance between sphere and high wall)
    # This reward is active when robot is past low wall and approaching large sphere
    sphere_to_wall_dist_x = large_sphere_pos[:, 0] - high_wall_pos[:, 0]
    reward_push_sphere = -torch.abs(sphere_to_wall_dist_x) * 5.0 # Reward for reducing distance
    
    # Reward for high wall falling (its z-position)
    # High wall's original z-center is high_wall_pos[:, 2]. When it falls, its z-center will decrease.
    # A threshold for "fallen" could be its z-position being significantly lower than its original height.
    # Assuming original z-pos is roughly HIGH_WALL_HEIGHT / 2.0 if placed on ground.
    # Let's use a relative threshold: 80% of its height below its initial z-position.
    high_wall_fallen_z_threshold = high_wall_pos[:, 2] - HIGH_WALL_HEIGHT * 0.8
    reward_high_wall_fallen = (high_wall_pos[:, 2] < high_wall_fallen_z_threshold).float() * 10.0 # Binary for significant fall

    # Phase 2 activation: active after clearing low wall and before clearing high wall
    phase2_active = (pelvis_pos_x >= low_wall_clear_x_target) & (pelvis_pos_x < high_wall_clear_x_target)
    total_reward += torch.where(phase2_active, reward_approach_large_sphere * 0.5, 0.0)
    total_reward += torch.where(phase2_active & (pelvis_pos_x > large_sphere_pos[:, 0] - LARGE_SPHERE_RADIUS - 0.5), reward_push_sphere * 1.0, 0.0)
    total_reward += torch.where(phase2_active, reward_high_wall_fallen * 1.0, 0.0)

    # --- Phase 3: Kick small sphere away from wall ---
    # Target x-position for kicking small sphere: center of small sphere - its radius
    small_sphere_kick_x_target = small_sphere_pos[:, 0] - SMALL_SPHERE_RADIUS
    # Target x-position after kicking small sphere: center of small sphere + its radius + clearance
    small_sphere_clear_x_target = small_sphere_pos[:, 0] + SMALL_SPHERE_RADIUS + 0.5

    # Reward for approaching small sphere (MANDATORY: relative distance, continuous)
    reward_approach_small_sphere = -torch.abs(pelvis_pos_x - small_sphere_kick_x_target)
    
    # Reward for kicking small sphere away from high wall's original position
    # This reward is active when robot is past high wall and approaching small sphere
    # We want the small sphere to move further in the positive x direction.
    # Use the initial high wall x-position as a reference point for "away".
    # Assuming high_wall_pos[:, 0] is the initial x-position of the high wall.
    small_sphere_away_dist_x = small_sphere_pos[:, 0] - high_wall_pos[:, 0]
    reward_kick_sphere_away = small_sphere_away_dist_x * 2.0 # Positive reward for increasing distance

    # Phase 3 activation: active after clearing high wall and before clearing small sphere
    phase3_active = (pelvis_pos_x >= high_wall_clear_x_target) & (pelvis_pos_x < small_sphere_clear_x_target)
    total_reward += torch.where(phase3_active, reward_approach_small_sphere * 0.5, 0.0)
    total_reward += torch.where(phase3_active & (pelvis_pos_x > small_sphere_pos[:, 0] - SMALL_SPHERE_RADIUS - 0.5), reward_kick_sphere_away * 1.0, 0.0)

    # --- Phase 4: Jump on top of block ---
    # Target x-position for jumping on block: center of block
    block_jump_x_target = block_pos[:, 0]
    # Target z-position for standing on block: block top surface + pelvis default height
    block_top_z = block_pos[:, 2] + BLOCK_HEIGHT / 2.0
    pelvis_on_block_z_target = block_top_z + 0.7 # 0.7m is default pelvis height

    # Reward for approaching block (MANDATORY: relative distance, continuous)
    reward_approach_block = -torch.abs(pelvis_pos_x - block_jump_x_target)
    
    # Reward for feet on top of block (MANDATORY: relative distance, continuous-like via conditions)
    # Check if feet are above block's top surface (z) and within block's x, y bounds.
    # Use a small tolerance for z to allow for slight variations.
    feet_on_block_condition = (left_foot_pos[:, 2] > block_top_z - 0.05) & (right_foot_pos[:, 2] > block_top_z - 0.05) & \
                              (torch.abs(left_foot_pos[:, 0] - block_pos[:, 0]) < BLOCK_SIZE / 2.0 + 0.1) & \
                              (torch.abs(right_foot_pos[:, 0] - block_pos[:, 0]) < BLOCK_SIZE / 2.0 + 0.1) & \
                              (torch.abs(left_foot_pos[:, 1] - block_pos[:, 1]) < BLOCK_SIZE / 2.0 + 0.1) & \
                              (torch.abs(right_foot_pos[:, 1] - block_pos[:, 1]) < BLOCK_SIZE / 2.0 + 0.1)
    reward_feet_on_block = feet_on_block_condition.float() * 5.0 # Binary for feet on block, but activated by continuous conditions
    
    # Reward for pelvis height on block (MANDATORY: relative distance, continuous)
    reward_pelvis_on_block_height = -torch.abs(pelvis_pos_z - pelvis_on_block_z_target) * 2.0

    # Phase 4 activation: active after clearing small sphere
    phase4_active = (pelvis_pos_x >= small_sphere_clear_x_target)
    total_reward += torch.where(phase4_active, reward_approach_block * 0.5, 0.0)
    total_reward += torch.where(phase4_active, reward_feet_on_block * 1.0, 0.0)
    total_reward += torch.where(phase4_active & feet_on_block_condition, reward_pelvis_on_block_height * 1.0, 0.0)

    reward = total_reward

    # MANDATORY: Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Penalizes the robot for colliding with obstacles it should not be touching or for touching obstacles at the wrong time.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (MANDATORY: direct access, ObjectN names)
    low_wall = env.scene['Object3']
    high_wall = env.scene['Object4']
    block = env.scene['Object5']

    # Access required robot parts (MANDATORY: using body_names.index)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # MANDATORY: relative positions
    pelvis_pos_x = pelvis_pos[:, 0]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Define collision thresholds (MANDATORY: relative distances, continuous)
    collision_threshold = 0.1 # Small clearance for collision detection

    # --- Penalize collision with low wall when not jumping over it ---
    low_wall_pos = low_wall.data.root_pos_w
    low_wall_x_min = low_wall_pos[:, 0] - LOW_WALL_DEPTH_X / 2.0
    low_wall_x_max = low_wall_pos[:, 0] + LOW_WALL_DEPTH_X / 2.0
    low_wall_y_min = low_wall_pos[:, 1] - LOW_WALL_WIDTH_Y / 2.0
    low_wall_y_max = low_wall_pos[:, 1] + LOW_WALL_WIDTH_Y / 2.0
    low_wall_z_max = low_wall_pos[:, 2] + LOW_WALL_HEIGHT / 2.0

    # Check if pelvis is within low wall's bounding box with a small buffer
    pelvis_in_low_wall_x = (pelvis_pos_x > low_wall_x_min - collision_threshold) & (pelvis_pos_x < low_wall_x_max + collision_threshold)
    pelvis_in_low_wall_y = (pelvis_pos[:, 1] > low_wall_y_min - collision_threshold) & (pelvis_pos[:, 1] < low_wall_y_max + collision_threshold)
    pelvis_in_low_wall_z = (pelvis_pos[:, 2] < low_wall_z_max + collision_threshold)

    collision_low_wall = pelvis_in_low_wall_x & pelvis_in_low_wall_y & pelvis_in_low_wall_z
    # Only penalize if not actively jumping (e.g., pelvis z is low relative to wall top)
    pelvis_low_z_for_wall = pelvis_pos[:, 2] < low_wall_pos[:, 2] + LOW_WALL_HEIGHT + 0.1 # 0.1m above wall top
    reward_low_wall_collision = torch.where(collision_low_wall & pelvis_low_z_for_wall, -5.0, 0.0)

    # --- Penalize collision with high wall when not pushing sphere or after it has fallen ---
    high_wall_pos = high_wall.data.root_pos_w
    high_wall_x_min = high_wall_pos[:, 0] - HIGH_WALL_DEPTH_X / 2.0
    high_wall_x_max = high_wall_pos[:, 0] + HIGH_WALL_DEPTH_X / 2.0
    high_wall_y_min = high_wall_pos[:, 1] - HIGH_WALL_WIDTH_Y / 2.0
    high_wall_y_max = high_wall_pos[:, 1] + HIGH_WALL_WIDTH_Y / 2.0
    high_wall_z_max = high_wall_pos[:, 2] + HIGH_WALL_HEIGHT / 2.0

    pelvis_in_high_wall_x = (pelvis_pos_x > high_wall_x_min - collision_threshold) & (pelvis_pos_x < high_wall_x_max + collision_threshold)
    pelvis_in_high_wall_y = (pelvis_pos[:, 1] > high_wall_y_min - collision_threshold) & (pelvis_pos[:, 1] < high_wall_y_max + collision_threshold)
    pelvis_in_high_wall_z = (pelvis_pos[:, 2] < high_wall_z_max + collision_threshold)

    collision_high_wall = pelvis_in_high_wall_x & pelvis_in_high_wall_y & pelvis_in_high_wall_z
    # Only penalize if high wall is still standing (z-pos is high)
    # The original high_wall_pos[:, 2] is the initial z-position.
    # If it's still standing, its current z-position should be close to its initial z-position.
    # If it has fallen, its z-position will be significantly lower.
    # A threshold for "standing" could be its z-position being above a certain fraction of its height.
    high_wall_standing = high_wall_pos[:, 2] > (high_wall.data.root_pos_w[:, 2] - HIGH_WALL_HEIGHT * 0.5) # More than half fallen
    reward_high_wall_collision = torch.where(collision_high_wall & high_wall_standing, -5.0, 0.0)

    # --- Penalize collision with block when not jumping on it ---
    block_pos = block.data.root_pos_w
    block_x_min = block_pos[:, 0] - BLOCK_SIZE / 2.0
    block_x_max = block_pos[:, 0] + BLOCK_SIZE / 2.0
    block_y_min = block_pos[:, 1] - BLOCK_SIZE / 2.0
    block_y_max = block_pos[:, 1] + BLOCK_SIZE / 2.0
    block_z_max = block_pos[:, 2] + BLOCK_HEIGHT / 2.0

    pelvis_in_block_x = (pelvis_pos_x > block_x_min - collision_threshold) & (pelvis_pos_x < block_x_max + collision_threshold)
    pelvis_in_block_y = (pelvis_pos[:, 1] > block_y_min - collision_threshold) & (pelvis_pos[:, 1] < block_y_max + collision_threshold)
    pelvis_in_block_z = (pelvis_pos[:, 2] < block_z_max + collision_threshold)

    collision_block = pelvis_in_block_x & pelvis_in_block_y & pelvis_in_block_z
    # Only penalize if not actively trying to jump on it (e.g., feet not above block)
    feet_above_block = (left_foot_pos[:, 2] > block_z_max + 0.1) | (right_foot_pos[:, 2] > block_z_max + 0.1)
    reward_block_collision = torch.where(collision_block & ~feet_above_block, -5.0, 0.0)

    reward = reward_low_wall_collision + reward_high_wall_collision + reward_block_collision

    # MANDATORY: Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stability_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain an upright and stable posture.
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required robot parts (MANDATORY: using body_names.index)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # MANDATORY: relative positions
    pelvis_pos_z = pelvis_pos[:, 2]

    # Target pelvis height for standing (MANDATORY: relative distance, continuous)
    target_pelvis_z = 0.7 # Default stable pelvis height for the robot
    reward_pelvis_height = -torch.abs(pelvis_pos_z - target_pelvis_z) * 0.5

    # Penalize high angular velocity to encourage stability (MANDATORY: relative distance, continuous)
    # Use correct attribute name 'body_ang_vel_w'
    pelvis_ang_vel = robot.data.body_ang_vel_w[:, pelvis_idx]
    reward_ang_vel = -torch.sum(torch.abs(pelvis_ang_vel), dim=-1) * 0.05 # Penalize high angular velocity

    reward = reward_pelvis_height + reward_ang_vel

    # MANDATORY: Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def sphere_interaction_guidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "sphere_interaction_reward") -> torch.Tensor:
    """
    Guides the robot to use appropriate body parts for interacting with the spheres (hands for pushing, feet for kicking).
    """
    # Get normalizer instance (MANDATORY)
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (MANDATORY: direct access, ObjectN names)
    large_sphere = env.scene['Object1']
    small_sphere = env.scene['Object2']
    low_wall = env.scene['Object3'] # Needed for phase progression
    high_wall = env.scene['Object4'] # Needed for phase progression

    # Access required robot parts (MANDATORY: using body_names.index)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # MANDATORY: relative positions
    pelvis_pos_x = pelvis_pos[:, 0]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Calculate object positions (MANDATORY: root_pos_w)
    low_wall_pos = low_wall.data.root_pos_w
    large_sphere_pos = large_sphere.data.root_pos_w
    high_wall_pos = high_wall.data.root_pos_w
    small_sphere_pos = small_sphere.data.root_pos_w

    total_reward = torch.zeros_like(pelvis_pos_x)

    # --- Phase 2: Push large sphere ---
    # Active when robot is past low wall and before high wall (using clearance targets from main reward)
    low_wall_clear_x_target = low_wall_pos[:, 0] + LOW_WALL_DEPTH_X / 2.0 + 0.5
    high_wall_clear_x_target = high_wall_pos[:, 0] + HIGH_WALL_DEPTH_X / 2.0 + 0.5
    phase2_active = (pelvis_pos_x >= low_wall_clear_x_target) & (pelvis_pos_x < high_wall_clear_x_target)

    # Reward for hands close to large sphere (MANDATORY: relative distance, continuous)
    dist_left_hand_large_sphere = torch.norm(left_hand_pos - large_sphere_pos, dim=-1)
    dist_right_hand_large_sphere = torch.norm(right_hand_pos - large_sphere_pos, dim=-1)
    reward_hands_near_large_sphere = -torch.min(dist_left_hand_large_sphere, dist_right_hand_large_sphere) * 0.5
    reward_push_contact = torch.where(phase2_active, reward_hands_near_large_sphere, 0.0)
    total_reward += reward_push_contact

    # --- Phase 3: Kick small sphere ---
    # Active when robot is past high wall and before small sphere is cleared (using clearance targets from main reward)
    small_sphere_clear_x_target = small_sphere_pos[:, 0] + SMALL_SPHERE_RADIUS + 0.5
    # Fix: Typo in variable name 'pelvel_pos_x' should be 'pelvis_pos_x'
    phase3_active = (pelvis_pos_x >= high_wall_clear_x_target) & (pelvis_pos_x < small_sphere_clear_x_target)

    # Reward for feet close to small sphere (MANDATORY: relative distance, continuous)
    dist_left_foot_small_sphere = torch.norm(left_foot_pos - small_sphere_pos, dim=-1)
    dist_right_foot_small_sphere = torch.norm(right_foot_pos - small_sphere_pos, dim=-1)
    reward_feet_near_small_sphere = -torch.min(dist_left_foot_small_sphere, dist_right_foot_small_sphere) * 0.5
    reward_kick_contact = torch.where(phase3_active, reward_feet_near_small_sphere, 0.0)
    total_reward += reward_kick_contact

    reward = total_reward

    # MANDATORY: Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def proximity_to_object5_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "proximity_object5_reward") -> torch.Tensor:
    """
    Rewards moving closer to the center of Object5 (block) in the x-y plane.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required entities
    block = env.scene['Object5']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')

    # Positions
    block_pos = block.data.root_pos_w  # [N, 3]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]  # [N, 3]

    # 2D distance in x-y plane
    distance_xy = torch.norm(pelvis_pos[:, :2] - block_pos[:, :2], dim=-1)

    # Reward is negative distance (closer -> higher reward)
    reward = -distance_xy

    # Reward Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    """
    Rewards increasing pelvis height (z position).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot pelvis height
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    reward = pelvis_pos_z

    # Reward Normalization
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
    Reward terms for the obstacle_course_seed42 skill.
    """
    # Reward for moving closer to Object5 center (primary)
    BlockCenterProximityReward = RewTerm(
        func=proximity_to_object5_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "proximity_object5_reward"},
    )

    # Reward for increasing pelvis height
    PelvisHeightReward = RewTerm(
        func=pelvis_height_reward,
        weight=0.5,
        params={"normalise": True, "normaliser_name": "pelvis_height_reward"},
    )