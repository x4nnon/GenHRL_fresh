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

def obstacle_course_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "obstacle_course_main_reward") -> torch.Tensor:
    """
    Main reward function for the obstacle course skill.
    This reward guides the robot through sequential tasks: jumping over a low wall,
    pushing a large sphere into a high wall, kicking a small sphere away, and jumping onto a block.
    Rewards are phased based on task completion.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object data using approved patterns
    robot = env.scene["robot"]
    low_wall = env.scene['Object3'] # Low wall for robot to jump over
    large_sphere = env.scene['Object1'] # Large sphere for robot to push
    high_wall = env.scene['Object4'] # High wall for large sphere to push over
    small_sphere = env.scene['Object2'] # Small sphere for robot to kick
    block_cube = env.scene['Object5'] # Block cube for robot to jump on top of

    # Access robot parts using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object dimensions (hardcoded from description as per requirements)
    low_wall_x_dim = 0.3
    low_wall_z_dim = 0.5
    large_sphere_radius = 1.0
    high_wall_z_dim = 1.0
    block_cube_size = 0.5

    # Phase 1: Jump over Low Wall (Object3)
    # Robot should be past the low wall and before the large sphere
    low_wall_x_pos = low_wall.data.root_pos_w[:, 0]
    low_wall_z_pos = low_wall.data.root_pos_w[:, 2]
    large_sphere_x_pos = large_sphere.data.root_pos_w[:, 0]
    
    # Condition for being past the low wall (pelvis x-position beyond the wall's far edge)
    # Using a small buffer (0.1m) to ensure full clearance. Uses relative position.
    past_low_wall_condition = (pelvis_pos_x > low_wall_x_pos + low_wall_x_dim / 2.0 + 0.1)

    # Reward for clearing the low wall (pelvis height over wall)
    # Encourages pelvis to be above the wall's top surface during the jump. Uses relative position.
    # Active only when robot is approaching or directly over the wall.
    pelvis_over_wall_target_z = low_wall_z_pos + low_wall_z_dim + 0.2 # Target 0.2m above wall
    pelvis_over_wall_reward = -torch.abs(pelvis_pos_z - pelvis_over_wall_target_z)
    # Only active when pelvis is within a certain x-range around the wall. Uses relative position.
    pelvis_over_wall_active_x_range = (pelvis_pos_x > low_wall_x_pos - low_wall_x_dim / 2.0 - 0.5) & \
                                      (pelvis_pos_x < low_wall_x_pos + low_wall_x_dim / 2.0 + 0.5)
    pelvis_over_wall_reward = torch.where(pelvis_over_wall_active_x_range, pelvis_over_wall_reward, 0.0)

    # Reward for moving past the low wall (progress in x-direction)
    # Encourages robot to move towards the target x-position after clearing the wall. Uses relative position.
    # Target is slightly past the wall to ensure full clearance.
    progress_past_low_wall_target_x = low_wall_x_pos + low_wall_x_dim / 2.0 + 0.5
    progress_past_low_wall_reward = -torch.abs(pelvis_pos_x - progress_past_low_wall_target_x)
    # Active only until the robot is well past the low wall and before reaching the large sphere. Uses relative position.
    progress_past_low_wall_reward = torch.where(pelvis_pos_x < large_sphere_x_pos - large_sphere_radius, progress_past_low_wall_reward, 0.0)

    # Phase 2: Push Large Sphere (Object1) into High Wall (Object4)
    # Condition for large sphere having pushed over high wall
    # High wall's Z position significantly reduced (e.g., fallen below 0.5m). Uses relative position.
    high_wall_pushed_over_condition = (high_wall.data.root_pos_w[:, 2] < high_wall_z_dim * 0.5)

    # Reward for pushing large sphere towards high wall
    # Encourages the large sphere to move closer to the high wall. Uses relative distance.
    large_sphere_to_high_wall_dist = torch.norm(large_sphere.data.root_pos_w - high_wall.data.root_pos_w, dim=1)
    large_sphere_push_reward = -large_sphere_to_high_wall_dist
    # Active after low wall is cleared and before high wall is pushed over.
    large_sphere_push_reward = torch.where(past_low_wall_condition & ~high_wall_pushed_over_condition, large_sphere_push_reward, 0.0)

    # Phase 3: Kick Small Sphere (Object2) Away
    # Condition for small sphere being kicked away
    # Small sphere's x-position is significantly far from the high wall's original x-position. Uses relative position.
    small_sphere_kicked_away_condition = (torch.abs(small_sphere.data.root_pos_w[:, 0] - high_wall.data.root_pos_w[:, 0]) > 2.0)

    # Reward for kicking small sphere away from high wall
    # Rewards increasing the distance between the small sphere and the high wall's original x-position. Uses relative distance.
    small_sphere_dist_from_high_wall_x = torch.abs(small_sphere.data.root_pos_w[:, 0] - high_wall.data.root_pos_w[:, 0])
    small_sphere_kick_reward = small_sphere_dist_from_high_wall_x # Positive reward for increasing distance
    # Active after high wall is pushed over and before small sphere is kicked away.
    small_sphere_kick_reward = torch.where(high_wall_pushed_over_condition & ~small_sphere_kicked_away_condition, small_sphere_kick_reward, 0.0)

    # Phase 4: Jump on Top of Block (Object5)
    block_cube_x_pos = block_cube.data.root_pos_w[:, 0]
    block_cube_y_pos = block_cube.data.root_pos_w[:, 1]
    block_cube_z_pos = block_cube.data.root_pos_w[:, 2]

    # Condition for robot being on top of the block
    # Both feet must be above the block's top surface and within its x-y bounds. Uses relative positions.
    on_block_condition = (
        (left_foot_pos[:, 2] > block_cube_z_pos + block_cube_size - 0.1) & # Left foot above block top
        (right_foot_pos[:, 2] > block_cube_z_pos + block_cube_size - 0.1) & # Right foot above block top
        (torch.abs(left_foot_pos[:, 0] - block_cube_x_pos) < block_cube_size / 2.0 - 0.05) & # Left foot within block x-bounds
        (torch.abs(right_foot_pos[:, 0] - block_cube_x_pos) < block_cube_size / 2.0 - 0.05) & # Right foot within block x-bounds
        (torch.abs(left_foot_pos[:, 1] - block_cube_y_pos) < block_cube_size / 2.0 - 0.05) & # Left foot within block y-bounds
        (torch.abs(right_foot_pos[:, 1] - block_cube_y_pos) < block_cube_size / 2.0 - 0.05) # Right foot within block y-bounds
    )

    # Reward for approaching and jumping on block
    # Encourages pelvis to reach a target position (x,y,z) relative to the block's top. Uses relative distances.
    pelvis_to_block_target_x = block_cube_x_pos
    pelvis_to_block_target_y = block_cube_y_pos
    pelvis_to_block_target_z = block_cube_z_pos + block_cube_size + 0.7 # Target pelvis height when standing on block
    
    pelvis_to_block_dist_x = pelvis_pos_x - pelvis_to_block_target_x
    pelvis_to_block_dist_y = pelvis_pos[:, 1] - pelvis_to_block_target_y
    pelvis_to_block_dist_z = pelvis_pos_z - pelvis_to_block_target_z
    
    approach_block_reward = -torch.sqrt(pelvis_to_block_dist_x**2 + pelvis_to_block_dist_y**2 + pelvis_to_block_dist_z**2)
    # Active after small sphere is kicked away.
    approach_block_reward = torch.where(small_sphere_kicked_away_condition, approach_block_reward, 0.0)

    # Combine rewards based on phase progression
    # Initial reward for moving past the low wall and clearing it
    reward = progress_past_low_wall_reward + pelvis_over_wall_reward
    # Once past the low wall, add reward for pushing the large sphere
    reward = torch.where(past_low_wall_condition, reward + large_sphere_push_reward, reward)
    # Once high wall is pushed, add reward for kicking the small sphere
    reward = torch.where(high_wall_pushed_over_condition, reward + small_sphere_kick_reward, reward)
    # Once small sphere is kicked, add reward for approaching and jumping on the block
    reward = torch.where(small_sphere_kicked_away_condition, reward + approach_block_reward, reward)

    # Final stability reward on block
    # Large positive reward for successfully being on top of the block.
    final_stability_reward = torch.where(on_block_condition, 10.0, 0.0)
    reward = reward + final_stability_reward

    # Normalization implementation as per requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_and_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_stability_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance and maintaining stability.
    Penalizes collisions with obstacles and encourages maintaining a stable upright posture.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object data using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    low_wall = env.scene['Object3']
    high_wall = env.scene['Object4']
    block_cube = env.scene['Object5']

    # Object dimensions (hardcoded from description as per requirements)
    low_wall_x_dim = 0.3
    low_wall_y_dim = 5.0
    low_wall_z_dim = 0.5
    high_wall_x_dim = 0.3
    high_wall_y_dim = 5.0
    high_wall_z_dim = 1.0
    block_cube_size = 0.5

    # Pelvis stability reward (encourage standing upright)
    # Target pelvis height for standing is 0.7m. Uses absolute z-position for pelvis height.
    target_pelvis_z = 0.7
    pelvis_stability_reward = -torch.abs(pelvis_pos_z - target_pelvis_z)
    
    # Condition for not jumping (feet close to ground)
    # Feet are considered on ground if their Z position is below a small threshold (0.1m). Uses absolute z-position.
    feet_on_ground_condition = (left_foot_pos[:, 2] < 0.1) & (right_foot_pos[:, 2] < 0.1)
    
    # Condition for not being on the block yet (to avoid penalizing low pelvis when on block)
    # Check if pelvis is significantly below the block's top or far from its x-position. Uses relative positions.
    block_cube_x_pos = block_cube.data.root_pos_w[:, 0]
    block_cube_z_pos = block_cube.data.root_pos_w[:, 2]
    not_on_block_yet = (pelvis_pos_x < block_cube_x_pos - block_cube_size / 2.0 - 0.1) | \
                       (pelvis_pos_x > block_cube_x_pos + block_cube_size / 2.0 + 0.1) | \
                       (pelvis_pos_z < block_cube_z_pos + block_cube_size + 0.1)
    
    # Apply pelvis stability reward only when robot is standing on the ground and not yet on the final block.
    pelvis_stability_reward = torch.where(feet_on_ground_condition & not_on_block_yet, pelvis_stability_reward, 0.0)

    # Collision avoidance with low wall (Object3)
    # Penalize if pelvis is inside the wall volume (with a small buffer). Uses relative distances.
    low_wall_pos = low_wall.data.root_pos_w
    low_wall_collision_x = torch.abs(pelvis_pos[:, 0] - low_wall_pos[:, 0]) < (low_wall_x_dim / 2.0 - 0.05)
    low_wall_collision_y = torch.abs(pelvis_pos[:, 1] - low_wall_pos[:, 1]) < (low_wall_y_dim / 2.0 - 0.05)
    low_wall_collision_z = torch.abs(pelvis_pos[:, 2] - low_wall_pos[:, 2]) < (low_wall_z_dim / 2.0 - 0.05)
    low_wall_collision_penalty = torch.where(low_wall_collision_x & low_wall_collision_y & low_wall_collision_z, -5.0, 0.0)

    # Collision avoidance with high wall (Object4)
    # Penalize if pelvis is inside the wall volume (with a small buffer). Uses relative distances.
    high_wall_pos = high_wall.data.root_pos_w
    high_wall_collision_x = torch.abs(pelvis_pos[:, 0] - high_wall_pos[:, 0]) < (high_wall_x_dim / 2.0 - 0.05)
    high_wall_collision_y = torch.abs(pelvis_pos[:, 1] - high_wall_pos[:, 1]) < (high_wall_y_dim / 2.0 - 0.05)
    high_wall_collision_z = torch.abs(pelvis_pos[:, 2] - high_wall_pos[:, 2]) < (high_wall_z_dim / 2.0 - 0.05)
    high_wall_collision_penalty = torch.where(high_wall_collision_x & high_wall_collision_y & high_wall_collision_z, -5.0, 0.0)

    # Collision avoidance with block (Object5) - only before jumping on it
    # Penalize if pelvis is inside the block volume, but not if the robot is successfully on top of it. Uses relative distances.
    block_cube_pos = block_cube.data.root_pos_w
    block_collision_x = torch.abs(pelvis_pos[:, 0] - block_cube_pos[:, 0]) < (block_cube_size / 2.0 - 0.05)
    block_collision_y = torch.abs(pelvis_pos[:, 1] - block_cube_pos[:, 1]) < (block_cube_size / 2.0 - 0.05)
    block_collision_z = torch.abs(pelvis_pos[:, 2] - block_cube_pos[:, 2]) < (block_cube_size / 2.0 - 0.05)
    
    # Re-evaluate on_block_condition for this function's scope to ensure it's based on current state. Uses relative positions.
    on_block_condition = (
        (left_foot_pos[:, 2] > block_cube_pos[:, 2] + block_cube_size - 0.1) &
        (right_foot_pos[:, 2] > block_cube_pos[:, 2] + block_cube_size - 0.1) &
        (torch.abs(left_foot_pos[:, 0] - block_cube_pos[:, 0]) < block_cube_size / 2.0 - 0.05) &
        (torch.abs(right_foot_pos[:, 0] - block_cube_pos[:, 0]) < block_cube_size / 2.0 - 0.05) &
        (torch.abs(left_foot_pos[:, 1] - block_cube_pos[:, 1]) < block_cube_size / 2.0 - 0.05) &
        (torch.abs(right_foot_pos[:, 1] - block_cube_pos[:, 1]) < block_cube_size / 2.0 - 0.05)
    )
    block_collision_penalty = torch.where(block_collision_x & block_collision_y & block_collision_z & ~on_block_condition, -5.0, 0.0)

    reward = pelvis_stability_reward + low_wall_collision_penalty + high_wall_collision_penalty + block_collision_penalty

    # Normalization implementation as per requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def efficient_movement_and_interaction_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "efficient_movement_reward") -> torch.Tensor:
    """
    Shaping reward for efficient movement and appropriate interaction posture.
    Encourages straight movement along the x-axis and being close to objects for interaction.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object data using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    low_wall = env.scene['Object3']
    large_sphere = env.scene['Object1']
    high_wall = env.scene['Object4']
    small_sphere = env.scene['Object2']
    block_cube = env.scene['Object5']

    # Object dimensions (hardcoded from description as per requirements)
    low_wall_x_dim = 0.3
    large_sphere_radius = 1.0
    high_wall_z_dim = 1.0
    small_sphere_radius = 0.2
    block_cube_size = 0.5

    # Y-axis deviation penalty (encourage straight movement along y=0)
    # Penalizes deviation from the central y-axis. Uses relative distance to y=0.
    y_deviation_penalty = -torch.abs(pelvis_pos_y - 0.0)
    
    # Condition for robot being on top of the block (to disable y-deviation penalty)
    # Re-evaluates on_block_condition for this function's scope. Uses relative positions.
    block_cube_x_pos = block_cube.data.root_pos_w[:, 0]
    block_cube_y_pos = block_cube.data.root_pos_w[:, 1]
    block_cube_z_pos = block_cube.data.root_pos_w[:, 2]
    on_block_condition = (
        (left_foot_pos[:, 2] > block_cube_z_pos + block_cube_size - 0.1) &
        (right_foot_pos[:, 2] > block_cube_z_pos + block_cube_size - 0.1) &
        (torch.abs(left_foot_pos[:, 0] - block_cube_x_pos) < block_cube_size / 2.0 - 0.05) &
        (torch.abs(right_foot_pos[:, 0] - block_cube_x_pos) < block_cube_size / 2.0 - 0.05) &
        (torch.abs(left_foot_pos[:, 1] - block_cube_y_pos) < block_cube_size / 2.0 - 0.05) &
        (torch.abs(right_foot_pos[:, 1] - block_cube_y_pos) < block_cube_size / 2.0 - 0.05)
    )
    y_deviation_penalty = torch.where(~on_block_condition, y_deviation_penalty, 0.0)

    # Phase progression conditions (re-evaluated for this function's scope)
    low_wall_x_pos = low_wall.data.root_pos_w[:, 0]
    past_low_wall_condition = (pelvis_pos_x > low_wall_x_pos + low_wall_x_dim / 2.0 + 0.1)
    high_wall_pushed_over_condition = (high_wall.data.root_pos_w[:, 2] < high_wall_z_dim * 0.5)
    small_sphere_kicked_away_condition = (torch.abs(small_sphere.data.root_pos_w[:, 0] - high_wall.data.root_pos_w[:, 0]) > 2.0)

    # Reward for being close to large sphere for pushing
    # Encourages pelvis to approach the large sphere. Uses relative 2D distance.
    large_sphere_approach_dist = torch.norm(large_sphere.data.root_pos_w[:, :2] - pelvis_pos[:, :2], dim=1)
    large_sphere_approach_reward = -large_sphere_approach_dist
    # Active after passing low wall and before high wall is pushed over.
    large_sphere_approach_reward = torch.where(past_low_wall_condition & ~high_wall_pushed_over_condition, large_sphere_approach_reward, 0.0)

    # Reward for being close to small sphere for kicking
    # Encourages pelvis to approach the small sphere. Uses relative 2D distance.
    small_sphere_approach_dist = torch.norm(small_sphere.data.root_pos_w[:, :2] - pelvis_pos[:, :2], dim=1)
    small_sphere_approach_reward = -small_sphere_approach_dist
    # Active after high wall is pushed over and before small sphere is kicked away.
    small_sphere_approach_reward = torch.where(high_wall_pushed_over_condition & ~small_sphere_kicked_away_condition, small_sphere_approach_reward, 0.0)

    # Reward for hands being close to large sphere for pushing
    # Encourages hands to be near the large sphere for interaction. Uses relative 3D distance.
    hand_to_large_sphere_dist_left = torch.norm(left_hand_pos - large_sphere.data.root_pos_w, dim=1)
    hand_to_large_sphere_dist_right = torch.norm(right_hand_pos - large_sphere.data.root_pos_w, dim=1)
    hand_push_reward = -torch.min(hand_to_large_sphere_dist_left, hand_to_large_sphere_dist_right)
    hand_push_reward = torch.where(past_low_wall_condition & ~high_wall_pushed_over_condition, hand_push_reward, 0.0)

    # Reward for foot being close to small sphere for kicking
    # Encourages feet to be near the small sphere for interaction. Uses relative 3D distance.
    foot_to_small_sphere_dist_left = torch.norm(left_foot_pos - small_sphere.data.root_pos_w, dim=1)
    foot_to_small_sphere_dist_right = torch.norm(right_foot_pos - small_sphere.data.root_pos_w, dim=1)
    foot_kick_reward = -torch.min(foot_to_small_sphere_dist_left, foot_to_small_sphere_dist_right)
    foot_kick_reward = torch.where(high_wall_pushed_over_condition & ~small_sphere_kicked_away_condition, foot_kick_reward, 0.0)

    reward = y_deviation_penalty + large_sphere_approach_reward + small_sphere_approach_reward + hand_push_reward + foot_kick_reward

    # Normalization implementation as per requirements
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
    Reward terms for the obstacle_course_seed456 skill.
    Defines the main task reward and two shaping rewards for stability/collision and efficient movement.
    """
    # Main reward for progressing through the obstacle course tasks with weight 1.0
    ObstacleCourseMainReward = RewTerm(func=obstacle_course_main_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "obstacle_course_main_reward"})

    # Shaping reward for collision avoidance and maintaining upright posture with weight 0.6
    CollisionAvoidanceAndStabilityReward = RewTerm(func=collision_avoidance_and_stability_reward, weight=0.6,
                                                   params={"normalise": True, "normaliser_name": "collision_stability_reward"})

    # Shaping reward for efficient movement and appropriate interaction postures with weight 0.3
    EfficientMovementAndInteractionReward = RewTerm(func=efficient_movement_and_interaction_reward, weight=0.3,
                                                    params={"normalise": True, "normaliser_name": "efficient_movement_reward"})