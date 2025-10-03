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

# Hardcoded object dimensions from the object configuration
# Object1: large sphere for robot to push
LARGE_SPHERE_RADIUS = 1.0
# Object2: small sphere for robot to kick
SMALL_SPHERE_RADIUS = 0.2
# Object3: low wall for robot to jump over
LOW_WALL_X_DIM = 0.3
LOW_WALL_Y_DIM = 5.0
LOW_WALL_Z_DIM = 0.5
# Object4: high wall for large sphere to push over
HIGH_WALL_X_DIM = 0.3
HIGH_WALL_Y_DIM = 5.0
HIGH_WALL_Z_DIM = 1.0
# Object5: block cube for robot to jump on top of
BLOCK_X_DIM = 0.5
BLOCK_Y_DIM = 0.5
BLOCK_Z_DIM = 0.5

def obstacle_course_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_reward") -> torch.Tensor:
    """
    Primary reward for completing the obstacle course.
    Combines rewards for jumping over the low wall, pushing the large sphere, kicking the small sphere,
    and jumping on the block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    # Access objects using approved patterns
    object3 = env.scene['Object3'] # Low wall
    object1 = env.scene['Object1'] # Large sphere
    object4 = env.scene['Object4'] # High wall
    object2 = env.scene['Object2'] # Small sphere
    object5 = env.scene['Object5'] # Block cube

    # Initialize phase rewards for batched environments
    reward_phase1 = torch.zeros_like(pelvis_pos_x)
    reward_phase2 = torch.zeros_like(pelvis_pos_x)
    reward_phase3 = torch.zeros_like(pelvis_pos_x)
    reward_phase4 = torch.zeros_like(pelvis_pos_x)

    # --- Phase 1: Jump over Low Wall (Object3) ---
    # Reward for approaching low wall (pelvis X-position relative to wall center)
    # Uses relative distance: object3.data.root_pos_w[:, 0] - pelvis_pos_x
    dist_pelvis_to_low_wall_x = object3.data.root_pos_w[:, 0] - pelvis_pos_x
    # Continuous reward: closer to wall, higher reward (negative absolute distance, so closer to 0 is better)
    reward_approach_low_wall = -torch.abs(dist_pelvis_to_low_wall_x)

    # Reward for clearing low wall (pelvis and feet height above wall top)
    # Wall top Z is relative to wall's root position, using hardcoded dimension
    wall_top_z = object3.data.root_pos_w[:, 2] + LOW_WALL_Z_DIM / 2.0
    pelvis_clearance_z = pelvis_pos_z - wall_top_z
    left_foot_clearance_z = left_foot_pos_z - wall_top_z
    right_foot_clearance_z = right_foot_pos_z - wall_top_z

    # Condition for being "over" the wall (pelvis X-position within wall bounds + buffer)
    # This ensures the clearance reward is only active when the robot is attempting to jump over.
    # Uses relative distances for X-bounds.
    is_over_low_wall_x = (pelvis_pos_x > object3.data.root_pos_w[:, 0] - LOW_WALL_X_DIM / 2.0 - 0.2) & \
                         (pelvis_pos_x < object3.data.root_pos_w[:, 0] + LOW_WALL_X_DIM / 2.0 + 0.2)

    # Smooth reward for pelvis and feet being above wall height
    # Sigmoid function to provide continuous reward based on clearance, with a small positive buffer (0.1m)
    reward_clear_low_wall_pelvis = 1.0 / (1.0 + torch.exp(-10.0 * (pelvis_clearance_z - 0.1)))
    reward_clear_low_wall_feet = 1.0 / (1.0 + torch.exp(-10.0 * (torch.min(left_foot_clearance_z, right_foot_clearance_z) - 0.1)))

    # Reward for landing past low wall and moving towards large sphere
    # Target X for landing is the front edge of the large sphere minus a small buffer, using relative position.
    target_x_after_low_wall = object1.data.root_pos_w[:, 0] - LARGE_SPHERE_RADIUS - 0.5
    reward_land_past_low_wall = -torch.abs(pelvis_pos_x - target_x_after_low_wall)

    # Combine phase 1 rewards based on robot's X-position relative to the low wall
    # Uses torch.where for conditional rewards, ensuring continuity.
    reward_phase1 = torch.where(pelvis_pos_x < object3.data.root_pos_w[:, 0] - LOW_WALL_X_DIM / 2.0,
                                reward_approach_low_wall,
                                torch.where(is_over_low_wall_x,
                                            reward_clear_low_wall_pelvis + reward_clear_low_wall_feet,
                                            reward_land_past_low_wall))

    # --- Phase 2: Push Large Sphere (Object1) into High Wall (Object4) ---
    # Condition for activating phase 2: Pelvis is past low wall and before high wall's original position
    # Uses relative distances to define the active zone for this phase.
    phase2_active_condition = (pelvis_pos_x > object3.data.root_pos_w[:, 0] + LOW_WALL_X_DIM / 2.0 + 0.5) & \
                              (pelvis_pos_x < object4.data.root_pos_w[:, 0] + HIGH_WALL_X_DIM / 2.0 + 0.5)

    # Reward for approaching large sphere (pelvis X-position relative to sphere center)
    dist_pelvis_to_large_sphere_x = object1.data.root_pos_w[:, 0] - pelvis_pos_x
    reward_approach_large_sphere = -torch.abs(dist_pelvis_to_large_sphere_x)

    # Reward for pushing large sphere towards high wall (sphere X-position relative to high wall X-position)
    # The goal is to reduce the distance between the sphere and the wall.
    dist_sphere_to_high_wall_x = object4.data.root_pos_w[:, 0] - object1.data.root_pos_w[:, 0]
    reward_push_sphere = -torch.abs(dist_sphere_to_high_wall_x)

    # Reward for high wall falling (check its Z position relative to its original height)
    # High wall's original Z center is object4.data.root_pos_w[:, 2]
    # A threshold is set slightly above ground to detect a fall, using hardcoded dimension.
    high_wall_fallen_threshold_z = object4.data.root_pos_w[:, 2] - HIGH_WALL_Z_DIM / 2.0 + 0.1 # 0.1m above ground
    # Binary reward for simplicity, as it's a clear task completion step.
    reward_high_wall_fallen = torch.where(object4.data.root_pos_w[:, 2] < high_wall_fallen_threshold_z, 10.0, 0.0)

    # Combine phase 2 rewards, active only when in the phase's designated area
    reward_phase2 = torch.where(phase2_active_condition, reward_approach_large_sphere, 0.0)
    # Reward for pushing sphere is active once the sphere is past the low wall, using relative position.
    reward_phase2 += torch.where(object1.data.root_pos_w[:, 0] > object3.data.root_pos_w[:, 0] + LOW_WALL_X_DIM / 2.0 + 0.5,
                                 reward_push_sphere, 0.0)
    reward_phase2 += reward_high_wall_fallen

    # --- Phase 3: Kick Small Sphere (Object2) ---
    # Condition for activating phase 3: High wall has fallen AND pelvis is past high wall's original position
    # AND pelvis is before the block's original position. Uses relative positions.
    phase3_active_condition = (object4.data.root_pos_w[:, 2] < high_wall_fallen_threshold_z) & \
                              (pelvis_pos_x > object4.data.root_pos_w[:, 0] + HIGH_WALL_X_DIM / 2.0 + 0.5) & \
                              (pelvis_pos_x < object5.data.root_pos_w[:, 0] - BLOCK_X_DIM / 2.0 - 0.5)

    # Reward for approaching small sphere (pelvis X-position relative to sphere center)
    dist_pelvis_to_small_sphere_x = object2.data.root_pos_w[:, 0] - pelvis_pos_x
    reward_approach_small_sphere = -torch.abs(dist_pelvis_to_small_sphere_x)

    # Reward for kicking small sphere away (increase its X position)
    # Reward for increasing the distance between the small sphere and the high wall.
    # Uses hardcoded dimensions for initial separation reference.
    dist_small_sphere_from_high_wall = object2.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    # Reward for increasing this distance, but only if the high wall has fallen.
    reward_kick_small_sphere = torch.where(object4.data.root_pos_w[:, 2] < high_wall_fallen_threshold_z,
                                           5.0 * (dist_small_sphere_from_high_wall - (HIGH_WALL_X_DIM/2.0 + SMALL_SPHERE_RADIUS + 3.0)), # Reward for moving beyond initial separation
                                           0.0)
    reward_kick_small_sphere = torch.clamp(reward_kick_small_sphere, min=0.0, max=5.0) # Clamp to prevent excessive reward

    # Combine phase 3 rewards
    reward_phase3 = torch.where(phase3_active_condition, reward_approach_small_sphere, 0.0)
    reward_phase3 += reward_kick_small_sphere

    # --- Phase 4: Jump on Block (Object5) ---
    # Condition for activating phase 4: Small sphere has been kicked AND pelvis is past small sphere's original position
    # Uses relative positions and hardcoded dimensions for thresholds.
    phase4_active_condition = (object2.data.root_pos_w[:, 0] > object4.data.root_pos_w[:, 0] + HIGH_WALL_X_DIM/2.0 + SMALL_SPHERE_RADIUS + 3.0 + 0.5) & \
                              (pelvis_pos_x > object2.data.root_pos_w[:, 0] + SMALL_SPHERE_RADIUS + 0.5)

    # Reward for approaching block (pelvis X-position relative to block center)
    dist_pelvis_to_block_x = object5.data.root_pos_w[:, 0] - pelvis_pos_x
    reward_approach_block = -torch.abs(dist_pelvis_to_block_x)

    # Reward for feet on top of block
    block_top_z = object5.data.root_pos_w[:, 2] + BLOCK_Z_DIM / 2.0
    # Check if feet are within block's X and Y bounds and close to its top Z, using relative positions and hardcoded dimensions.
    block_min_x = object5.data.root_pos_w[:, 0] - BLOCK_X_DIM / 2.0
    block_max_x = object5.data.root_pos_w[:, 0] + BLOCK_X_DIM / 2.0
    block_min_y = object5.data.root_pos_w[:, 1] - BLOCK_Y_DIM / 2.0
    block_max_y = object5.data.root_pos_w[:, 1] + BLOCK_Y_DIM / 2.0

    # Check left foot on block
    left_foot_on_block_x = (left_foot_pos[:, 0] > block_min_x) & (left_foot_pos[:, 0] < block_max_x)
    left_foot_on_block_y = (left_foot_pos[:, 1] > block_min_y) & (left_foot_pos[:, 1] < block_max_y)
    left_foot_on_block_z_close = torch.abs(left_foot_pos_z - block_top_z) < 0.1 # Within 0.1m of block top
    left_foot_on_block = left_foot_on_block_x & left_foot_on_block_y & left_foot_on_block_z_close

    # Check right foot on block
    right_foot_on_block_x = (right_foot_pos[:, 0] > block_min_x) & (right_foot_pos[:, 0] < block_max_x)
    right_foot_on_block_y = (right_foot_pos[:, 1] > block_min_y) & (right_foot_pos[:, 1] < block_max_y)
    right_foot_on_block_z_close = torch.abs(right_foot_pos_z - block_top_z) < 0.1
    right_foot_on_block = right_foot_on_block_x & right_foot_on_block_y & right_foot_on_block_z_close

    feet_on_block_condition = left_foot_on_block | right_foot_on_block

    # Reward for feet being on the block, higher when closer to the top surface
    # Uses relative Z distance for continuous reward.
    reward_feet_on_block = torch.where(feet_on_block_condition,
                                       5.0 - (torch.abs(left_foot_pos_z - block_top_z) + torch.abs(right_foot_pos_z - block_top_z)),
                                       0.0)
    reward_feet_on_block = torch.clamp(reward_feet_on_block, min=0.0) # Ensure non-negative

    # Reward for pelvis stability on block (target pelvis_z = block_top_z + 0.7, assuming robot stands ~0.7m tall)
    # Uses relative Z distance for continuous reward.
    target_pelvis_z_on_block = block_top_z + 0.7
    reward_pelvis_stable_on_block = -torch.abs(pelvis_pos_z - target_pelvis_z_on_block)

    # Combine phase 4 rewards
    reward_phase4 = torch.where(phase4_active_condition, reward_approach_block, 0.0)
    reward_phase4 += torch.where(feet_on_block_condition, reward_feet_on_block + reward_pelvis_stable_on_block, 0.0)

    # Final primary reward combines all phases
    # Summing rewards from different phases, assuming sequential progression.
    primary_reward = reward_phase1 + reward_phase2 + reward_phase3 + reward_phase4

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Penalizes collisions between the robot's body parts and the obstacles,
    except for intentional interactions (pushing Object1, kicking Object2, jumping on Object5).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Access objects using approved patterns
    object3 = env.scene['Object3'] # Low wall
    object1 = env.scene['Object1'] # Large sphere
    object4 = env.scene['Object4'] # High wall
    object2 = env.scene['Object2'] # Small sphere
    object5 = env.scene['Object5'] # Block cube

    collision_penalty = torch.zeros_like(pelvis_pos[:, 0])

    # Helper function for box collision check (relative distances)
    # Uses relative position of robot part to object center and hardcoded dimensions.
    def is_colliding_box(robot_part_pos, obj_pos, obj_x_dim, obj_y_dim, obj_z_dim, buffer=0.1):
        rel_pos = robot_part_pos - obj_pos
        
        colliding_x = (rel_pos[:, 0] > -(obj_x_dim / 2.0 + buffer)) & (rel_pos[:, 0] < (obj_x_dim / 2.0 + buffer))
        colliding_y = (rel_pos[:, 1] > -(obj_y_dim / 2.0 + buffer)) & (rel_pos[:, 1] < (obj_y_dim / 2.0 + buffer))
        colliding_z = (rel_pos[:, 2] > -(obj_z_dim / 2.0 + buffer)) & (rel_pos[:, 2] < (obj_z_dim / 2.0 + buffer))
        return colliding_x & colliding_y & colliding_z

    # Helper function for sphere collision check (relative distances)
    # Uses Euclidean distance between robot part and sphere center and hardcoded radius.
    def is_colliding_sphere(robot_part_pos, obj_pos, obj_radius, buffer=0.1):
        dist_sq = torch.sum((robot_part_pos - obj_pos)**2, dim=1)
        return dist_sq < (obj_radius + buffer)**2

    # Penalize collision with low wall (Object3) - except when jumping over
    # Condition for jumping over: pelvis is roughly over the wall and high enough. Uses relative positions and hardcoded dimensions.
    low_wall_back_edge_x = object3.data.root_pos_w[:, 0] + LOW_WALL_X_DIM / 2.0
    is_jumping_over_low_wall = (pelvis_pos_x > object3.data.root_pos_w[:, 0] - LOW_WALL_X_DIM / 2.0 - 0.2) & \
                               (pelvis_pos_x < low_wall_back_edge_x + 0.2) & \
                               (pelvis_pos_z > object3.data.root_pos_w[:, 2] + LOW_WALL_Z_DIM / 2.0 - 0.1)

    if_colliding_low_wall = is_colliding_box(pelvis_pos, object3.data.root_pos_w, LOW_WALL_X_DIM, LOW_WALL_Y_DIM, LOW_WALL_Z_DIM) | \
                            is_colliding_box(left_hand_pos, object3.data.root_pos_w, LOW_WALL_X_DIM, LOW_WALL_Y_DIM, LOW_WALL_Z_DIM) | \
                            is_colliding_box(right_hand_pos, object3.data.root_pos_w, LOW_WALL_X_DIM, LOW_WALL_Y_DIM, LOW_WALL_Z_DIM) | \
                            is_colliding_box(left_foot_pos, object3.data.root_pos_w, LOW_WALL_X_DIM, LOW_WALL_Y_DIM, LOW_WALL_Z_DIM) | \
                            is_colliding_box(right_foot_pos, object3.data.root_pos_w, LOW_WALL_X_DIM, LOW_WALL_Y_DIM, LOW_WALL_Z_DIM)

    collision_penalty += torch.where(if_colliding_low_wall & ~is_jumping_over_low_wall, -5.0, 0.0)

    # Penalize collision with high wall (Object4) - except when it's already fallen
    # Condition: High wall is still standing (Z > threshold). Uses relative Z position and hardcoded dimension.
    high_wall_standing_threshold_z = object4.data.root_pos_w[:, 2] - HIGH_WALL_Z_DIM / 2.0 + 0.2 # Slightly above ground
    is_high_wall_standing = object4.data.root_pos_w[:, 2] > high_wall_standing_threshold_z

    if_colliding_high_wall = is_colliding_box(pelvis_pos, object4.data.root_pos_w, HIGH_WALL_X_DIM, HIGH_WALL_Y_DIM, HIGH_WALL_Z_DIM) | \
                             is_colliding_box(left_hand_pos, object4.data.root_pos_w, HIGH_WALL_X_DIM, HIGH_WALL_Y_DIM, HIGH_WALL_Z_DIM) | \
                             is_colliding_box(right_hand_pos, object4.data.root_pos_w, HIGH_WALL_X_DIM, HIGH_WALL_Y_DIM, HIGH_WALL_Z_DIM) | \
                             is_colliding_box(left_foot_pos, object4.data.root_pos_w, HIGH_WALL_X_DIM, HIGH_WALL_Y_DIM, HIGH_WALL_Z_DIM) | \
                             is_colliding_box(right_foot_pos, object4.data.root_pos_w, HIGH_WALL_X_DIM, HIGH_WALL_Y_DIM, HIGH_WALL_Z_DIM)

    collision_penalty += torch.where(if_colliding_high_wall & is_high_wall_standing, -5.0, 0.0)

    # Penalize collision with block (Object5) - except when jumping on it
    # Condition for jumping on block: pelvis is roughly over the block and high enough. Uses relative positions and hardcoded dimensions.
    block_front_edge_x = object5.data.root_pos_w[:, 0] - BLOCK_X_DIM / 2.0
    is_jumping_on_block = (pelvis_pos_x > block_front_edge_x - 0.2) & \
                          (pelvis_pos_x < object5.data.root_pos_w[:, 0] + BLOCK_X_DIM / 2.0 + 0.2) & \
                          (pelvis_pos_z > object5.data.root_pos_w[:, 2] + BLOCK_Z_DIM / 2.0 - 0.1)

    if_colliding_block = is_colliding_box(pelvis_pos, object5.data.root_pos_w, BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM) | \
                         is_colliding_box(left_hand_pos, object5.data.root_pos_w, BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM) | \
                         is_colliding_box(right_hand_pos, object5.data.root_pos_w, BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM) | \
                         is_colliding_box(left_foot_pos, object5.data.root_pos_w, BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM) | \
                         is_colliding_box(right_foot_pos, object5.data.root_pos_w, BLOCK_X_DIM, BLOCK_Y_DIM, BLOCK_Z_DIM)

    collision_penalty += torch.where(if_colliding_block & ~is_jumping_on_block, -5.0, 0.0)

    # Penalize collision with large sphere (Object1) - except when pushing it
    # Condition for pushing: pelvis is roughly aligned with the sphere in X and close. Uses relative positions and hardcoded radius.
    is_pushing_large_sphere = (pelvis_pos_x > object1.data.root_pos_w[:, 0] - LARGE_SPHERE_RADIUS - 0.5) & \
                              (pelvis_pos_x < object1.data.root_pos_w[:, 0] + LARGE_SPHERE_RADIUS + 0.5)

    if_colliding_large_sphere = is_colliding_sphere(pelvis_pos, object1.data.root_pos_w, LARGE_SPHERE_RADIUS) | \
                                is_colliding_sphere(left_hand_pos, object1.data.root_pos_w, LARGE_SPHERE_RADIUS) | \
                                is_colliding_sphere(right_hand_pos, object1.data.root_pos_w, LARGE_SPHERE_RADIUS) | \
                                is_colliding_sphere(left_foot_pos, object1.data.root_pos_w, LARGE_SPHERE_RADIUS) | \
                                is_colliding_sphere(right_foot_pos, object1.data.root_pos_w, LARGE_SPHERE_RADIUS)

    collision_penalty += torch.where(if_colliding_large_sphere & ~is_pushing_large_sphere, -5.0, 0.0)

    # Penalize collision with small sphere (Object2) - except when kicking it
    # Condition for kicking: pelvis is roughly aligned with the sphere in X and close. Uses relative positions and hardcoded radius.
    is_kicking_small_sphere = (pelvis_pos_x > object2.data.root_pos_w[:, 0] - SMALL_SPHERE_RADIUS - 0.5) & \
                              (pelvis_pos_x < object2.data.root_pos_w[:, 0] + SMALL_SPHERE_RADIUS + 0.5)

    if_colliding_small_sphere = is_colliding_sphere(pelvis_pos, object2.data.root_pos_w, SMALL_SPHERE_RADIUS) | \
                                is_colliding_sphere(left_hand_pos, object2.data.root_pos_w, SMALL_SPHERE_RADIUS) | \
                                is_colliding_sphere(right_hand_pos, object2.data.root_pos_w, SMALL_SPHERE_RADIUS) | \
                                is_colliding_sphere(left_foot_pos, object2.data.root_pos_w, SMALL_SPHERE_RADIUS) | \
                                is_colliding_sphere(right_foot_pos, object2.data.root_pos_w, SMALL_SPHERE_RADIUS)

    collision_penalty += torch.where(if_colliding_small_sphere & ~is_kicking_small_sphere, -5.0, 0.0)

    shaping_reward_1 = collision_penalty

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_1)
        return scaled_reward
    return shaping_reward_1

def pelvis_height_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_stability_reward") -> torch.Tensor:
    """
    Encourages the robot to maintain a stable pelvis height (around 0.7m) when not actively jumping or interacting with objects.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Access objects for context using approved patterns
    object3 = env.scene['Object3'] # Low wall
    object5 = env.scene['Object5'] # Block cube

    # Define target pelvis height for standing (hardcoded as a general target)
    target_pelvis_z = 0.7

    # Define conditions where this reward should NOT apply (i.e., when jumping or on block)
    # During jump over low wall: pelvis is over the wall's X range and significantly higher than normal standing height.
    # Uses relative positions and hardcoded dimensions.
    is_jumping_low_wall = (pelvis_pos_x > object3.data.root_pos_w[:, 0] - LOW_WALL_X_DIM / 2.0 - 0.5) & \
                          (pelvis_pos_x < object3.data.root_pos_w[:, 0] + LOW_WALL_X_DIM / 2.0 + 0.5) & \
                          (pelvis_pos_z > target_pelvis_z + 0.1) # Actively jumping high

    # When on top of the block (Object5): pelvis is roughly at the standing height on the block and within block X/Y bounds.
    # Uses relative positions and hardcoded dimensions.
    block_top_z = object5.data.root_pos_w[:, 2] + BLOCK_Z_DIM / 2.0
    is_on_block = (torch.abs(pelvis_pos_z - (block_top_z + 0.7)) < 0.15) & \
                  (pelvis_pos_x > object5.data.root_pos_w[:, 0] - BLOCK_X_DIM / 2.0 - 0.1) & \
                  (pelvis_pos_x < object5.data.root_pos_w[:, 0] + BLOCK_X_DIM / 2.0 + 0.1) & \
                  (torch.abs(pelvis_pos[:, 1] - object5.data.root_pos_w[:, 1]) < BLOCK_Y_DIM / 2.0 + 0.1) # Check Y alignment

    # Condition for applying stability reward: not actively jumping or on block
    apply_stability_reward_condition = ~(is_jumping_low_wall | is_on_block)

    # Reward for maintaining pelvis height (negative absolute difference from target)
    # Uses relative Z distance for continuous reward.
    reward_pelvis_height = -torch.abs(pelvis_pos_z - target_pelvis_z)

    shaping_reward_2 = torch.where(apply_stability_reward_condition, reward_pelvis_height, 0.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_2)
        return scaled_reward
    return shaping_reward_2

def forward_progress_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_reward") -> torch.Tensor:
    """
    Provides a small, continuous reward for general forward movement along the x-axis,
    encouraging the robot to keep moving through the course.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot parts using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]

    # Access objects using approved patterns
    object5 = env.scene['Object5'] # Block cube (final object)

    # Calculate distance to the end of the course (e.g., center of Object5)
    # Uses relative X distance.
    dist_to_end_x = object5.data.root_pos_w[:, 0] - pelvis_pos_x

    # Reward for reducing distance to the end, but with diminishing returns as it gets very close
    # Uses a sigmoid-like function for continuous reward.
    reward_forward_progress = 1.0 / (1.0 + torch.exp(dist_to_end_x - 1.0))

    # Ensure it doesn't reward overshooting the final block
    # If pelvis is past the block's center + a small buffer, reward is 0. Uses relative position and hardcoded dimension.
    overshoot_threshold_x = object5.data.root_pos_w[:, 0] + BLOCK_X_DIM / 2.0 + 0.5
    shaping_reward_3 = torch.where(pelvis_pos_x < overshoot_threshold_x, reward_forward_progress, 0.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward_3)
        RewNormalizer.update_stats(normaliser_name, shaping_reward_3)
        return scaled_reward
    return shaping_reward_3

@configclass
class TaskRewardsCfg:
    # Primary reward for completing the main task objectives
    primary_reward = RewTerm(func=obstacle_course_primary_reward, weight=1.0,
                             params={"normalise": True, "normaliser_name": "primary_reward"})

    # Shaping reward for avoiding unintended collisions
    collision_avoidance_reward = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                         params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining stable pelvis height
    pelvis_height_stability_reward = RewTerm(func=pelvis_height_stability_reward, weight=0.2,
                                              params={"normalise": True, "normaliser_name": "pelvis_height_stability_reward"})

    # Shaping reward for general forward progress
    forward_progress_reward = RewTerm(func=forward_progress_reward, weight=0.1,
                                      params={"normalise": True, "normaliser_name": "forward_progress_reward"})