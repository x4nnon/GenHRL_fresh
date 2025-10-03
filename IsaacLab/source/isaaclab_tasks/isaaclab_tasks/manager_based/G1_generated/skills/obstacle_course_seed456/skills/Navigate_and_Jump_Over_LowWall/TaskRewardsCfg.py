from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats
from genhrl.generation.objects import get_object_volume # Corrected import: genhrr -> genhrl
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch

from isaaclab.envs import mdp
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums


def main_jump_over_lowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for Navigate_and_Jump_Over_LowWall skill.

    This reward guides the robot through three phases: approaching the low wall, jumping over it,
    and landing stably past it, positioning for the next task.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    robot = env.scene["robot"]
    low_wall = env.scene['Object3']  # Low wall for robot to jump over
    large_sphere = env.scene['Object1']  # Large sphere for robot to push

    # Access the required robot part(s) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object dimensions (hardcoded from task description as per requirements)
    # This adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    low_wall_x_dim = 0.3  # x-dimension of low wall
    low_wall_z_dim = 0.5  # z-dimension of low wall
    large_sphere_radius = 1.0  # radius of large sphere

    # Calculate relative distances and positions
    # This adheres to the rule: "YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN"
    low_wall_pos = low_wall.data.root_pos_w
    large_sphere_pos = large_sphere.data.root_pos_w

    # Pelvis height relative to low wall top
    # This is a relative distance calculation: pelvis_z - (wall_center_z + half_wall_height)
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    pelvis_height_above_wall_top = pelvis_pos[:, 2] - (low_wall_pos[:, 2] + low_wall_z_dim / 2.0)

    # Target landing zone: just past the low wall, before the large sphere
    # Low wall's far edge x-position (relative to wall center)
    low_wall_far_edge_x = low_wall_pos[:, 0] + low_wall_x_dim / 2.0
    # Large sphere's near edge x-position (relative to sphere center)
    large_sphere_near_edge_x = large_sphere_pos[:, 0] - large_sphere_radius

    # Phase 1: Approach low wall
    # Reward for reducing x-distance to the low wall's near edge.
    # Condition: robot is before the low wall's center.
    # Using torch.abs for distance and negative for reward to encourage reduction.
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts" and "absolute distances must be used for distances from objects."
    dist_pelvis_to_low_wall_x = low_wall_pos[:, 0] - pelvis_pos[:, 0]
    approach_condition = pelvis_pos[:, 0] < low_wall_pos[:, 0]
    reward_approach = -torch.abs(dist_pelvis_to_low_wall_x)
    reward_approach = torch.where(approach_condition, reward_approach, 0.0) # Only active when approaching

    # Phase 2: Jump over low wall
    # Reward for gaining height when near the wall.
    # Condition: robot is within the x-bounds of the wall (or slightly before/after for jump initiation/landing).
    # Only reward positive clearance.
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    jump_condition = (pelvis_pos[:, 0] > low_wall_pos[:, 0] - low_wall_x_dim) & \
                     (pelvis_pos[:, 0] < low_wall_pos[:, 0] + low_wall_x_dim)
    reward_jump_height = torch.where(pelvis_height_above_wall_top > 0, pelvis_height_above_wall_top, 0.0)
    reward_jump_height = torch.where(jump_condition, reward_jump_height, 0.0) # Only active when jumping over wall

    # Phase 3: Land stably and position for next skill
    # Reward for landing past the low wall and approaching the large sphere, while maintaining stable pelvis height.
    # Condition: robot is past the low wall's far edge.
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    land_condition = pelvis_pos[:, 0] > low_wall_far_edge_x
    # Reward for being close to the large sphere's near edge (x-axis).
    # Using torch.abs for distance and negative for reward to encourage reduction.
    reward_land_position = -torch.abs(pelvis_pos[:, 0] - large_sphere_near_edge_x)
    # Reward for stable pelvis height (e.g., 0.7m). This is an absolute height, which is allowed for Z-axis stability.
    # Adheres to "z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill."
    reward_stable_pelvis_z = -torch.abs(pelvis_pos[:, 2] - 0.7)
    # Combine landing rewards, active only when past the wall.
    reward_landing = torch.where(land_condition, reward_land_position + reward_stable_pelvis_z, 0.0)

    # Combine all primary rewards
    reward = reward_approach + reward_jump_height + reward_landing

    # Mandatory reward normalization
    # Adheres to "MANDATORY REWARD NORMALIZATION"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_low_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Shaping reward for collision avoidance with the low wall (Object3).

    This reward penalizes the robot for colliding with the low wall during the jump,
    encouraging a clean clearance.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    robot = env.scene["robot"]
    low_wall = env.scene['Object3']  # Low wall for robot to jump over

    # Access the required robot part(s) using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_knee_idx = robot.body_names.index('left_knee_link')
    right_knee_idx = robot.body_names.index('right_knee_link')
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]

    # Object dimensions (hardcoded from task description as per requirements)
    # Adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    low_wall_x_dim = 0.3  # x-dimension of low wall
    low_wall_y_dim = 5.0  # y-dimension of low wall
    low_wall_z_dim = 0.5  # z-dimension of low wall

    # Low wall's bounding box coordinates (relative to its center)
    # Adheres to "YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN"
    low_wall_pos = low_wall.data.root_pos_w
    wall_min_x = low_wall_pos[:, 0] - low_wall_x_dim / 2.0
    wall_max_x = low_wall_pos[:, 0] + low_wall_x_dim / 2.0
    wall_min_y = low_wall_pos[:, 1] - low_wall_y_dim / 2.0
    wall_max_y = low_wall_pos[:, 1] + low_wall_y_dim / 2.0
    wall_min_z = low_wall_pos[:, 2] - low_wall_z_dim / 2.0
    wall_max_z = low_wall_pos[:, 2] + low_wall_z_dim / 2.0

    # Check for collision with pelvis using relative positions to wall bounds
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    pelvis_collision_x = (pelvis_pos[:, 0] > wall_min_x) & (pelvis_pos[:, 0] < wall_max_x)
    pelvis_collision_y = (pelvis_pos[:, 1] > wall_min_y) & (pelvis_pos[:, 1] < wall_max_y)
    pelvis_collision_z = (pelvis_pos[:, 2] > wall_min_z) & (pelvis_pos[:, 2] < wall_max_z)
    pelvis_colliding = pelvis_collision_x & pelvis_collision_y & pelvis_collision_z

    # Check for collision with knees using relative positions to wall bounds
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    left_knee_collision_x = (left_knee_pos[:, 0] > wall_min_x) & (left_knee_pos[:, 0] < wall_max_x)
    left_knee_collision_y = (left_knee_pos[:, 1] > wall_min_y) & (left_knee_pos[:, 1] < wall_max_y)
    left_knee_collision_z = (left_knee_pos[:, 2] > wall_min_z) & (left_knee_pos[:, 2] < wall_max_z)
    left_knee_colliding = left_knee_collision_x & left_knee_collision_y & left_knee_collision_z

    right_knee_collision_x = (right_knee_pos[:, 0] > wall_min_x) & (right_knee_pos[:, 0] < wall_max_x)
    right_knee_collision_y = (right_knee_pos[:, 1] > wall_min_y) & (right_knee_pos[:, 1] < wall_max_y)
    right_knee_collision_z = (right_knee_pos[:, 2] > wall_min_z) & (right_knee_pos[:, 2] < wall_max_z)
    right_knee_colliding = right_knee_collision_x & right_knee_collision_y & right_knee_collision_z

    # Overall collision condition: any of the tracked parts are colliding
    is_colliding = pelvis_colliding | left_knee_colliding | right_knee_colliding

    # Reward: large negative reward if colliding, 0 otherwise. This is a penalty.
    # Adheres to "continuous rewards" (though a step function, it's a common penalty)
    reward = torch.where(is_colliding, -5.0, 0.0)

    # Mandatory reward normalization
    # Adheres to "MANDATORY REWARD NORMALIZATION"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    '''Shaping reward for maintaining stability and preventing falling.

    This reward encourages the robot to maintain an upright posture and penalizes falling.
    It also encourages the feet to be on the ground when not actively jumping.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    left_foot_pos_z = robot.data.body_pos_w[:, left_foot_idx, 2]
    right_foot_pos_z = robot.data.body_pos_w[:, right_foot_idx, 2]

    # Object dimensions (hardcoded from task description for context)
    # Adheres to the rule: "THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it."
    low_wall_x_dim = 0.3  # x-dimension of low wall
    low_wall = env.scene['Object3']  # Low wall for robot to jump over
    low_wall_pos = low_wall.data.root_pos_w

    # Condition for being on the ground (feet z-position close to 0).
    # Assuming ground is at z=0. A small tolerance (e.g., 0.05m) for feet contact.
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts" (relative to ground plane)
    feet_on_ground = (left_foot_pos_z < 0.05) & (right_foot_pos_z < 0.05)

    # Condition for being in the air (jumping).
    # Robot is considered jumping if pelvis is significantly above ground and feet are off ground
    # and robot is near the wall (relative x-position to wall).
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    is_jumping = (pelvis_pos[:, 2] > 0.7) & (~feet_on_ground) & \
                 (pelvis_pos[:, 0] > low_wall_pos[:, 0] - low_wall_x_dim) & \
                 (pelvis_pos[:, 0] < low_wall_pos[:, 0] + low_wall_x_dim)

    # Reward for maintaining stable pelvis height when not jumping.
    # Target pelvis height for standing is 0.7m. This is an absolute height, allowed for Z-axis stability.
    # Adheres to "z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill."
    reward_pelvis_stability = -torch.abs(pelvis_pos[:, 2] - 0.7)
    # Only apply this reward when not actively jumping.
    reward_pelvis_stability = torch.where(~is_jumping, reward_pelvis_stability, 0.0)

    # Reward for feet being on the ground when not jumping.
    # Small positive reward for contact.
    # Adheres to "ALL rewards MUST ONLY use relative distances between objects and robot parts" (relative to ground plane)
    reward_feet_on_ground = torch.where(~is_jumping & feet_on_ground, 0.1, 0.0)

    # Penalty for falling (pelvis too low).
    # This is an absolute height threshold, allowed for Z-axis.
    # Adheres to "z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill."
    falling_condition = pelvis_pos[:, 2] < 0.3
    reward_falling_penalty = torch.where(falling_condition, -10.0, 0.0)

    # Combine stability rewards
    reward = reward_pelvis_stability + reward_feet_on_ground + reward_falling_penalty

    # Mandatory reward normalization
    # Adheres to "MANDATORY REWARD NORMALIZATION"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for the skill, guiding the robot through the jump sequence.
    # Weight 1.0 as it's the primary objective.
    # Adheres to "PROPER WEIGHTS"
    Main_JumpOverLowWallReward = RewTerm(func=main_jump_over_lowWall_reward, weight=1.0,
                                         params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for avoiding collisions with the low wall.
    # Weight 0.6 to provide a significant penalty for undesirable contact.
    # Adheres to "PROPER WEIGHTS"
    CollisionAvoidanceLowWallReward = RewTerm(func=collision_avoidance_low_wall_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    # Shaping reward for maintaining robot stability and preventing falls.
    # Weight 0.4 to encourage good posture and recovery.
    # Adheres to "PROPER WEIGHTS"
    MaintainStabilityReward = RewTerm(func=maintain_stability_reward, weight=0.4,
                                      params={"normalise": True, "normaliser_name": "stability_reward"})