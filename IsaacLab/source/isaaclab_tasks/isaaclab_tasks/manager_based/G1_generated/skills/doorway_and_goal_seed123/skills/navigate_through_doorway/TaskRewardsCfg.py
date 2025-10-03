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


def primary_navigate_to_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_navigate_to_block_reward") -> torch.Tensor:
    """
    Primary reward for guiding the robot through the doorway and then to the small block (Object3).
    This reward is active only after the robot's pelvis has passed the doorway's exit y-coordinate.
    It encourages the robot to minimize the Euclidean distance to Object3.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object3 = env.scene['Object3'] # Small Block
    object3_pos = object3.data.root_pos_w

    # Access required robot part(s)
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Define doorway exit y-position
    # Task description: "Object3 is 2m past the doorway in the y axis"
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Requirement: NEVER use hard-coded positions or arbitrary thresholds.
    # The doorway exit y is 2m before Object3's y-position.
    doorway_exit_y = object3_pos[:, 1] - 2.0

    # Calculate Euclidean distance to Object3
    # Requirement: All operations must work with batched environments
    # Requirement: Use only relative distances between objects and robot parts
    distance_to_object3_x = object3_pos[:, 0] - pelvis_pos_x
    distance_to_object3_y = object3_pos[:, 1] - pelvis_pos_y
    # Object3 is 0.3m cubed, so its center z is 0.15m. Pelvis target z is 0.7m.
    # We want the pelvis to be at a reasonable height relative to the block.
    # The task is to walk *to* the block, not necessarily on top of it.
    # A fixed target height for the pelvis (0.7m) is generally good for standing.
    # Requirement: z is the only absolute position allowed, used sparingly.
    pelvis_target_z = 0.7
    
    # Euclidean distance squared to Object3
    # Requirement: Rewards should be continuous and positive where possible, or negative for penalties.
    # Negative distance encourages minimization.
    dist_sq_to_object3 = distance_to_object3_x**2 + distance_to_object3_y**2 + (pelvis_pos_z - pelvis_target_z)**2
    primary_reward = -torch.sqrt(dist_sq_to_object3)

    # Condition: Robot's pelvis has passed the doorway in the y-direction
    # Requirement: All operations must work with batched environments
    pelvis_past_doorway_condition = (pelvis_pos_y > doorway_exit_y)

    # Apply condition: reward is 0 if not past doorway, otherwise it's the distance reward
    primary_reward = torch.where(pelvis_past_doorway_condition, primary_reward, torch.tensor(0.0, device=env.device))

    # Requirement: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def shaping_doorway_alignment_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_doorway_alignment_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to align with the doorway and approach it.
    This reward is active only *before* the robot's pelvis passes the doorway.
    It also encourages maintaining a stable upright posture (pelvis z-height).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object3 = env.scene['Object3'] # Small Block (used for doorway exit y-pos)

    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Access required robot part(s)
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Doorway center x-position (midpoint between Object1 and Object2)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    doorway_center_x = (object1_pos[:, 0] + object2_pos[:, 0]) / 2.0

    # Doorway exit y-position (same as in primary reward)
    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    doorway_exit_y = object3_pos[:, 1] - 2.0

    # Distance to doorway center in x (penalize deviation from center)
    # Requirement: Use only relative distances between objects and robot parts
    dist_x_to_doorway_center = torch.abs(pelvis_pos_x - doorway_center_x)
    # Requirement: Rewards should be continuous and negative for penalties.
    reward_x_alignment = -dist_x_to_doorway_center # Negative reward for misalignment

    # Distance to doorway in y (encourage approaching the doorway)
    # Reward is higher (less negative) as robot gets closer to the doorway's y-plane.
    # Requirement: Use only relative distances between objects and robot parts
    dist_y_to_doorway = torch.abs(doorway_exit_y - pelvis_pos_y)
    # Requirement: Rewards should be continuous and negative for penalties.
    reward_y_approach = -dist_y_to_doorway # Negative reward for distance to doorway

    # Pelvis height reward (encourage standing upright)
    # Requirement: z is the only absolute position allowed, used sparingly.
    pelvis_z_target = 0.7 # Target height for pelvis
    # Requirement: Rewards should be continuous and negative for penalties.
    pelvis_height_reward = -torch.abs(pelvis_pos_z - pelvis_z_target) # Negative reward for deviation from target height

    # Condition: Robot's pelvis is before or at the doorway's exit y-coordinate
    # Requirement: All operations must work with batched environments
    pelvis_before_doorway_condition = (pelvis_pos_y <= doorway_exit_y)

    # Combine rewards
    shaping_reward1 = reward_x_alignment + reward_y_approach + pelvis_height_reward

    # Apply condition: reward is 0 if past doorway, otherwise it's the combined shaping reward
    shaping_reward1 = torch.where(pelvis_before_doorway_condition, shaping_reward1, torch.tensor(0.0, device=env.device))

    # Requirement: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to penalize the robot for getting too close to the heavy cubes (Object1 and Object2)
    that form the doorway. Applies to multiple robot body parts.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)

    object1_pos = object1.data.root_pos_w
    object2_pos = object2.data.root_pos_w

    # Wall dimensions (from description: x=0.5m, y=5m, z=1.5m)
    # Requirement: THERE IS NO way to access the SIZE of an object. Hardcode values from config.
    wall_x_dim = 0.5
    wall_y_dim = 5.0
    wall_z_dim = 1.5

    # Robot parts for collision avoidance
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    robot_parts_to_check = ['pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link',
                            'left_palm_link', 'right_palm_link', 'head_link']
    
    # Initialize collision reward for all environments
    # Requirement: All operations must work with batched environments
    # Use a 1D tensor over environments (shape: [num_envs])
    collision_reward = torch.zeros_like(object1_pos[:, 0])

    # Collision threshold for x-direction (distance from wall face)
    # This is a buffer distance, if a part is closer than this, it gets penalized.
    collision_threshold_x = 0.2 # Robot part radius + small buffer for clearance

    for part_name in robot_parts_to_check:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx]
        part_pos_x = part_pos[:, 0]
        part_pos_y = part_pos[:, 1]
        part_pos_z = part_pos[:, 2]

        # For Object1 (left wall)
        # Object1 is the left wall, so its inner face is its right-most x-coordinate.
        # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
        object1_inner_x = object1_pos[:, 0] + (wall_x_dim / 2.0)
        
        # Y and Z extents of Object1 to check if robot part is within the wall's volume in those dimensions
        # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
        object1_y_min = object1_pos[:, 1] - (wall_y_dim / 2.0)
        object1_y_max = object1_pos[:, 1] + (wall_y_dim / 2.0)
        object1_z_min = object1_pos[:, 2] - (wall_z_dim / 2.0)
        object1_z_max = object1_pos[:, 2] + (wall_z_dim / 2.0)

        # Distance from robot part to Object1's inner x-face (positive if part is to the right of the face)
        # Requirement: Use only relative distances between objects and robot parts
        dist_x_obj1 = part_pos_x - object1_inner_x
        
        # Check if part is within y and z bounds of the wall
        # Requirement: All operations must work with batched environments
        in_y_bounds_obj1 = (part_pos_y > object1_y_min) & (part_pos_y < object1_y_max)
        in_z_bounds_obj1 = (part_pos_z > object1_z_min) & (part_pos_z < object1_z_max)

        # Penalize if too close to Object1's inner x-face and within y/z bounds
        # We want to penalize if dist_x_obj1 is small and positive (robot part is just to the right of the wall's inner face)
        # Requirement: Rewards should be continuous and negative for penalties.
        # Using exponential decay for a smooth, strong penalty when very close.
        reward_obj1_x = torch.where(
            (dist_x_obj1 < collision_threshold_x) & (dist_x_obj1 > 0) & in_y_bounds_obj1 & in_z_bounds_obj1,
            -torch.exp(-dist_x_obj1 / collision_threshold_x), # More negative as dist_x_obj1 approaches 0
            torch.tensor(0.0, device=env.device)
        )
        collision_reward += reward_obj1_x

        # For Object2 (right wall)
        # Object2 is the right wall, so its inner face is its left-most x-coordinate.
        # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
        object2_inner_x = object2_pos[:, 0] - (wall_x_dim / 2.0)
        
        # Y and Z extents of Object2
        # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
        object2_y_min = object2_pos[:, 1] - (wall_y_dim / 2.0)
        object2_y_max = object2_pos[:, 1] + (wall_y_dim / 2.0)
        object2_z_min = object2_pos[:, 2] - (wall_z_dim / 2.0)
        object2_z_max = object2_pos[:, 2] + (wall_z_dim / 2.0)

        # Distance from robot part to Object2's inner x-face (positive if part is to the left of the face)
        # Requirement: Use only relative distances between objects and robot parts
        dist_x_obj2 = object2_inner_x - part_pos_x

        # Check if part is within y and z bounds of the wall
        # Requirement: All operations must work with batched environments
        in_y_bounds_obj2 = (part_pos_y > object2_y_min) & (part_pos_y < object2_y_max)
        in_z_bounds_obj2 = (part_pos_z > object2_z_min) & (part_pos_z < object2_z_max)

        # Penalize if too close to Object2's inner x-face and within y/z bounds
        # We want to penalize if dist_x_obj2 is small and positive (robot part is just to the left of the wall's inner face)
        # Requirement: Rewards should be continuous and negative for penalties.
        reward_obj2_x = torch.where(
            (dist_x_obj2 < collision_threshold_x) & (dist_x_obj2 > 0) & in_y_bounds_obj2 & in_z_bounds_obj2,
            -torch.exp(-dist_x_obj2 / collision_threshold_x), # More negative as dist_x_obj2 approaches 0
            torch.tensor(0.0, device=env.device)
        )
        collision_reward += reward_obj2_x
    
    shaping_reward2 = collision_reward

    # Requirement: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # Primary reward for navigating to the block after passing the doorway
    # Requirement: Primary reward weight ~1.0
    PrimaryNavigateToBlockReward = RewTerm(func=primary_navigate_to_block_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "primary_navigate_to_block_reward"})
    
    # Shaping reward for aligning with and approaching the doorway, and maintaining posture
    # Requirement: Supporting reward weights <1.0
    ShapingDoorwayAlignmentReward = RewTerm(func=shaping_doorway_alignment_reward, weight=0.6,
                                            params={"normalise": True, "normaliser_name": "shaping_doorway_alignment_reward"})
    
    # Shaping reward for avoiding collisions with the doorway walls
    # Requirement: Supporting reward weights <1.0
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.3,
                                              params={"normalise": True, "normaliser_name": "shaping_collision_avoidance_reward"})