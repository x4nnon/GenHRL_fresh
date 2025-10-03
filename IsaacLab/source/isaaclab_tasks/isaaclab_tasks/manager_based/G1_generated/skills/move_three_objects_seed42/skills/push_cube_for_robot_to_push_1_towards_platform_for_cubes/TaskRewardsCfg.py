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

def primary_push_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_push_reward") -> torch.Tensor:
    """
    Primary reward for pushing Object1 towards Object4 (Platform for cubes).
    Encourages Object1 to move closer to Object4 in the XY plane and maintain a correct Z height relative to the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    # Hardcode Object1 dimensions (from skill info: 0.5m cubed block)
    # Reasoning: Object dimensions must be hardcoded from the object configuration, not accessed from the object itself.
    object1_half_size = 0.25 # 0.5 / 2

    # Calculate distance vector between Object1 and Object4
    # Reasoning: All rewards MUST ONLY use relative distances between objects.
    distance_x = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    distance_y = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    distance_z_obj = object1.data.root_pos_w[:, 2] - object4.data.root_pos_w[:, 2]

    # Reward for Object1 moving towards Object4 (x and y components)
    # Using negative Euclidean distance in XY plane, so closer is higher reward.
    # Adding a constant to ensure positive reward and gradient for better RL learning.
    # Reasoning: Rewards should be continuous and positive where possible.
    reward_xy = -torch.sqrt(distance_x**2 + distance_y**2) + 5.0

    # Reward for Object1 staying on the ground (z component relative to platform)
    # Object1's z-center should be around object4's z-center + object1_half_size.
    # Object4's z is 0.001 (from skill info), so Object1's target z should be around 0.001 + 0.25 = 0.251.
    # Reasoning: Relative distance for Z component, ensuring the cube is on the platform.
    target_object1_z = object4.data.root_pos_w[:, 2] + object1_half_size
    reward_z = -torch.abs(object1.data.root_pos_w[:, 2] - target_object1_z)

    # Combine rewards
    reward = reward_xy + reward_z

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_hand_approach_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_hand_approach_reward") -> torch.Tensor:
    """
    Shaping reward encouraging the robot's right hand to approach Object1.
    Active until the hand is very close to Object1, then primary reward takes over.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object1 = env.scene['Object1']
    robot = env.scene["robot"]
    robot_hand_idx = robot.body_names.index('right_palm_link') # Reasoning: Accessing robot part index using approved pattern.
    robot_hand_pos = robot.data.body_pos_w[:, robot_hand_idx] # Reasoning: Accessing robot part position using approved pattern.

    # Calculate distance vector between Object1 and robot hand
    # Reasoning: All rewards MUST ONLY use relative distances between objects and robot parts.
    distance_x = object1.data.root_pos_w[:, 0] - robot_hand_pos[:, 0]
    distance_y = object1.data.root_pos_w[:, 1] - robot_hand_pos[:, 1]
    distance_z = object1.data.root_pos_w[:, 2] - robot_hand_pos[:, 2]

    # Total Euclidean distance
    hand_to_object1_dist = torch.sqrt(distance_x**2 + distance_y**2 + distance_z**2)

    # Activation condition: Reward is active when hand is further than 0.1m from Object1.
    # This encourages approach, and then the primary reward takes over for pushing.
    # Reasoning: Continuous reward with an activation condition to guide behavior.
    activation_condition = hand_to_object1_dist > 0.1

    # Reward is negative distance, so closer is higher reward. Add a small constant for positive gradient.
    # Reasoning: Rewards should be continuous and positive where possible.
    shaping_reward = -hand_to_object1_dist + 1.0

    # Apply activation condition
    reward = torch.where(activation_condition, shaping_reward, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_posture_collision_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_posture_collision_reward") -> torch.Tensor:
    """
    Shaping reward encouraging stable, upright posture (pelvis height) and avoiding collisions
    with all cubes (Object1, Object2, Object3) and the platform (Object4) using robot feet.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']
    robot = env.scene["robot"]

    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    robot_left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    robot_left_foot_pos = robot.data.body_pos_w[:, robot_left_foot_idx]
    robot_right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    robot_right_foot_pos = robot.data.body_pos_w[:, robot_right_foot_idx]

    # Pelvis height reward
    # Reasoning: Relative distance for Z component (height), continuous reward.
    target_pelvis_z = 0.7 # Target height for pelvis
    pelvis_height_reward = -torch.abs(robot_pelvis_pos_z - target_pelvis_z) + 0.5 # Add constant for positive gradient

    # Collision avoidance for feet with cubes and platform
    # Hardcode object dimensions from skill info
    # Reasoning: Object dimensions must be hardcoded from the object configuration.
    cube_half_size = 0.25 # 0.5m cubed block
    platform_half_x = 1.0 # 2m platform
    platform_half_y = 1.0 # 2m platform
    platform_half_z = 0.0005 # 0.001m platform

    # Function to calculate collision penalty for a robot part with an object
    # Reasoning: Uses relative distances to check for proximity/collision.
    def calculate_collision_penalty(robot_part_pos, obj_pos, obj_half_size_x, obj_half_size_y, obj_half_size_z):
        dist_x = torch.abs(robot_part_pos[:, 0] - obj_pos[:, 0])
        dist_y = torch.abs(robot_part_pos[:, 1] - obj_pos[:, 1])
        dist_z = torch.abs(robot_part_pos[:, 2] - obj_pos[:, 2])

        # Define a proximity threshold for collision penalty
        proximity_threshold = 0.2 # meters, slightly larger than object half-size for early warning

        # Check if robot part is within object bounds + proximity_threshold
        # This creates a continuous penalty based on how close the part is to the object's surface.
        # A fixed penalty is applied if within the threshold.
        is_colliding = (dist_x < (obj_half_size_x + proximity_threshold)) & \
                       (dist_y < (obj_half_size_y + proximity_threshold)) & \
                       (dist_z < (obj_half_size_z + proximity_threshold))
        
        # Penalize based on proximity. A simple fixed penalty for being too close.
        penalty = torch.where(is_colliding, -1.0, 0.0)
        return penalty

    collision_penalty_feet = 0.0
    # Penalties for left foot with cubes
    collision_penalty_feet += calculate_collision_penalty(robot_left_foot_pos, object1.data.root_pos_w, cube_half_size, cube_half_size, cube_half_size)
    collision_penalty_feet += calculate_collision_penalty(robot_left_foot_pos, object2.data.root_pos_w, cube_half_size, cube_half_size, cube_half_size)
    collision_penalty_feet += calculate_collision_penalty(robot_left_foot_pos, object3.data.root_pos_w, cube_half_size, cube_half_size, cube_half_size)
    # Penalties for right foot with cubes
    collision_penalty_feet += calculate_collision_penalty(robot_right_foot_pos, object1.data.root_pos_w, cube_half_size, cube_half_size, cube_half_size)
    collision_penalty_feet += calculate_collision_penalty(robot_right_foot_pos, object2.data.root_pos_w, cube_half_size, cube_half_size, cube_half_size)
    collision_penalty_feet += calculate_collision_penalty(robot_right_foot_pos, object3.data.root_pos_w, cube_half_size, cube_half_size, cube_half_size)

    # Penalties for feet with platform (Object4)
    collision_penalty_feet += calculate_collision_penalty(robot_left_foot_pos, object4.data.root_pos_w, platform_half_x, platform_half_y, platform_half_z)
    collision_penalty_feet += calculate_collision_penalty(robot_right_foot_pos, object4.data.root_pos_w, platform_half_x, platform_half_y, platform_half_z)

    reward = pelvis_height_reward + collision_penalty_feet

    # Mandatory reward normalization
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
    Reward terms for the push_cube_for_robot_to_push_1_towards_platform_for_cubes skill.
    """
    # Primary reward for pushing Object1 towards Object4
    primary_push_reward = RewTerm(func=primary_push_reward, weight=1.0,
                                  params={"normalise": True, "normaliser_name": "primary_push_reward"})

    # Shaping reward for the robot's right hand to approach Object1
    shaping_hand_approach_reward = RewTerm(func=shaping_hand_approach_reward, weight=0.4,
                                           params={"normalise": True, "normaliser_name": "shaping_hand_approach_reward"})

    # Shaping reward for maintaining posture and avoiding collisions
    shaping_posture_collision_reward = RewTerm(func=shaping_posture_collision_reward, weight=0.3,
                                                params={"normalise": True, "normaliser_name": "shaping_posture_collision_reward"})