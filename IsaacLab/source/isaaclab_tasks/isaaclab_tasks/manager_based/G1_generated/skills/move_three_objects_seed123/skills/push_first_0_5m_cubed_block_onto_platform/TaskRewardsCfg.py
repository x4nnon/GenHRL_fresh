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


def main_push_block_onto_platform_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for pushing the 'first 0.5m cubed block' (Object1) entirely onto the 'platform' (Object4).
    This reward encourages horizontal proximity, correct Z-height, and being within the platform's bounds.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'
    object4 = env.scene['Object4'] # 'platform'

    # Object1 dimensions (0.5m cubed block) - NEVER use data.size or data.root_size, hardcode from config
    object1_half_size = 0.25 # 0.5m / 2

    # Object4 dimensions (platform x=2m y=2m) - NEVER use data.size or data.root_size, hardcode from config
    platform_half_x = 1.0 # 2m / 2
    platform_half_y = 1.0 # 2m / 2
    platform_z_thickness = 0.001 # from description, z=0.001 means thickness is 0.001

    # Calculate distance vector between Object1 and Object4 center
    # ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Object4's root_pos_w is its center.
    distance_obj1_obj4_x = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    distance_obj1_obj4_y = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    # Target Z for Object1 resting on platform: platform_center_z + platform_half_z + object1_half_size
    # Since platform_z is given as 0.001 (thickness), its center is at root_pos_w[:, 2] + platform_z_thickness / 2.
    # The top surface is at root_pos_w[:, 2] + platform_z_thickness.
    # So, Object1's center should be at (platform_top_surface_z + object1_half_size)
    target_obj1_z = object4.data.root_pos_w[:, 2] + platform_z_thickness + object1_half_size
    distance_obj1_obj4_z = object1.data.root_pos_w[:, 2] - target_obj1_z

    # Reward for horizontal proximity of Object1 to Object4 center
    # Use a small constant to prevent division by zero and ensure positive reward for being close
    # Rewards should be continuous and positive for being closer
    horizontal_dist_reward = -torch.sqrt(distance_obj1_obj4_x**2 + distance_obj1_obj4_y**2)

    # Reward for Object1 being at the correct Z height relative to the platform
    z_height_reward = -torch.abs(distance_obj1_obj4_z)

    # Reward for Object1 being within the X and Y bounds of the platform
    # Object1's center must be within platform_half_x and platform_half_y from platform center
    # Consider Object1's half size for "entirely on"
    # NEVER use hard-coded positions or arbitrary thresholds. Thresholds derived from object dimensions.
    on_platform_x_condition = (torch.abs(distance_obj1_obj4_x) <= (platform_half_x - object1_half_size))
    on_platform_y_condition = (torch.abs(distance_obj1_obj4_y) <= (platform_half_y - object1_half_size))

    # Reward for being within bounds, higher when both conditions are met.
    # This can be made continuous by using a sigmoid or similar function, but for "entirely on", a binary-like
    # reward with a smooth transition is often used. Here, we'll use a simple multiplication for continuity.
    # A higher reward for being within bounds.
    bounds_reward = (on_platform_x_condition.float() + on_platform_y_condition.float()) / 2.0 # 0.0 if neither, 0.5 if one, 1.0 if both

    # Combine rewards. Weights can be tuned.
    # Ensure rewards are continuous and positive for desired behavior.
    # Negative distances become positive rewards when multiplied by -1.
    reward = (horizontal_dist_reward * 0.5) + (z_height_reward * 0.3) + (bounds_reward * 0.2)

    # MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_hand_proximity_to_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward encouraging the robot's 'right_palm_link' to get close to and maintain contact with
    'first 0.5m cubed block' (Object1) for pushing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'

    # Access the required robot part(s)
    # ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    robot_hand_idx = robot.body_names.index('right_palm_link')
    robot_hand_pos = robot.data.body_pos_w[:, robot_hand_idx]

    # Object1 dimensions (0.5m cubed block) - hardcoded from config
    object1_half_size = 0.25

    # Calculate distance vector between robot hand and Object1
    # ALL rewards MUST ONLY use relative distances between objects and robot parts
    distance_hand_obj1_x = object1.data.root_pos_w[:, 0] - robot_hand_pos[:, 0]
    distance_hand_obj1_y = object1.data.root_pos_w[:, 1] - robot_hand_pos[:, 1]
    distance_hand_obj1_z = object1.data.root_pos_w[:, 2] - robot_hand_pos[:, 2]

    # Thresholds for proximity and alignment - NEVER use arbitrary thresholds, derive from context
    # Proximity threshold slightly more than half block size to allow contact
    proximity_threshold_xy = object1_half_size + 0.05 # 0.25 + 0.05 = 0.3m
    z_alignment_threshold = 0.2 # Allow some vertical variation for pushing, but keep it reasonable

    # Condition for activation: hand is close to Object1 and roughly aligned vertically
    # This ensures the reward is only active when the hand is in a plausible pushing position.
    activation_condition = (torch.abs(distance_hand_obj1_x) < proximity_threshold_xy) & \
                           (torch.abs(distance_hand_obj1_y) < proximity_threshold_xy) & \
                           (torch.abs(distance_hand_obj1_z) < z_alignment_threshold)

    # Reward for inverse distance to Object1's center, when condition is met
    # Use negative Euclidean distance to make closer mean higher reward (closer to 0).
    hand_proximity_reward_raw = -torch.sqrt(distance_hand_obj1_x**2 + distance_hand_obj1_y**2 + distance_hand_obj1_z**2)

    # Apply the activation condition. Reward is 0 if not activated.
    reward = torch.where(activation_condition, hand_proximity_reward_raw, torch.tensor(0.0, device=env.device))

    # MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_stability_and_readiness_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_readiness_reward") -> torch.Tensor:
    """
    Shaping reward encouraging robot stability (pelvis height) and readiness to approach the next block (Object2)
    after Object1 is pushed onto the platform.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'
    object2 = env.scene['Object2'] # 'second 0.5m cubed block'
    object4 = env.scene['Object4'] # 'platform'

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object1 dimensions (0.5m cubed block) - hardcoded from config
    object1_half_size = 0.25

    # Object4 dimensions (platform x=2m y=2m) - hardcoded from config
    platform_half_x = 1.0
    platform_half_y = 1.0
    platform_z_thickness = 0.001

    # Condition for Object1 being on platform (re-used from primary reward for consistency)
    # ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_obj1_obj4_x = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    dist_obj1_obj4_y = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]
    on_platform_x_condition = (torch.abs(dist_obj1_obj4_x) <= (platform_half_x - object1_half_size))
    on_platform_y_condition = (torch.abs(dist_obj1_obj4_y) <= (platform_half_y - object1_half_size))
    object1_on_platform = on_platform_x_condition & on_platform_y_condition

    # Reward for maintaining stable pelvis height (around 0.7m)
    # z is the only absolute position allowed for height.
    target_pelvis_z = 0.7 # A common stable pelvis height for humanoid robots
    pelvis_height_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for robot's pelvis being in a good position relative to Object2 after Object1 is on platform
    # This encourages the robot to move towards Object2's general area, but not too close yet.
    # ALL rewards MUST ONLY use relative distances between objects and robot parts
    distance_pelvis_obj2_x = object2.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    distance_pelvis_obj2_y = object2.data.root_pos_w[:, 1] - pelvis_pos[:, 1]
    distance_pelvis_obj2_horizontal = torch.sqrt(distance_pelvis_obj2_x**2 + distance_pelvis_obj2_y**2)

    # Penalize being too far from Object2's initial horizontal position.
    # This reward should be negative, so closer means less negative (higher reward).
    ready_pos_reward = -distance_pelvis_obj2_horizontal

    # Combine rewards, activate ready_pos_reward only when Object1 is on platform
    # Ensure rewards are continuous.
    reward = pelvis_height_reward * 0.5 # Pelvis height is always rewarded
    reward += torch.where(object1_on_platform, ready_pos_reward * 0.5, torch.tensor(0.0, device=env.device))

    # MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward penalizing collisions between robot body parts and objects,
    excluding intended pushing contact with Object1.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'
    object2 = env.scene['Object2'] # 'second 0.5m cubed block'
    object3 = env.scene['Object3'] # 'third 0.5m cubed block'
    object4 = env.scene['Object4'] # 'platform'

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')]
    right_hand_pos = robot.data.body_pos_w[:, robot.body_names.index('right_palm_link')]
    left_hand_pos = robot.data.body_pos_w[:, robot.body_names.index('left_palm_link')]
    right_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('right_ankle_roll_link')]
    left_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('left_ankle_roll_link')]

    # Object dimensions (0.5m cubed blocks) - hardcoded from config
    block_half_size = 0.25
    # Platform dimensions (x=2m y=2m) - hardcoded from config
    platform_half_x = 1.0
    platform_half_y = 1.0
    platform_z_thickness = 0.001 # z-dimension of platform

    collision_penalty = torch.zeros(env.num_envs, device=env.device)
    # Collision threshold: a small buffer to detect near-collisions or actual contact.
    # This should be small, representing the "skin" of the robot part.
    collision_threshold = 0.05 # meters, a small buffer for collision detection

    # Function to calculate collision penalty for a robot part and a box object
    # ALL rewards MUST ONLY use relative distances between objects and robot parts
    def calculate_box_collision_penalty(robot_part_pos, obj_pos, obj_half_x, obj_half_y, obj_half_z):
        # Calculate absolute distances from object center to robot part center in each dimension
        dist_x = torch.abs(robot_part_pos[:, 0] - obj_pos[:, 0])
        dist_y = torch.abs(robot_part_pos[:, 1] - obj_pos[:, 1])
        dist_z = torch.abs(robot_part_pos[:, 2] - obj_pos[:, 2])

        # Check if robot part is "overlapping" or very close to the object's bounding box
        # This is a simplified check. A more robust one would consider robot part dimensions.
        # Here, we assume robot parts are point-like for simplicity relative to object size.
        x_overlap = (dist_x < (obj_half_x + collision_threshold))
        y_overlap = (dist_y < (obj_half_y + collision_threshold))
        z_overlap = (dist_z < (obj_half_z + collision_threshold))

        # Penalize if there's overlap in all dimensions.
        # Use a continuous penalty based on how deep the overlap is, or a fixed penalty.
        # For simplicity, a fixed negative penalty when collision is detected.
        # This is a binary-like penalty, but it's common for collision avoidance.
        return torch.where(x_overlap & y_overlap & z_overlap, -1.0, 0.0)

    # Penalize collisions with Object1 (first block)
    # Exclude the right_palm_link from Object1 collision penalty as it's intended for pushing.
    collision_penalty += calculate_box_collision_penalty(pelvis_pos, object1.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(left_hand_pos, object1.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(right_foot_pos, object1.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(left_foot_pos, object1.data.root_pos_w, block_half_size, block_half_size, block_half_size)

    # Collisions with Object2 (second block)
    collision_penalty += calculate_box_collision_penalty(pelvis_pos, object2.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(right_hand_pos, object2.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(left_hand_pos, object2.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(right_foot_pos, object2.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(left_foot_pos, object2.data.root_pos_w, block_half_size, block_half_size, block_half_size)

    # Collisions with Object3 (third block)
    collision_penalty += calculate_box_collision_penalty(pelvis_pos, object3.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(right_hand_pos, object3.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(left_hand_pos, object3.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(right_foot_pos, object3.data.root_pos_w, block_half_size, block_half_size, block_half_size)
    collision_penalty += calculate_box_collision_penalty(left_foot_pos, object3.data.root_pos_w, block_half_size, block_half_size, block_half_size)

    # Collisions with Object4 (platform)
    # Platform is at z=0.001, so its half_z is very small.
    # For collision, consider the robot part being below the platform's top surface or inside its volume.
    # The platform is 2x2m, so half_x and half_y are 1.0m.
    collision_penalty += calculate_box_collision_penalty(pelvis_pos, object4.data.root_pos_w, platform_half_x, platform_half_y, platform_z_thickness / 2.0)
    collision_penalty += calculate_box_collision_penalty(right_hand_pos, object4.data.root_pos_w, platform_half_x, platform_half_y, platform_z_thickness / 2.0)
    collision_penalty += calculate_box_collision_penalty(left_hand_pos, object4.data.root_pos_w, platform_half_x, platform_half_y, platform_z_thickness / 2.0)
    collision_penalty += calculate_box_collision_penalty(right_foot_pos, object4.data.root_pos_w, platform_half_x, platform_half_y, platform_z_thickness / 2.0)
    collision_penalty += calculate_box_collision_penalty(left_foot_pos, object4.data.root_pos_w, platform_half_x, platform_half_y, platform_z_thickness / 2.0)

    reward = collision_penalty

    # MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for pushing the block onto the platform
    MainPushBlockOntoPlatform = RewTerm(func=main_push_block_onto_platform_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for robot hand proximity to the block
    RobotHandProximityToBlock = RewTerm(func=robot_hand_proximity_to_block_reward, weight=0.4,
                                        params={"normalise": True, "normaliser_name": "hand_proximity_reward"})

    # Shaping reward for robot stability and readiness for the next skill
    RobotStabilityAndReadiness = RewTerm(func=robot_stability_and_readiness_reward, weight=0.3,
                                         params={"normalise": True, "normaliser_name": "stability_readiness_reward"})

    # Shaping reward for collision avoidance
    CollisionAvoidance = RewTerm(func=collision_avoidance_reward, weight=0.2, # Weight is positive, penalty is negative
                                 params={"normalise": True, "normaliser_name": "collision_reward"})