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
# "Object1": "Cylinder Column 1 for the robot to knock over",
# "Object2": "Cylinder Column 2 for the robot to knock over",
# "Object3": "Cylinder Column 3 for the robot to knock over",
# "Object4": "Cylinder Column 4 for the robot to knock over",
# "Object5": "Cylinder Column 5 for the robot to knock over"
# All cylinder columns have z dimension of 2m and a radius of 0.3m.
CYLINDER_RADIUS = 0.3
CYLINDER_HEIGHT = 2.0

def main_knock_over_cylinderColumn5_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'knock_over_cylinderColumn5' skill.
    This reward guides the robot to first approach Object5 and then encourages it to knock Object5 over
    until it lies flat on the floor. It also penalizes overshooting Object5 towards Object4.
    """
    # Get normalizer instance
    # CRITICAL RULE: DO NOT DEFINE or IMPORT RewNormalizer, it's handled by get_normalizer
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    # CRITICAL RULE: ALWAYS access objects using env.scene['ObjectName']
    object5 = env.scene['Object5']
    object4 = env.scene['Object4'] # Used for overshoot penalty
    robot = env.scene["robot"]

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object5_pos = object5.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Calculate horizontal distance to Object5 (using pelvis as primary approach body part)
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    horizontal_dist_to_object5 = torch.norm(object5_pos[:, :2] - pelvis_pos[:, :2], dim=1)

    # Calculate Object5's current Z-position (absolute Z is allowed for height checks)
    object5_z_pos = object5_pos[:, 2]

    # Phase 1: Approach reward (when Object5 is still standing)
    # Reward for reducing horizontal distance to Object5. Negative distance means positive reward for getting closer.
    # CRITICAL RULE: Rewards should be continuous.
    approach_reward = -horizontal_dist_to_object5

    # Phase 2: Knock over reward (when Object5 is falling or fallen)
    # Reward for reducing Object5's Z-position towards its radius (when it's lying flat).
    # A cylinder lying flat will have its root_pos_w[:, 2] approximately equal to its radius.
    # CRITICAL RULE: Rewards should be continuous.
    knock_over_reward = -torch.abs(object5_z_pos - CYLINDER_RADIUS)

    # Condition for switching between phases: Object5's Z-position
    # If Object5 is still standing (Z > CYLINDER_RADIUS + small_threshold), use approach reward.
    # Otherwise (Z <= CYLINDER_RADIUS + small_threshold), use knock over reward.
    # A small threshold is added to ensure the switch happens clearly after it starts falling.
    # CRITICAL RULE: NEVER use arbitrary thresholds for positions, but small offsets for state transitions are acceptable.
    switch_threshold = CYLINDER_RADIUS + 0.1 # e.g., 0.3 + 0.1 = 0.4m

    # CRITICAL RULE: All operations must work with batched environments
    reward = torch.where(object5_z_pos > switch_threshold, approach_reward, knock_over_reward)

    # Penalize robot for overshooting Object5 significantly towards Object4
    # This assumes Object4 is further along the X-axis than Object5, and the robot approaches from a smaller X.
    # The robot should ideally stop around Object5's X position or slightly past it, but not near Object4's X position.
    # A penalty for being too far past Object5's X position.
    # The threshold is relative to Object5's X position.
    # CRITICAL RULE: NEVER use hard-coded positions.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # The overshoot penalty threshold is relative to Object5's X position.
    overshoot_penalty_threshold_x = object5_pos[:, 0] + 2.0 # 2m past Object5's X, assuming objects are 4m apart.
    overshoot_penalty = torch.where(pelvis_pos[:, 0] > overshoot_penalty_threshold_x,
                                    -torch.abs(pelvis_pos[:, 0] - overshoot_penalty_threshold_x),
                                    0.0)
    # CRITICAL RULE: Rewards should be continuous.
    reward += overshoot_penalty

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def robot_stability_and_contact_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_contact_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain a stable standing posture (pelvis Z-position)
    while interacting with Object5. It also encourages the robot's hands to be close to Object5,
    facilitating the push.
    """
    # Get normalizer instance
    # CRITICAL RULE: DO NOT DEFINE or IMPORT RewNormalizer, it's handled by get_normalizer
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    # CRITICAL RULE: ALWAYS access objects using env.scene['ObjectName']
    object5 = env.scene['Object5']
    robot = env.scene["robot"]

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]
    left_palm_idx = robot.body_names.index('left_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]

    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object5_pos = object5.data.root_pos_w

    # Target pelvis Z-position for stability (a reasonable standing height for the robot)
    # CRITICAL RULE: NEVER use hard-coded positions or arbitrary thresholds for object positions.
    # However, a target height for the robot's own body part (pelvis) is acceptable as it's a self-referential stability goal.
    target_pelvis_z = 0.7 # Example stable pelvis height

    # Reward for maintaining stable pelvis height
    # CRITICAL RULE: Rewards should be continuous.
    stability_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for hands being close to Object5 (encourages contact)
    # Aim for the object's X/Y and a Z that is within the cylinder's height, e.g., mid-height.
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    object5_mid_height_z = object5_pos[:, 2] + (CYLINDER_HEIGHT / 2.0)

    # Distance from right palm to Object5's mid-height center
    dist_right_palm_to_object5_x = object5_pos[:, 0] - right_palm_pos[:, 0]
    dist_right_palm_to_object5_y = object5_pos[:, 1] - right_palm_pos[:, 1]
    dist_right_palm_to_object5_z = object5_mid_height_z - right_palm_pos[:, 2]
    # CRITICAL RULE: Rewards should be continuous.
    hand_proximity_reward_right = -torch.sqrt(dist_right_palm_to_object5_x**2 + dist_right_palm_to_object5_y**2 + dist_right_palm_to_object5_z**2)

    # Distance from left palm to Object5's mid-height center
    dist_left_palm_to_object5_x = object5_pos[:, 0] - left_palm_pos[:, 0]
    dist_left_palm_to_object5_y = object5_pos[:, 1] - left_palm_pos[:, 1]
    dist_left_palm_to_object5_z = object5_mid_height_z - left_palm_pos[:, 2]
    # CRITICAL RULE: Rewards should be continuous.
    hand_proximity_reward_left = -torch.sqrt(dist_left_palm_to_object5_x**2 + dist_left_palm_to_object5_y**2 + dist_left_palm_to_object5_z**2)

    # Take the maximum of the two hands' proximity rewards, as one hand might be sufficient for pushing.
    hand_proximity_reward = torch.max(hand_proximity_reward_right, hand_proximity_reward_left)

    # Activation condition: Only apply hand proximity reward when Object5 is still standing (or just starting to fall)
    # and the robot is relatively close to the object horizontally.
    object5_z_pos = object5_pos[:, 2]
    # CRITICAL RULE: NEVER use arbitrary thresholds for positions. Small offsets for state transitions are acceptable.
    standing_condition_z = object5_z_pos > (CYLINDER_RADIUS + 0.05) # Object5 is still standing
    # Horizontal distance check for activation
    horizontal_dist_right_palm = torch.norm(object5_pos[:, :2] - right_palm_pos[:, :2], dim=1)
    horizontal_dist_left_palm = torch.norm(object5_pos[:, :2] - left_palm_pos[:, :2], dim=1)
    # Activate if either hand is within 1.0m horizontally
    close_enough_condition_hands = (horizontal_dist_right_palm < 1.0) | (horizontal_dist_left_palm < 1.0)

    activation_condition_hands = standing_condition_z & close_enough_condition_hands

    # CRITICAL RULE: All operations must work with batched environments
    reward = stability_reward + torch.where(activation_condition_hands, hand_proximity_reward, 0.0)

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    This reward penalizes collisions with other cylinder columns (Object1, Object2, Object3, Object4)
    and self-collisions (e.g., robot's feet hitting the target cylinder before it's knocked over).
    It encourages the robot to navigate safely.
    """
    # Get normalizer instance
    # CRITICAL RULE: DO NOT DEFINE or IMPORT RewNormalizer, it's handled by get_normalizer
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    # CRITICAL RULE: ALWAYS access objects using env.scene['ObjectName']
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']
    object5 = env.scene['Object5'] # For self-collision with target object before it falls
    robot = env.scene["robot"]

    # CRITICAL RULE: ALWAYS access robot parts using robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx]
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx]

    # CRITICAL RULE: All operations must work with batched environments
    collision_penalty = torch.zeros_like(pelvis_pos[:, 0]) # Initialize penalty tensor for batch

    # Penalize collisions with other columns (Object1, Object2, Object3, Object4)
    # Check distance from pelvis to other objects. A simple distance check can act as a soft collision avoidance.
    # Clearance threshold based on cylinder radius + approximate half robot width.
    # CRITICAL RULE: NEVER use arbitrary thresholds for positions.
    clearance_threshold_xy = CYLINDER_RADIUS + 0.25 # 0.3 (cylinder radius) + 0.25 (half robot width) = 0.55m

    objects_to_avoid = [object1, object2, object3, object4]

    for obj in objects_to_avoid:
        # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
        obj_pos = obj.data.root_pos_w
        # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
        horizontal_dist = torch.norm(obj_pos[:, :2] - pelvis_pos[:, :2], dim=1)
        # Penalize if too close horizontally. Use a continuous penalty that increases as distance decreases.
        # CRITICAL RULE: Rewards should be continuous. Exponential decay provides a smooth penalty.
        penalty_obj = torch.where(horizontal_dist < clearance_threshold_xy,
                                  -torch.exp(-horizontal_dist / clearance_threshold_xy), # Stronger penalty closer
                                  0.0)
        collision_penalty += penalty_obj

    # Penalize robot's feet hitting Object5 before it's knocked over (e.g., tripping over it)
    # This is active only when Object5 is still standing.
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    object5_pos = object5.data.root_pos_w
    # CRITICAL RULE: NEVER use arbitrary thresholds for positions. Small offsets for state transitions are acceptable.
    standing_condition_object5 = object5_pos[:, 2] > (CYLINDER_RADIUS + 0.05) # Object5 is still standing

    # Distance from feet to Object5's base (root_pos_w is at the base for cylinders)
    # CRITICAL RULE: ALL rewards MUST ONLY use relative distances between objects and robot parts
    dist_right_ankle_x = object5_pos[:, 0] - right_ankle_pos[:, 0]
    dist_right_ankle_y = object5_pos[:, 1] - right_ankle_pos[:, 1]
    dist_right_ankle_z = object5_pos[:, 2] - right_ankle_pos[:, 2] # Z-distance to object base

    dist_left_ankle_x = object5_pos[:, 0] - left_ankle_pos[:, 0]
    dist_left_ankle_y = object5_pos[:, 1] - left_ankle_pos[:, 1]
    dist_left_ankle_z = object5_pos[:, 2] - left_ankle_pos[:, 2]

    # If feet are too close horizontally and too low vertically to Object5's base while it's standing
    # CRITICAL RULE: NEVER use arbitrary thresholds for positions.
    feet_clearance_threshold_xy = CYLINDER_RADIUS + 0.1 # 0.3 + 0.1 = 0.4m
    feet_clearance_threshold_z = 0.2 # Feet should be above object base if not touching, so penalize if Z-diff is small (meaning ankle is low)

    # CRITICAL RULE: Rewards should be continuous. Using a smooth penalty instead of binary -1.0.
    penalty_right_ankle = torch.where(
        standing_condition_object5 &
        (torch.norm(torch.stack([dist_right_ankle_x, dist_right_ankle_y], dim=1), dim=1) < feet_clearance_threshold_xy) &
        (dist_right_ankle_z < feet_clearance_threshold_z), # If ankle is below threshold Z relative to object base
        -torch.exp(-torch.norm(torch.stack([dist_right_ankle_x, dist_right_ankle_y, dist_right_ankle_z], dim=1), dim=1) / (feet_clearance_threshold_xy + feet_clearance_threshold_z)),
        0.0
    )

    penalty_left_ankle = torch.where(
        standing_condition_object5 &
        (torch.norm(torch.stack([dist_left_ankle_x, dist_left_ankle_y], dim=1), dim=1) < feet_clearance_threshold_xy) &
        (dist_left_ankle_z < feet_clearance_threshold_z),
        -torch.exp(-torch.norm(torch.stack([dist_left_ankle_x, dist_left_ankle_y, dist_left_ankle_z], dim=1), dim=1) / (feet_clearance_threshold_xy + feet_clearance_threshold_z)),
        0.0
    )

    # CRITICAL RULE: All operations must work with batched environments
    collision_penalty += penalty_right_ankle + penalty_left_ankle
    reward = collision_penalty

    # CRITICAL RULE: MANDATORY REWARD NORMALIZATION
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # Main reward for knocking over Cylinder Column 5
    MainKnockOverCylinderColumn5Reward = RewTerm(func=main_knock_over_cylinderColumn5_reward, weight=1.0,
                                                 params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for robot stability and hand contact
    RobotStabilityAndContactReward = RewTerm(func=robot_stability_and_contact_reward, weight=0.5,
                                             params={"normalise": True, "normaliser_name": "stability_contact_reward"})

    # Shaping reward for collision avoidance with other objects and self-collision with target
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.2,
                                        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})