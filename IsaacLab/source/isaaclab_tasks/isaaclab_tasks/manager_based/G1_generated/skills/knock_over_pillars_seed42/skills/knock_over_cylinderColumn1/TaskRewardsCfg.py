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


def primary_knock_over_cylinder1_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_reward") -> torch.Tensor:
    """
    Primary reward for the 'knock_over_cylinderColumn1' skill.
    This reward encourages the robot to first approach Cylinder Column 1 (Object1) and then to reduce its Z-position
    until it falls completely. It combines an approach reward with a goal-state reward.
    The approach reward guides the robot towards the cylinder, and once close, the primary focus shifts to knocking it over.
    The reward for the cylinder's Z-position is continuous, encouraging it to fall lower.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object1 and robot's pelvis
    object1 = env.scene['Object1']
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Cylinder dimensions from description: height 2m, radius 0.3m
    # Initial Z-position for a standing cylinder (half of height)
    cylinder_initial_z = 1.0
    # Z-position when cylinder is completely fallen (its radius)
    cylinder_fallen_z = 0.3

    # Calculate XY distance from robot's pelvis to Object1 for approach reward
    # This uses relative distances between robot and object in the XY plane.
    distance_xy_pelvis_to_object1 = torch.norm(
        robot_pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2],
        dim=1
    )
    # Reward for approaching the cylinder (closer is better, hence negative distance)
    approach_reward = -distance_xy_pelvis_to_object1

    # Calculate reward for knocking over the cylinder (lower Z is better)
    # Normalize Object1's Z-position to be between 0 (fallen) and 1 (standing)
    # This ensures the reward is continuous and scaled.
    normalized_object1_z = (object1.data.root_pos_w[:, 2] - cylinder_fallen_z) / (cylinder_initial_z - cylinder_fallen_z)
    normalized_object1_z = torch.clamp(normalized_object1_z, 0.0, 1.0) # Ensure it's within [0, 1]
    # Reward for object falling (1.0 when fallen, 0.0 when standing)
    fall_reward = 1.0 - normalized_object1_z

    # Combine approach and fall rewards based on proximity
    # If robot is far, focus more on approach. If close, focus more on fall.
    approach_threshold = 1.0 # meters, a relative threshold for switching focus
    primary_reward = torch.where(
        distance_xy_pelvis_to_object1 > approach_threshold,
        approach_reward * 0.5, # Weaker approach reward when far
        approach_reward * 0.1 + fall_reward * 0.9 # Stronger fall reward when close
    )

    # Add a small constant positive reward to encourage progress
    primary_reward += 0.01

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def shaping_hand_contact_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_hand_contact") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to use its hands to make contact with and push Object1.
    It rewards reducing the distance between the robot's hands and Object1, specifically in the X and Y dimensions,
    and ensuring the hands are at an appropriate Z-height to push the cylinder.
    This reward is active when the robot is relatively close to the cylinder.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object1 and robot's hands and pelvis
    object1 = env.scene['Object1']
    robot = env.scene["robot"]
    robot_left_hand_idx = robot.body_names.index('left_palm_link')
    robot_right_hand_idx = robot.body_names.index('right_palm_link')
    robot_left_hand_pos = robot.data.body_pos_w[:, robot_left_hand_idx]
    robot_right_hand_pos = robot.data.body_pos_w[:, robot_right_hand_idx]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Cylinder dimensions from description: height 2m, radius 0.3m
    cylinder_height = 2.0
    cylinder_radius = 0.3

    # Target Z-height range for hands to push the cylinder (e.g., middle to upper part)
    # These are relative to the ground, but chosen based on cylinder dimensions.
    target_hand_z_min = cylinder_radius + 0.5 # Above base, e.g., 0.3 + 0.5 = 0.8m
    target_hand_z_max = cylinder_height - 0.5 # Below top, e.g., 2.0 - 0.5 = 1.5m

    # Calculate XY distance from hands to Object1
    # These are relative distances between hand positions and object position in XY.
    distance_xy_left_hand_to_object1 = torch.norm(
        robot_left_hand_pos[:, :2] - object1.data.root_pos_w[:, :2],
        dim=1
    )
    distance_xy_right_hand_to_object1 = torch.norm(
        robot_right_hand_pos[:, :2] - object1.data.root_pos_w[:, :2],
        dim=1
    )
    # Reward for hands being close to the cylinder in XY (closer is better)
    hand_approach_reward = -torch.min(distance_xy_left_hand_to_object1, distance_xy_right_hand_to_object1)

    # Reward for hands being at appropriate Z-height for pushing
    # This uses the absolute Z-position of the hands, which is appropriate for height checks.
    left_hand_z_reward = torch.where(
        (robot_left_hand_pos[:, 2] > target_hand_z_min) & (robot_left_hand_pos[:, 2] < target_hand_z_max),
        0.1, # Small positive reward for being in range
        -0.05 # Small negative penalty for being out of range
    )
    right_hand_z_reward = torch.where(
        (robot_right_hand_pos[:, 2] > target_hand_z_min) & (robot_right_hand_pos[:, 2] < target_hand_z_max),
        0.1,
        -0.05
    )
    hand_z_alignment_reward = left_hand_z_reward + right_hand_z_reward

    # Activation condition: only activate when the robot is relatively close to the cylinder
    # Use pelvis distance as a proxy for overall robot proximity.
    distance_xy_pelvis_to_object1 = torch.norm(
        robot_pelvis_pos[:, :2] - object1.data.root_pos_w[:, :2],
        dim=1
    )
    activation_condition = (distance_xy_pelvis_to_object1 < 1.5) # Activate when within 1.5m, a relative threshold.

    shaping_reward1 = torch.where(
        activation_condition,
        hand_approach_reward * 0.5 + hand_z_alignment_reward,
        0.0
    )

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def shaping_stability_and_collision_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_stability_collision") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain stability and avoid falling over,
    especially after knocking over the cylinder. It also penalizes collision with other standing cylinders
    (Object2, Object3, Object4, Object5) and encourages the robot not to move significantly past the fallen cylinder.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object1 and robot's pelvis
    object1 = env.scene['Object1']
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Pelvis Z-height for stability (a target relative Z-height for standing)
    pelvis_stable_z = 0.7 # meters, a sensible relative height for a bipedal robot's pelvis
    # Penalize deviation from stable Z-height, encouraging stability.
    stability_reward = -torch.abs(robot_pelvis_pos[:, 2] - pelvis_stable_z) * 0.1

    # Reward for not moving too far past Object1 after it has fallen
    # Cylinder dimensions from description: height 2m, radius 0.3m
    cylinder_fallen_z = 0.3 # radius when lying flat
    # Check if Object1 has fallen (Z-position close to its radius)
    object1_fallen_condition = (object1.data.root_pos_w[:, 2] < cylinder_fallen_z + 0.1) # within 0.1m of fallen Z

    # Define an overshoot threshold relative to Object1's X position.
    # This ensures the robot doesn't move too far forward past the fallen cylinder.
    overshoot_threshold = 1.0 # meters past Object1's X, a relative distance.
    overshoot_penalty = torch.where(
        object1_fallen_condition & (robot_pelvis_pos[:, 0] > object1.data.root_pos_w[:, 0] + overshoot_threshold),
        -torch.abs(robot_pelvis_pos[:, 0] - (object1.data.root_pos_w[:, 0] + overshoot_threshold)) * 0.5, # Penalize overshooting
        0.0
    )

    # Collision avoidance with other standing cylinders (Object2, Object3, Object4, Object5)
    # Cylinder radius is 0.3m. Add a small buffer for robot body clearance.
    collision_distance_threshold = 0.8 # Object radius + robot body clearance, a relative distance.

    collision_penalty = torch.zeros_like(robot_pelvis_pos[:, 0]) # Initialize penalty tensor
    for i in range(2, 6): # Iterate through Object2 to Object5
        other_object = env.scene[f'Object{i}']
        # Only penalize if the other object is still standing (Z-position above a certain threshold)
        other_object_standing_condition = (other_object.data.root_pos_w[:, 2] > cylinder_fallen_z + 0.5) # Check if it's still upright

        # Calculate XY distance from robot's pelvis to the other object
        distance_xy_pelvis_to_other_object = torch.norm(
            robot_pelvis_pos[:, :2] - other_object.data.root_pos_w[:, :2],
            dim=1
        )
        # Penalize if too close to a standing other object
        collision_penalty += torch.where(
            other_object_standing_condition & (distance_xy_pelvis_to_other_object < collision_distance_threshold),
            -torch.abs(distance_xy_pelvis_to_other_object - collision_distance_threshold) * 0.5, # Penalize getting too close
            0.0
        )

    shaping_reward2 = stability_reward + overshoot_penalty + collision_penalty

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # Primary reward for knocking over Cylinder Column 1
    PrimaryKnockOverCylinder1Reward = RewTerm(
        func=primary_knock_over_cylinder1_reward,
        weight=1.0, # High weight as it's the main goal
        params={"normalise": True, "normaliser_name": "primary_reward"}
    )

    # Shaping reward for encouraging hand contact and proper Z-height for pushing
    ShapingHandContactReward = RewTerm(
        func=shaping_hand_contact_reward,
        weight=0.4, # Moderate weight to guide interaction
        params={"normalise": True, "normaliser_name": "shaping_hand_contact"}
    )

    # Shaping reward for maintaining stability, avoiding overshoot, and preventing collisions with other cylinders
    ShapingStabilityAndCollisionReward = RewTerm(
        func=shaping_stability_and_collision_reward,
        weight=0.6, # Moderate weight for safety and readiness for next skill
        params={"normalise": True, "normaliser_name": "shaping_stability_collision"}
    )