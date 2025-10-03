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


def primary_push_cube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_push_cube_reward") -> torch.Tensor:
    """
    Primary reward for pushing Object1 (Cube for robot to push) onto Object4 (Platform for cubes)
    and ensuring it is stable and fully contained.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    # Hardcoded dimensions from the object configuration description
    # Object1: Cube for robot to push, size = [0.5, 0.5, 0.5]
    object1_half_size_x = 0.25
    object1_half_size_y = 0.25
    object1_half_size_z = 0.25
    # Object4: Platform for cubes, size = [2.0, 2.0, 0.001]
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0
    object4_height = 0.001

    # Calculate distances for Object1 relative to Object4's center
    # All rewards use relative distances between objects.
    dist_obj1_obj4_x = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    dist_obj1_obj4_y = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]

    # Reward for Object1 being horizontally centered on Object4
    # Max reward when x and y distances are 0, continuous negative reward for deviation.
    reward_center_x = -torch.abs(dist_obj1_obj4_x)
    reward_center_y = -torch.abs(dist_obj1_obj4_y)

    # Reward for Object1 being vertically on Object4
    # Object1's bottom surface should be at or slightly above Object4's top surface.
    # Target z for Object1's center: Object4's top surface + Object1's half height.
    target_obj1_z = object4.data.root_pos_w[:, 2] + object4_height / 2.0 + object1_half_size_z
    # Continuous negative reward for deviation from target Z.
    reward_on_platform_z = -torch.abs(object1.data.root_pos_w[:, 2] - target_obj1_z)

    # Condition for Object1 being fully within the horizontal bounds of Object4
    # This ensures the entire cube is on the platform, not just its center.
    is_on_platform_x = (torch.abs(dist_obj1_obj4_x) <= (object4_half_size_x - object1_half_size_x))
    is_on_platform_y = (torch.abs(dist_obj1_obj4_y) <= (object4_half_size_y - object1_half_size_y))
    is_on_platform_horizontal = is_on_platform_x & is_on_platform_y

    # Reward for Object1 stability (low velocity) when on platform
    # Penalize high velocity only when the object is horizontally on the platform.
    object1_vel_norm = torch.norm(object1.data.root_vel_w, dim=-1)
    reward_stability = torch.where(is_on_platform_horizontal, -object1_vel_norm, 0.0)

    # Combine primary rewards
    reward = reward_center_x + reward_center_y + reward_on_platform_z + reward_stability

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_robot_contact_and_collision_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_contact_collision_reward") -> torch.Tensor:
    """
    Shaping reward to encourage robot hands to be close to Object1 for pushing,
    and discourage other robot parts from colliding with Object1 or Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]
    left_palm_idx = robot.body_names.index('left_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    feet_left_idx = robot.body_names.index('left_ankle_roll_link')
    feet_left_pos = robot.data.body_pos_w[:, feet_left_idx]
    feet_right_idx = robot.body_names.index('right_ankle_roll_link')
    feet_right_pos = robot.data.body_pos_w[:, feet_right_idx]

    # Hardcoded dimensions from the object configuration description
    # Object1: Cube for robot to push, size = [0.5, 0.5, 0.5]
    object1_half_size_x = 0.25
    object1_half_size_y = 0.25
    object1_half_size_z = 0.25
    # Object4: Platform for cubes, size = [2.0, 2.0, 0.001]
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0
    object4_height = 0.001

    # Distance from right palm to Object1 (relative distance)
    dist_right_palm_obj1 = object1.data.root_pos_w - right_palm_pos
    reward_right_palm_to_obj1 = -torch.norm(dist_right_palm_obj1, dim=-1)

    # Distance from left palm to Object1 (relative distance)
    dist_left_palm_obj1 = object1.data.root_pos_w - left_palm_pos
    reward_left_palm_to_obj1 = -torch.norm(dist_left_palm_obj1, dim=-1)

    # Encourage one hand to be close to Object1 (max of the two rewards)
    reward_hands_to_obj1 = torch.max(reward_right_palm_to_obj1, reward_left_palm_to_obj1)

    # Collision avoidance for robot body parts with Object1 (e.g., pelvis)
    # Only penalize if pelvis is too close to Object1, not if hands are pushing.
    # Using relative distances for collision checks.
    pelvis_obj1_dist_x = torch.abs(object1.data.root_pos_w[:, 0] - pelvis_pos[:, 0])
    pelvis_obj1_dist_y = torch.abs(object1.data.root_pos_w[:, 1] - pelvis_pos[:, 1])
    pelvis_obj1_dist_z = torch.abs(object1.data.root_pos_w[:, 2] - pelvis_pos[:, 2])

    # Define a safe distance threshold for pelvis from Object1 (Object1 half-size + clearance)
    pelvis_obj1_safe_dist_x = object1_half_size_x + 0.1 # 0.1m clearance
    pelvis_obj1_safe_dist_y = object1_half_size_y + 0.1
    pelvis_obj1_safe_dist_z = object1_half_size_z + 0.1

    # Penalize if pelvis is too close to Object1 (continuous penalty based on overlap)
    # This is a conditional reward based on proximity.
    pelvis_obj1_collision_penalty = torch.where(
        (pelvis_obj1_dist_x < pelvis_obj1_safe_dist_x) &
        (pelvis_obj1_dist_y < pelvis_obj1_safe_dist_y) &
        (pelvis_obj1_dist_z < pelvis_obj1_safe_dist_z),
        -1.0 * (pelvis_obj1_safe_dist_x - pelvis_obj1_dist_x) - 1.0 * (pelvis_obj1_safe_dist_y - pelvis_obj1_dist_y) - 1.0 * (pelvis_obj1_safe_dist_z - pelvis_obj1_dist_z), # Linear penalty for overlap
        0.0
    )

    # Collision avoidance for robot body parts with Object4 (e.g., feet)
    # Robot should not be on the platform itself.
    # Check if feet are on Object4 using relative positions and platform bounds.
    is_left_foot_on_platform_x = (torch.abs(feet_left_pos[:, 0] - object4.data.root_pos_w[:, 0]) < object4_half_size_x)
    is_left_foot_on_platform_y = (torch.abs(feet_left_pos[:, 1] - object4.data.root_pos_w[:, 1]) < object4_half_size_y)
    # Z-check: foot is above platform's top surface, with a small tolerance.
    is_left_foot_on_platform_z = (feet_left_pos[:, 2] > object4.data.root_pos_w[:, 2] + object4_height / 2.0 - 0.05)
    is_left_foot_on_platform = is_left_foot_on_platform_x & is_left_foot_on_platform_y & is_left_foot_on_platform_z

    is_right_foot_on_platform_x = (torch.abs(feet_right_pos[:, 0] - object4.data.root_pos_w[:, 0]) < object4_half_size_x)
    is_right_foot_on_platform_y = (torch.abs(feet_right_pos[:, 1] - object4.data.root_pos_w[:, 1]) < object4_half_size_y)
    is_right_foot_on_platform_z = (feet_right_pos[:, 2] > object4.data.root_pos_w[:, 2] + object4_height / 2.0 - 0.05)
    is_right_foot_on_platform = is_right_foot_on_platform_x & is_right_foot_on_platform_y & is_right_foot_on_platform_z

    # Penalize if either foot is on the platform.
    robot_on_platform_penalty = torch.where(
        is_left_foot_on_platform | is_right_foot_on_platform,
        -1.0, # Large negative reward for robot being on platform
        0.0
    )

    reward = reward_hands_to_obj1 + pelvis_obj1_collision_penalty + robot_on_platform_penalty

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_robot_posture_and_overshoot_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_posture_overshoot_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable standing posture (pelvis at a target height)
    and ensures Object1 is not pushed too far past the center of Object4, preparing for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Hardcoded dimensions from the object configuration description
    # Object1: Cube for robot to push, size = [0.5, 0.5, 0.5]
    object1_half_size_x = 0.25
    object1_half_size_y = 0.25
    # Object4: Platform for cubes, size = [2.0, 2.0, 0.001]
    object4_half_size_x = 1.0
    object4_half_size_y = 1.0

    # Reward for maintaining stable pelvis height
    # Target pelvis z-position is an absolute value, which is allowed for height.
    target_pelvis_z = 0.7
    # Continuous negative reward for deviation from target pelvis height.
    reward_pelvis_height = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Reward for not pushing Object1 too far past Object4's center (in x-direction)
    # This helps prevent overshooting and keeps the robot in a good position for the next cube.
    # Calculate relative distance in x for Object1 and Object4.
    dist_obj1_obj4_x = object1.data.root_pos_w[:, 0] - object4.data.root_pos_w[:, 0]
    dist_obj1_obj4_y = object1.data.root_pos_w[:, 1] - object4.data.root_pos_w[:, 1]

    # Check if Object1 is horizontally on the platform (same condition as primary reward)
    is_on_platform_x = (torch.abs(dist_obj1_obj4_x) <= (object4_half_size_x - object1_half_size_x))
    is_on_platform_y = (torch.abs(dist_obj1_obj4_y) <= (object4_half_size_y - object1_half_size_y))
    is_on_platform_horizontal = is_on_platform_x & is_on_platform_y

    # Penalize if Object1 is pushed too far in the x-direction beyond the center of Object4.
    # A small buffer (e.g., 0.1m) beyond the center is acceptable.
    max_acceptable_x_offset = 0.1 # 0.1m past center of platform
    # Continuous linear penalty for overshooting, active only when on platform and overshooting.
    reward_overshoot_x = torch.where(
        (dist_obj1_obj4_x > max_acceptable_x_offset) & is_on_platform_horizontal,
        -(dist_obj1_obj4_x - max_acceptable_x_offset), # Linear penalty for overshooting
        0.0
    )

    reward = reward_pelvis_height + reward_overshoot_x

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
    Reward terms for the push_cube_for_robot_to_push_1_onto_platform_for_cubes skill.
    """
    # Primary reward for positioning and stabilizing Object1 on Object4.
    primary_push_cube_reward = RewTerm(func=primary_push_cube_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "primary_push_cube_reward"})

    # Shaping reward for robot contact with Object1 and collision avoidance with Object1/Object4.
    shaping_robot_contact_and_collision_reward = RewTerm(func=shaping_robot_contact_and_collision_reward, weight=0.6,
                                                         params={"normalise": True, "normaliser_name": "shaping_contact_collision_reward"})

    # Shaping reward for maintaining robot posture and preventing Object1 from overshooting.
    shaping_robot_posture_and_overshoot_reward = RewTerm(func=shaping_robot_posture_and_overshoot_reward, weight=0.4,
                                                          params={"normalise": True, "normaliser_name": "shaping_posture_overshoot_reward"})
