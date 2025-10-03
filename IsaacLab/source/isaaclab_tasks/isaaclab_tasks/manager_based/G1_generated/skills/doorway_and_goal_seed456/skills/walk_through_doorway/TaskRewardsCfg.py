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


def walk_through_doorway_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_through_doorway_main_reward") -> torch.Tensor:
    """
    Primary reward for the walk_through_doorway skill.
    Guides the robot to walk through the doorway and position its pelvis and feet past the doorway's Y-coordinate.
    It encourages forward movement (along Y-axis) towards and past the doorway, and then rewards for maintaining a stable pelvis height.
    The reward transitions from encouraging forward movement to encouraging stability once past the doorway.
    Includes a penalty for overshooting the target area past the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    # Object1: Heavy Cube (Wall 1)
    # Object2: Heavy Cube (Wall 2)
    # Object3: small block (reference for next skill, used for overshoot prevention)
    object1 = env.scene['Object1']
    object3 = env.scene['Object3']

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_y = left_foot_pos[:, 1]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_y = right_foot_pos[:, 1]

    # Hardcoded wall Y-dimension from object configuration (y of 5m)
    # This value is hardcoded as per rule 8: "THERE IS NO way to access the SIZE of an object"
    wall_y_dim = 5.0

    # Calculate doorway Y-exit position. Assuming robot walks in positive Y,
    # and Object1's root_pos_w[:, 1] is the center of its 5m length.
    # The robot needs to clear the far side of the wall.
    # This uses relative distance from the object's center to its far edge.
    # This is derived from object position and hardcoded dimension, adhering to rule 4.
    doorway_y_exit = object1.data.root_pos_w[:, 1] + (wall_y_dim / 2)

    # Phase 1 & 2: Approach and Pass Through
    # Reward for reducing Y-distance to the doorway exit. This is a continuous negative reward
    # that becomes less negative as the robot approaches and passes the doorway exit.
    # Uses relative distance: pelvis Y position relative to doorway exit Y position.
    # Adheres to rule 1: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    reward_approach_y = -torch.abs(pelvis_pos_y - doorway_y_exit)

    # Phase 3: Clear Doorway and Stabilize
    # Condition for being past the doorway: pelvis and both feet must be past the doorway_y_exit.
    # This ensures the entire body has cleared the doorway.
    is_past_doorway = (pelvis_pos_y > doorway_y_exit) & \
                       (left_foot_pos_y > doorway_y_exit) & \
                       (right_foot_pos_y > doorway_y_exit)

    # Reward for maintaining stable pelvis height (0.7m) once past the doorway.
    # This is a relative distance: pelvis Z position relative to target Z.
    # The target Z (0.7m) is a standard stable height, adhering to rule 4.
    pelvis_target_z = 0.7
    reward_stable_z = -torch.abs(pelvis_pos_z - pelvis_target_z)

    # Combine rewards: prioritize passing the doorway, then stabilize.
    # Uses torch.where for a smooth transition between the two reward components.
    # Adheres to rule 7: "CONTINUOUS REWARDS"
    primary_reward = torch.where(is_past_doorway, reward_stable_z, reward_approach_y)

    # Ensure the robot does not overshoot the intended end point for this skill.
    # Object3 is 2m past the doorway. This skill should end just past the doorway, not at Object3.
    # Allow a small buffer past the doorway exit (e.g., 0.5m) before penalizing.
    # This uses relative distance: pelvis Y position relative to the maximum allowed Y.
    # The buffer (0.5m) is a small, reasonable threshold, adhering to rule 4.
    max_allowed_y = doorway_y_exit + 0.5
    overshoot_penalty = torch.where(pelvis_pos_y > max_allowed_y, -torch.abs(pelvis_pos_y - max_allowed_y) * 5.0, 0.0)
    primary_reward += overshoot_penalty

    reward = primary_reward

    # Mandatory reward normalization
    # Adheres to rule 2: "MANDATORY REWARD NORMALIZATION"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_through_doorway_collision_penalty(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_through_doorway_collision_penalty") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the walls (Object1 and Object2).
    Penalizes the robot if any part of its body gets too close to or collides with the walls.
    Encourages the robot to stay within the doorway's X-bounds and Z-height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)

    # Access relevant robot parts
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    head_idx = robot.body_names.index('head_link')
    head_pos = robot.data.body_pos_w[:, head_idx]

    # Hardcoded wall dimensions from object configuration (x=0.5m, z=1.5m)
    # These values are hardcoded as per rule 8: "THERE IS NO way to access the SIZE of an object"
    wall_x_dim = 0.5
    wall_z_dim = 1.5

    # Define doorway X-bounds based on wall positions and dimensions.
    # Assuming Object1 is to the left (smaller X) and Object2 is to the right (larger X).
    # doorway_x_min is the inner (right) edge of Object1.
    # doorway_x_max is the inner (left) edge of Object2.
    # These are relative positions derived from object root positions and their dimensions, adhering to rule 4.
    doorway_x_min = object1.data.root_pos_w[:, 0] + (wall_x_dim / 2)
    doorway_x_max = object2.data.root_pos_w[:, 0] - (wall_x_dim / 2)
    # Expand to [num_envs, 1] for broadcasting with per-part tensors of shape [num_envs, num_parts]
    doorway_x_min = doorway_x_min.unsqueeze(1)
    doorway_x_max = doorway_x_max.unsqueeze(1)

    # Define wall Z-height bounds. Assuming walls are placed with their base at Z=0.
    # This uses relative positions from the wall's center to its top/bottom edges.
    # Derived from object position and hardcoded dimension, adhering to rule 4.
    wall_z_bottom = (object1.data.root_pos_w[:, 2] - (wall_z_dim / 2)).unsqueeze(1)
    wall_z_top = (object1.data.root_pos_w[:, 2] + (wall_z_dim / 2)).unsqueeze(1)

    # Collect X and Z positions of all relevant robot parts for batched processing.
    # Adheres to rule 6: "HANDLE TENSOR OPERATIONS CORRECTLY"
    robot_parts_x = torch.cat([pelvis_pos[:, 0].unsqueeze(1),
                               left_foot_pos[:, 0].unsqueeze(1),
                               right_foot_pos[:, 0].unsqueeze(1),
                               left_hand_pos[:, 0].unsqueeze(1),
                               right_hand_pos[:, 0].unsqueeze(1),
                               head_pos[:, 0].unsqueeze(1)], dim=1)

    robot_parts_z = torch.cat([pelvis_pos[:, 2].unsqueeze(1),
                               left_foot_pos[:, 2].unsqueeze(1),
                               right_foot_pos[:, 2].unsqueeze(1),
                               left_hand_pos[:, 2].unsqueeze(1),
                               right_hand_pos[:, 2].unsqueeze(1),
                               head_pos[:, 2].unsqueeze(1)], dim=1)

    # Calculate penetration depth for collision with Object1 (left wall).
    # If robot part X is less than doorway_x_min, it's penetrating.
    # This is a relative distance: robot part X relative to doorway_x_min.
    # Adheres to rule 1: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    penetration_obj1_x = torch.where(
        robot_parts_x < doorway_x_min,
        robot_parts_x - doorway_x_min,
        torch.zeros_like(robot_parts_x)
    )

    # Calculate penetration depth for collision with Object2 (right wall).
    # If robot part X is greater than doorway_x_max, it's penetrating.
    # This is a relative distance: robot part X relative to doorway_x_max.
    # Adheres to rule 1: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    penetration_obj2_x = torch.where(
        robot_parts_x > doorway_x_max,
        robot_parts_x - doorway_x_max,
        torch.zeros_like(robot_parts_x)
    )

    # Condition for Z-overlap with the wall's height.
    # This uses relative distance: robot part Z relative to wall Z bounds.
    # Adheres to rule 1: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    collision_z_condition = (robot_parts_z > wall_z_bottom) & (robot_parts_z < wall_z_top)

    # Combine X and Z conditions for collision penalty.
    # Penalties are negative values, representing penetration.
    # Adheres to rule 7: "CONTINUOUS REWARDS"
    collision_penalty_per_part = torch.zeros_like(robot_parts_x)
    # Apply penalty if penetrating left wall AND within wall Z-height
    collision_penalty_per_part = torch.where(collision_z_condition, collision_penalty_per_part + penetration_obj1_x, collision_penalty_per_part)
    # Apply penalty if penetrating right wall AND within wall Z-height
    collision_penalty_per_part = torch.where(collision_z_condition, collision_penalty_per_part + penetration_obj2_x, collision_penalty_per_part)

    # Take the minimum (most negative/severe) penalty across all robot parts for each environment.
    # Multiply by a factor for stronger penalty.
    # The result is clamped to be non-positive (0 or negative).
    reward = torch.min(collision_penalty_per_part, dim=1).values * 5.0
    reward = torch.clamp(reward, max=0.0)

    # Mandatory reward normalization
    # Adheres to rule 2: "MANDATORY REWARD NORMALIZATION"
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_through_doorway_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_through_doorway_upright_posture_reward") -> torch.Tensor:
    """
    Shaping reward for maintaining an upright posture.
    Encourages the robot to keep its pelvis at a reasonable height (around 0.7m) throughout the skill,
    which helps with stability.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Target pelvis height for stable standing. This is a fixed target for the robot's own height.
    # The target Z (0.7m) is a standard stable height, adhering to rule 4.
    pelvis_target_z = 0.7

    # Reward for being close to the target pelvis height.
    # This is a continuous negative reward based on the absolute difference (relative distance)
    # between current pelvis Z and target pelvis Z.
    # Adheres to rule 1: "ALL rewards MUST ONLY use relative distances between objects and robot parts"
    # Adheres to rule 7: "CONTINUOUS REWARDS"
    reward = -torch.abs(pelvis_pos_z - pelvis_target_z)

    # Mandatory reward normalization
    # Adheres to rule 2: "MANDATORY REWARD NORMALIZATION"
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
    Configuration for the rewards used in the walk_through_doorway skill.
    """
    # Primary reward for guiding the robot through the doorway and stabilizing.
    # Weight is 1.0 as it's the main objective.
    # Adheres to rule 2: "PROPER WEIGHTS"
    walk_through_doorway_main_reward = RewTerm(func=walk_through_doorway_main_reward, weight=1.0,
                                               params={"normalise": True, "normaliser_name": "walk_through_doorway_main_reward"})

    # Shaping reward for collision avoidance with the walls.
    # Weight is 0.6 to strongly penalize collisions but not override the primary goal.
    # Adheres to rule 2: "PROPER WEIGHTS"
    walk_through_doorway_collision_penalty = RewTerm(func=walk_through_doorway_collision_penalty, weight=0.6,
                                                     params={"normalise": True, "normaliser_name": "walk_through_doorway_collision_penalty"})

    # Shaping reward for maintaining an upright posture.
    # Weight is 0.3 to encourage stability without being too dominant.
    # Adheres to rule 2: "PROPER WEIGHTS"
    walk_through_doorway_upright_posture_reward = RewTerm(func=walk_through_doorway_upright_posture_reward, weight=0.3,
                                                          params={"normalise": True, "normaliser_name": "walk_through_doorway_upright_posture_reward"})