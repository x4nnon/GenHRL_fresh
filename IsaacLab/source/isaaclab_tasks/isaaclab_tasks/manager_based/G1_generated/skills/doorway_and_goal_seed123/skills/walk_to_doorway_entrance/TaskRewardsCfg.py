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


def walk_to_doorway_entrance_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_doorway_entrance_primary_reward") -> torch.Tensor:
    """
    Primary reward for the walk_to_doorway_entrance skill.
    Encourages the robot's pelvis to reach a position just before the doorway,
    centered between Object1 (Heavy Cube Wall 1) and Object2 (Heavy Cube Wall 2).
    Also encourages maintaining a stable pelvis height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1']  # Heavy Cube (Wall 1)
    object2 = env.scene['Object2']  # Heavy Cube (Wall 2)

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"] # Access robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Calculate doorway center x-position using relative distances
    # The doorway is formed by Object1 and Object2, so its center x is the average of their x-positions.
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0

    # Determine the target y-position just before the doorway.
    # Assuming the doorway starts at the y-coordinate of Object1/Object2.
    # A small offset (0.2m) is used to position the robot "just before" the doorway.
    target_y_pos = object1.data.root_pos_w[:, 1] - 0.2

    # Calculate distances to the target x and y positions.
    # Using absolute distances for continuous positive reward as distance decreases.
    distance_x = torch.abs(pelvis_pos_x - doorway_center_x)
    distance_y = torch.abs(pelvis_pos_y - target_y_pos)

    # Primary reward: Negative sum of distances to encourage approaching the target.
    reward_primary = - (distance_x + distance_y)

    # Encourage maintaining a stable pelvis height (e.g., around 0.7m for standing).
    # This is a small penalty for deviation from a reasonable standing height.
    pelvis_target_z = 0.7
    reward_pelvis_z = -torch.abs(pelvis_pos_z - pelvis_target_z) * 0.1

    # Combine rewards
    reward = reward_primary + reward_pelvis_z

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_doorway_entrance_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_doorway_entrance_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the doorway walls (Object1 and Object2).
    Penalizes the robot if its feet or pelvis get too close to the inner edges of the walls in the x-direction.
    This reward is active only when the robot is approaching or at the doorway entrance.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1']  # Heavy Cube (Wall 1)
    object2 = env.scene['Object2']  # Heavy Cube (Wall 2)

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"] # Access robot object
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    pelvis_idx = robot.body_names.index('pelvis')

    left_foot_pos_x = robot.data.body_pos_w[:, left_foot_idx, 0]
    right_foot_pos_x = robot.data.body_pos_w[:, right_foot_idx, 0]
    pelvis_pos_x = robot.data.body_pos_w[:, pelvis_idx, 0]
    pelvis_pos_y = robot.data.body_pos_w[:, pelvis_idx, 1]

    # Wall dimensions: Hardcoded from the environment setup description.
    # "x of 0.5" for the walls.
    wall_x_dim = 0.5

    # Calculate the inner edges of the doorway relative to wall positions.
    # Assuming Object1 is to the left (smaller x) and Object2 to the right (larger x).
    object1_inner_x = object1.data.root_pos_w[:, 0] + (wall_x_dim / 2.0)  # Right edge of Object1
    object2_inner_x = object2.data.root_pos_w[:, 0] - (wall_x_dim / 2.0)  # Left edge of Object2

    # Define a safe zone offset from the inner edges.
    safe_zone_offset = 0.1  # Distance from wall inner edge to start penalizing

    # Re-use target_y_pos from primary reward for activation condition.
    target_y_pos = object1.data.root_pos_w[:, 1] - 0.2

    # Calculate distances of robot parts to the inner wall edges.
    # These are relative distances.
    dist_left_foot_wall1 = left_foot_pos_x - object1_inner_x
    dist_pelvis_wall1 = pelvis_pos_x - object1_inner_x

    dist_right_foot_wall2 = object2_inner_x - right_foot_pos_x
    dist_pelvis_wall2 = object2_inner_x - pelvis_pos_x

    # Penalize if left foot or pelvis is too far left (colliding with Object1).
    # Using an exponential penalty for continuous and strong punishment when very close.
    penalty_left = torch.where(
        (dist_left_foot_wall1 < safe_zone_offset) | (dist_pelvis_wall1 < safe_zone_offset),
        -torch.exp(-dist_left_foot_wall1 / 0.1) - torch.exp(-dist_pelvis_wall1 / 0.1),
        torch.zeros_like(dist_left_foot_wall1)
    )

    # Penalize if right foot or pelvis is too far right (colliding with Object2).
    penalty_right = torch.where(
        (dist_right_foot_wall2 < safe_zone_offset) | (dist_pelvis_wall2 < safe_zone_offset),
        -torch.exp(-dist_right_foot_wall2 / 0.1) - torch.exp(-dist_pelvis_wall2 / 0.1),
        torch.zeros_like(dist_right_foot_wall2)
    )

    # Only apply collision avoidance when robot is approaching the doorway (not significantly past it).
    # This ensures the reward is relevant to the current skill's objective.
    # A small buffer (+0.5m) is added to target_y_pos to keep the reward active until the robot is clearly past the immediate collision zone.
    activation_condition = (pelvis_pos_y < target_y_pos + 0.5)

    # Apply penalties only when the activation condition is met.
    reward = torch.where(activation_condition, penalty_left + penalty_right, torch.zeros_like(penalty_left))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_to_doorway_entrance_forward_progress_and_stopping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_doorway_entrance_forward_progress_and_stopping_reward") -> torch.Tensor:
    """
    Shaping reward for forward progress and precise stopping at the doorway entrance.
    Encourages positive y-velocity when far from the target and penalizes high y-velocity
    as the robot approaches and reaches the target y-position, promoting precise stopping.
    Also penalizes overshooting the target y-position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects (only Object1 needed for target_y_pos)
    object1 = env.scene['Object1']  # Heavy Cube (Wall 1)

    # Access required robot part(s) and their velocities
    robot = env.scene["robot"] # Access robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos_y = robot.data.body_pos_w[:, pelvis_idx, 1]
    pelvis_vel_y = robot.data.body_vel_w[:, pelvis_idx, 1]  # Accessing y-velocity of pelvis

    # Target y-position (re-used from primary reward)
    target_y_pos = object1.data.root_pos_w[:, 1] - 0.2

    # Define thresholds for different phases of movement.
    approach_threshold = 1.5  # Start slowing down when 1.5m from target y
    stop_threshold = 0.1    # When within 0.1m of target y, penalize high velocity

    # Reward for moving forward (positive y-velocity) when far from the target.
    # This encourages initial movement towards the doorway.
    reward_forward_progress = torch.where(
        pelvis_pos_y < target_y_pos - approach_threshold,
        pelvis_vel_y,  # Reward positive y velocity
        torch.zeros_like(pelvis_vel_y)
    )

    # Penalty for high velocity when close to the target.
    # This encourages the robot to slow down and stop precisely.
    reward_stopping = torch.where(
        (pelvis_pos_y >= target_y_pos - approach_threshold) & (pelvis_pos_y <= target_y_pos + stop_threshold),
        -torch.abs(pelvis_vel_y) * 2.0,  # Penalize absolute velocity more strongly
        torch.zeros_like(pelvis_vel_y)
    )

    # Penalty for overshooting the target y-position significantly.
    # This is crucial for preparing for the next skill.
    overshoot_penalty_threshold = 0.5  # If pelvis goes 0.5m past target y
    reward_overshoot_penalty = torch.where(
        pelvis_pos_y > target_y_pos + overshoot_penalty_threshold,
        -(pelvis_pos_y - (target_y_pos + overshoot_penalty_threshold)) * 5.0,  # Linear penalty for overshooting
        torch.zeros_like(pelvis_pos_y)
    )

    # Combine rewards
    reward = reward_forward_progress + reward_stopping + reward_overshoot_penalty

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
    # Primary reward for reaching the doorway entrance
    walk_to_doorway_entrance_primary_reward = RewTerm(
        func=walk_to_doorway_entrance_primary_reward, weight=1.0,
        params={"normalise": True, "normaliser_name": "walk_to_doorway_entrance_primary_reward"}
    )

    # Shaping reward for collision avoidance with walls
    walk_to_doorway_entrance_collision_avoidance_reward = RewTerm(
        func=walk_to_doorway_entrance_collision_avoidance_reward, weight=0.6,
        params={"normalise": True, "normaliser_name": "walk_to_doorway_entrance_collision_avoidance_reward"}
    )

    # Shaping reward for forward progress and precise stopping
    walk_to_doorway_entrance_forward_progress_and_stopping_reward = RewTerm(
        func=walk_to_doorway_entrance_forward_progress_and_stopping_reward, weight=0.4,
        params={"normalise": True, "normaliser_name": "walk_to_doorway_entrance_forward_progress_and_stopping_reward"}
    )