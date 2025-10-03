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


def walk_clear_of_doorway_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_clear_of_doorway_primary_reward") -> torch.Tensor:
    """
    Primary reward for the walk_clear_of_doorway skill.
    Encourages the robot to move its pelvis forward (positive y-direction) past the doorway,
    while also encouraging it to stay centered in the x-axis within the doorway gap.
    It also includes a component to ensure the robot does not overshoot the target area for the next skill (Object3).
    The reward is highest when the robot's pelvis is just past the doorway and centered,
    and decreases if it moves too far past the doorway or deviates too much in x.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects and robot parts using approved patterns
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    object3 = env.scene['Object3'] # Small Block
    robot = env.scene["robot"] # Access robot object

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Hardcoded dimensions from task description for walls: x=0.5m (thickness), y=5m (length), z=1.5m (height).
    wall_length_y = 5.0
    wall_thickness_x = 0.5

    # Calculate doorway y-coordinates based on Object1's y-center and wall length.
    # Assuming Object1.data.root_pos_w[:, 1] is the y-center of the 5m long wall.
    # Doorway starts at y_center - half_length and ends at y_center + half_length.
    doorway_y_start = object1.data.root_pos_w[:, 1] - (wall_length_y / 2.0)
    doorway_y_exit = object1.data.root_pos_w[:, 1] + (wall_length_y / 2.0)

    # Calculate doorway x-center. Assuming Object1 is left, Object2 is right.
    # The inner edge of Object1 is Object1.data.root_pos_w[:, 0] + (wall_thickness_x / 2.0)
    # The inner edge of Object2 is Object2.data.root_pos_w[:, 0] - (wall_thickness_x / 2.0)
    # The center of the doorway gap is the average of these inner edges.
    doorway_x_center = ( (object1.data.root_pos_w[:, 0] + (wall_thickness_x / 2.0)) + \
                         (object2.data.root_pos_w[:, 0] - (wall_thickness_x / 2.0)) ) / 2.0

    # Target y-position: slightly past the doorway, but before Object3.
    # Object3 is 2m past the doorway in y.
    # target_y_min: A small buffer past the doorway exit to ensure clearance.
    target_y_min = doorway_y_exit + 0.2
    # target_y_max: A buffer before Object3 to prevent overshooting.
    target_y_max = object3.data.root_pos_w[:, 1] - 0.5

    # Reward for moving past the doorway in y and staying within target y range.
    # Uses a Gaussian-like function centered between target_y_min and target_y_max.
    # Reward is highest when pelvis_pos_y is at the ideal y-position.
    ideal_y_pos = (target_y_min + target_y_max) / 2.0
    y_progress_reward = torch.exp(-torch.abs(pelvis_pos_y - ideal_y_pos) * 2.0) # Scale factor 2.0 for sensitivity

    # Reward for staying centered in x.
    # Uses a Gaussian-like function centered at doorway_x_center.
    x_center_reward = torch.exp(-torch.abs(pelvis_pos_x - doorway_x_center) * 5.0) # Scale factor 5.0 for sensitivity

    # Combine y-progress and x-centering rewards.
    reward = y_progress_reward + x_center_reward

    # Encourage stable standing posture (pelvis z-height around 0.7m) when past the doorway.
    pelvis_target_z = 0.7
    z_stability_reward = torch.exp(-torch.abs(pelvis_pos_z - pelvis_target_z) * 5.0) # Scale factor 5.0 for sensitivity

    # Only apply z_stability_reward when robot's pelvis is past the doorway's y-exit.
    condition_past_doorway = (pelvis_pos_y > doorway_y_exit)
    reward += torch.where(condition_past_doorway, z_stability_reward, 0.0)

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_clear_of_doorway_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_clear_of_doorway_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with walls (Object1 and Object2).
    Penalizes the robot for getting too close to the walls in the x-direction, especially when passing through the doorway.
    Applies to relevant body parts (pelvis, hands, feet, head).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects and robot parts
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    object2 = env.scene['Object2'] # Heavy Cube (Wall 2)
    robot = env.scene["robot"]

    # Hardcoded dimensions from task description for walls: x=0.5m (thickness), y=5m (length), z=1.5m (height).
    wall_length_y = 5.0
    wall_thickness_x = 0.5

    # Define the y-extent of the doorway where collision is relevant.
    # Assuming Object1.data.root_pos_w[:, 1] is the y-center of the 5m long wall.
    doorway_y_start = object1.data.root_pos_w[:, 1] - (wall_length_y / 2.0)
    doorway_y_end = object1.data.root_pos_w[:, 1] + (wall_length_y / 2.0)

    # Define inner x-edges of the walls.
    # Inner edge of Object1 (left wall) is its center + half thickness.
    wall1_inner_x = object1.data.root_pos_w[:, 0] + (wall_thickness_x / 2.0)
    # Inner edge of Object2 (right wall) is its center - half thickness.
    wall2_inner_x = object2.data.root_pos_w[:, 0] - (wall_thickness_x / 2.0)

    # Relevant robot body parts to check for collision.
    body_parts_to_check = ['pelvis', 'left_palm_link', 'right_palm_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'head_link']
    collision_reward = torch.zeros(env.num_envs, device=env.device)
    clearance_threshold = 0.1 # Robot part should maintain at least 0.1m distance from wall's inner edge

    for part_name in body_parts_to_check:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx]
        part_pos_x = part_pos[:, 0]
        part_pos_y = part_pos[:, 1]

        # Calculate signed distance to inner edges. Positive means outside the wall.
        # For left wall (Object1), positive distance means part_pos_x > wall1_inner_x.
        # For right wall (Object2), positive distance means part_pos_x < wall2_inner_x.
        dist_to_wall1_inner_x = part_pos_x - wall1_inner_x
        dist_to_wall2_inner_x = wall2_inner_x - part_pos_x

        # Condition for being within the y-extent of the doorway.
        condition_in_doorway_y = (part_pos_y > doorway_y_start) & (part_pos_y < doorway_y_end)

        # Penalize if too close to Object1 (left wall) and within doorway y-extent.
        # Reward is negative exponential, becoming more negative as distance decreases below threshold.
        condition_too_close_wall1 = (dist_to_wall1_inner_x < clearance_threshold) & condition_in_doorway_y
        # The term (clearance_threshold - dist_to_wall1_inner_x) is positive when too close.
        collision_reward += torch.where(condition_too_close_wall1, -torch.exp((clearance_threshold - dist_to_wall1_inner_x) * 10.0), 0.0)

        # Penalize if too close to Object2 (right wall) and within doorway y-extent.
        condition_too_close_wall2 = (dist_to_wall2_inner_x < clearance_threshold) & condition_in_doorway_y
        # The term (clearance_threshold - dist_to_wall2_inner_x) is positive when too close.
        collision_reward += torch.where(condition_too_close_wall2, -torch.exp((clearance_threshold - dist_to_wall2_inner_x) * 10.0), 0.0)

    reward = collision_reward

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def walk_clear_of_doorway_forward_progress_initial_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_clear_of_doorway_forward_progress_initial_reward") -> torch.Tensor:
    """
    Shaping reward for initial forward progress towards the doorway.
    Encourages the robot to move forward (positive y-direction) towards the doorway.
    This reward is primarily active when the robot is still before the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects and robot parts
    object1 = env.scene['Object1'] # Heavy Cube (Wall 1)
    robot = env.scene["robot"]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Hardcoded dimensions from task description for walls: y=5m (length).
    wall_length_y = 5.0

    # Assuming doorway starts at Object1.data.root_pos_w[:, 1] - (wall_length_y / 2.0).
    doorway_y_start = object1.data.root_pos_w[:, 1] - (wall_length_y / 2.0)

    # Reward for decreasing distance to the doorway's y-start.
    # This reward should be positive and continuous, increasing as the robot gets closer to the doorway start.
    # Use a negative exponential of the distance to make it positive and provide a gradient.
    # The distance is `doorway_y_start - pelvis_pos_y`.
    # If pelvis_pos_y is far behind, distance is large positive. If pelvis_pos_y is close, distance is small positive.
    distance_to_doorway_y = doorway_y_start - pelvis_pos_y
    forward_progress_reward = torch.exp(-torch.abs(distance_to_doorway_y) * 2.0) # Scale factor 2.0 for sensitivity

    # Condition: Only active when the robot's pelvis is before the doorway's y-start.
    condition_before_doorway = (pelvis_pos_y < doorway_y_start)
    reward = torch.where(condition_before_doorway, forward_progress_reward, 0.0)

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for clearing the doorway and positioning for the next skill.
    WalkClearOfDoorwayPrimaryReward = RewTerm(func=walk_clear_of_doorway_primary_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "walk_clear_of_doorway_primary_reward"})

    # Shaping reward for avoiding collisions with the walls.
    WalkClearOfDoorwayCollisionAvoidanceReward = RewTerm(func=walk_clear_of_doorway_collision_avoidance_reward, weight=0.6,
                                                         params={"normalise": True, "normaliser_name": "walk_clear_of_doorway_collision_avoidance_reward"})

    # Shaping reward for initial forward progress towards the doorway.
    WalkClearOfDoorwayForwardProgressInitialReward = RewTerm(func=walk_clear_of_doorway_forward_progress_initial_reward, weight=0.4,
                                                              params={"normalise": True, "normaliser_name": "walk_clear_of_doorway_forward_progress_initial_reward"})