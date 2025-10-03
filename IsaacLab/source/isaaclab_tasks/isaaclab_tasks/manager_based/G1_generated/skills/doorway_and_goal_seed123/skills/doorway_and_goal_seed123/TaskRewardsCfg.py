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


def doorway_and_goal_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "doorway_and_goal_primary_reward") -> torch.Tensor:
    """
    Primary reward for the doorway_and_goal_seed123 skill.
    The robot should walk through the doorway and then walk to the small block.

    This reward is phased:
    Phase 1: Encourage the robot's pelvis to move past the doorway's y-coordinate.
    Phase 2: Once past the doorway, encourage the robot's pelvis to move towards the small block (Object3) in x and y.
    Additionally, a small reward component encourages the pelvis to maintain a stable height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    object1 = env.scene['Object1']  # Heavy Cube (Wall 1)
    object3 = env.scene['Object3']  # Small Block

    # Access required robot part(s)
    robot = env.scene["robot"] # Access the robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Determine the doorway's y-position. Assuming walls are aligned at the same y-coordinate.
    # This is a relative position to the wall's root.
    doorway_y_pos = object1.data.root_pos_w[:, 1]

    # Define a small buffer to ensure the robot is clearly past the doorway before switching phases.
    # This buffer is a relative distance from the doorway's y-position.
    doorway_pass_buffer = 0.2  # meters

    # Phase 1: Approach and pass through doorway
    # Reward is negative absolute distance in y to the doorway.
    # This encourages movement in the positive y-direction towards the doorway.
    reward_phase1 = -torch.abs(pelvis_pos_y - doorway_y_pos)

    # Phase 2: Approach the small block (Object3)
    # Reward is negative 2D distance (x and y) to Object3.
    # This encourages the robot to minimize its distance to the block in the horizontal plane.
    distance_x_to_block = object3.data.root_pos_w[:, 0] - pelvis_pos_x
    distance_y_to_block = object3.data.root_pos_w[:, 1] - pelvis_pos_y
    reward_phase2 = -torch.sqrt(distance_x_to_block**2 + distance_y_to_block**2)

    # Combine phases using a conditional based on pelvis y-position relative to the doorway.
    # The reward switches from focusing on passing the doorway to reaching the block.
    primary_reward = torch.where(pelvis_pos_y < doorway_y_pos + doorway_pass_buffer, reward_phase1, reward_phase2)

    # Add a small component for pelvis height stability.
    # This encourages the robot to maintain a reasonable upright posture.
    pelvis_target_z = 0.7  # A common stable pelvis height for humanoid robots.
    reward_pelvis_z = -torch.abs(pelvis_pos_z - pelvis_target_z)
    primary_reward += reward_pelvis_z * 0.1  # Small weight for stability

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def doorway_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "doorway_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward for collision avoidance with the doorway walls (Object1 and Object2).
    This penalizes the robot for getting too close to the walls, especially when passing through the doorway.
    It considers the pelvis and feet positions relative to the wall centers and their dimensions.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    object1 = env.scene['Object1']  # Heavy Cube (Wall 1)
    object2 = env.scene['Object2']  # Heavy Cube (Wall 2)

    # Access required robot part(s)
    robot = env.scene["robot"] # Access the robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_x = left_foot_pos[:, 0]
    left_foot_pos_y = left_foot_pos[:, 1]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]

    # Hardcoded wall dimensions from the skill description: x=0.5m, y=5m
    wall_x_width = 0.5  # Width of the wall in the x-direction
    wall_y_length = 5.0  # Length of the wall in the y-direction

    # Determine the doorway's y-position (from primary reward logic)
    doorway_y_pos = object1.data.root_pos_w[:, 1]

    # Define a region around the doorway in y where collision avoidance is critical.
    # This region extends from before the walls to after them, considering their length.
    # A buffer of 0.5m is added to ensure the penalty is active slightly before and after the walls.
    doorway_y_region_start = doorway_y_pos - (wall_y_length / 2.0) - 0.5
    doorway_y_region_end = doorway_y_pos + (wall_y_length / 2.0) + 0.5

    # Condition to activate collision penalty: robot's pelvis is within the critical y-region.
    is_in_doorway_y_region = (pelvis_pos_y > doorway_y_region_start) & (pelvis_pos_y < doorway_y_region_end)

    # Define a collision buffer: how close the robot can get to the wall's inner edge before penalty.
    # This accounts for the robot's approximate radius (e.g., 0.2m) and a small safety margin.
    collision_buffer = 0.25  # meters (e.g., robot radius ~0.2m + small margin)

    # Calculate the x-coordinates of the inner edges of the doorway walls.
    # Object1 is the left wall, Object2 is the right wall.
    obj1_inner_x_edge = object1.data.root_pos_w[:, 0] + wall_x_width / 2.0
    obj2_inner_x_edge = object2.data.root_pos_w[:, 0] - wall_x_width / 2.0

    # Penalize if pelvis is too far left (colliding with Object1's inner edge).
    # The penalty is an exponential decay, becoming very negative as the distance decreases below the buffer.
    # The term `(pelvis_pos_x - (obj1_inner_x_edge + collision_buffer))` calculates how much the pelvis has
    # penetrated the "safe zone" to the left.
    penalty_pelvis_obj1_x = torch.where(pelvis_pos_x < obj1_inner_x_edge + collision_buffer,
                                        -torch.exp(-((pelvis_pos_x - (obj1_inner_x_edge + collision_buffer))**2) / 0.05),
                                        0.0)

    # Penalize if pelvis is too far right (colliding with Object2's inner edge).
    # Similar exponential penalty for penetration into the right safe zone.
    penalty_pelvis_obj2_x = torch.where(pelvis_pos_x > obj2_inner_x_edge - collision_buffer,
                                        -torch.exp(-(((obj2_inner_x_edge - collision_buffer) - pelvis_pos_x)**2) / 0.05),
                                        0.0)

    # Combine pelvis collision penalties, active only within the doorway y-region.
    collision_penalty_pelvis = torch.where(is_in_doorway_y_region, penalty_pelvis_obj1_x + penalty_pelvis_obj2_x, 0.0)

    # Apply similar collision penalties for the feet.
    # Left foot collision with Object1 (left wall).
    penalty_left_foot_obj1_x = torch.where(left_foot_pos_x < obj1_inner_x_edge + collision_buffer,
                                           -torch.exp(-((left_foot_pos_x - (obj1_inner_x_edge + collision_buffer))**2) / 0.05),
                                           0.0)
    # Right foot collision with Object2 (right wall).
    penalty_right_foot_obj2_x = torch.where(right_foot_pos_x > obj2_inner_x_edge - collision_buffer,
                                            -torch.exp(-(((obj2_inner_x_edge - collision_buffer) - right_foot_pos_x)**2) / 0.05),
                                            0.0)

    # Combine feet collision penalties, active only within the doorway y-region.
    collision_penalty_feet = torch.where(is_in_doorway_y_region, penalty_left_foot_obj1_x + penalty_right_foot_obj2_x, 0.0)

    # Total collision reward is the sum of pelvis and feet penalties.
    reward = collision_penalty_pelvis + collision_penalty_feet

    # Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def forward_progress_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_reward") -> torch.Tensor:
    """
    Shaping reward for general forward progress along the y-axis.
    This encourages continuous movement towards the final target (Object3) in the y-direction,
    helping to prevent the robot from getting stuck or moving backward.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    object3 = env.scene['Object3']  # Small Block (final target)

    # Access required robot part(s)
    robot = env.scene["robot"] # Access the robot object
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_y = pelvis_pos[:, 1]

    # The overall target y-position is Object3's y-position.
    target_y = object3.data.root_pos_w[:, 1]

    # Reward for decreasing the absolute y-distance to the final target (Object3).
    # This provides a continuous gradient, encouraging the robot to always move towards the goal's y-coordinate.
    # The reward is negative, so minimizing this value means getting closer.
    reward = -torch.abs(pelvis_pos_y - target_y)

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
    # Primary reward for navigating through the doorway and to the goal block.
    # Weight is 1.0 as it's the main objective.
    DoorwayAndGoalPrimaryReward = RewTerm(func=doorway_and_goal_primary_reward, weight=1.0,
                                          params={"normalise": True, "normaliser_name": "doorway_and_goal_primary_reward"})

    # Shaping reward for avoiding collisions with the doorway walls.
    # Weight is 0.6 to provide a significant penalty for collisions without overshadowing the primary goal.
    DoorwayCollisionAvoidanceReward = RewTerm(func=doorway_collision_avoidance_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "doorway_collision_avoidance_reward"})

    # Shaping reward for general forward progress towards the final goal.
    # Weight is 0.3 to encourage consistent movement without being too dominant.
    ForwardProgressReward = RewTerm(func=forward_progress_reward, weight=0.3,
                                    params={"normalise": True, "normaliser_name": "forward_progress_reward"})