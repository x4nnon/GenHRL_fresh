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


def primary_approach_goal_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_approach_goal_block_reward") -> torch.Tensor:
    """
    Primary reward: Encourages the robot's pelvis to move towards and align with the target small block (Object3)
    in all three dimensions (x, y, z). The z-component also encourages maintaining a stable standing height
    around 0.7m relative to the block's base.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3']

    # Access the required robot part(s) using approved patterns
    pelvis_idx = env.scene["robot"].body_names.index('pelvis')
    pelvis_pos = env.scene["robot"].data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Object3 dimensions (from object config: 0.3m cubed) - Hardcoded as per requirements
    object3_half_height = 0.3 / 2.0

    # Calculate target z for pelvis (pelvis_z = 0.7m for standing, relative to object base)
    # Assuming Object3's root_pos_w[:, 2] is its center, so base is center - half_height
    # This uses relative positions: target_pelvis_z is relative to object3's base.
    target_pelvis_z = object3.data.root_pos_w[:, 2] - object3_half_height + 0.7

    # Calculate the distance vector between the object and the robot part
    # All distances are relative between robot part and object position.
    distance_x = object3.data.root_pos_w[:, 0] - pelvis_pos_x
    distance_y = object3.data.root_pos_w[:, 1] - pelvis_pos_y
    distance_z = target_pelvis_z - pelvis_pos_z

    # Reward is negative absolute distance, so closer is higher reward. This provides a continuous gradient.
    reward = -torch.abs(distance_x) - torch.abs(distance_y) - torch.abs(distance_z)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_doorway_passage_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_doorway_passage_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to pass through the doorway formed by Object1 and Object2.
    It rewards the robot for being centered horizontally (x-axis) between the two walls and for progressing
    forward (y-axis) through the doorway. It is active only when the robot's pelvis is within the y-range of the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']  # Wall 1 (left)
    object2 = env.scene['Object2']  # Wall 2 (right)

    # Access the required robot part(s) using approved patterns
    pelvis_idx = env.scene["robot"].body_names.index('pelvis')
    pelvis_pos = env.scene["robot"].data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Object dimensions (from task description: z=1.5m, x=0.5m, y=5m for walls) - Hardcoded as per requirements
    wall_y_dimension = 5.0

    # Calculate doorway center x-position relative to the walls
    doorway_center_x = (object1.data.root_pos_w[:, 0] + object2.data.root_pos_w[:, 0]) / 2.0

    # Define the doorway's y-range based on the walls' y-dimensions.
    # Assuming walls are centered at some y, and extend +/- 2.5m (half of 5m).
    # This uses relative positions: doorway_y_start/end are relative to wall's y-center.
    wall_y_center = object1.data.root_pos_w[:, 1]  # Assuming both walls have same y-center
    doorway_y_start = wall_y_center - (wall_y_dimension / 2.0)
    doorway_y_end = wall_y_center + (wall_y_dimension / 2.0)

    # Reward for being centered in x within the doorway. Uses relative distance.
    reward_x_centering = -torch.abs(pelvis_pos_x - doorway_center_x)

    # Reward for progressing through the doorway in y.
    # This encourages movement in the positive y direction through the doorway.
    # The robot should be rewarded for having a y-position greater than the start of the doorway.
    # Normalized to 0-1 range within doorway, providing a continuous gradient.
    # Uses relative position: pelvis_pos_y relative to doorway_y_start.
    reward_y_progress = (pelvis_pos_y - doorway_y_start) / (doorway_y_end - doorway_y_start + 1e-6) # Add epsilon for stability

    # Activation condition: Robot's pelvis is within the y-range of the doorway
    activation_condition = (pelvis_pos_y > doorway_y_start) & (pelvis_pos_y < doorway_y_end)

    # Combine rewards, only active within the doorway. Uses torch.where for conditional reward.
    reward = torch.where(activation_condition, reward_x_centering + reward_y_progress, torch.tensor(0.0, device=env.device))

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def shaping_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 2: Penalizes the robot for getting too close to the walls (Object1 and Object2),
    encouraging collision avoidance. It applies a negative reward that increases sharply as the robot
    approaches the walls. This reward is always active.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object1 = env.scene['Object1']  # Wall 1
    object2 = env.scene['Object2']  # Wall 2

    # Access the required robot part(s) using approved patterns
    pelvis_idx = env.scene["robot"].body_names.index('pelvis')
    pelvis_pos = env.scene["robot"].data.body_pos_w[:, pelvis_idx]

    # Object dimensions (from object config: x=0.5m for walls) - Hardcoded as per requirements
    wall_half_width_x = 0.5 / 2.0

    # Define a small clearance threshold for penalty activation
    clearance_threshold = 0.15 # A small buffer distance

    # Calculate distance to Object1 (left wall)
    # Distance from pelvis to the right edge of Object1. All distances are relative.
    dist_to_obj1_x = pelvis_pos[:, 0] - (object1.data.root_pos_w[:, 0] + wall_half_width_x)

    # Penalty for being too close to Object1 (left wall)
    # Penalize if pelvis is to the right of Object1's right edge but too close (within clearance_threshold)
    # The penalty is inverse proportional to distance, making it sharp.
    penalty_obj1 = torch.where(dist_to_obj1_x < clearance_threshold, -1.0 / (dist_to_obj1_x + 1e-6), torch.tensor(0.0, device=env.device))
    # Also penalize if pelvis is actually inside Object1 (pelvis_x < Object1's right edge)
    penalty_obj1 = torch.where(pelvis_pos[:, 0] < object1.data.root_pos_w[:, 0] + wall_half_width_x, penalty_obj1 - 5.0, penalty_obj1)


    # Calculate distance to Object2 (right wall)
    # Distance from pelvis to the left edge of Object2. All distances are relative.
    dist_to_obj2_x = (object2.data.root_pos_w[:, 0] - wall_half_width_x) - pelvis_pos[:, 0]

    # Penalty for being too close to Object2 (right wall)
    # Penalize if pelvis is to the left of Object2's left edge but too close (within clearance_threshold)
    penalty_obj2 = torch.where(dist_to_obj2_x < clearance_threshold, -1.0 / (dist_to_obj2_x + 1e-6), torch.tensor(0.0, device=env.device))
    # Also penalize if pelvis is actually inside Object2 (pelvis_x > Object2's left edge)
    penalty_obj2 = torch.where(pelvis_pos[:, 0] > object2.data.root_pos_w[:, 0] - wall_half_width_x, penalty_obj2 - 5.0, penalty_obj2)

    # Combine penalties
    reward = penalty_obj1 + penalty_obj2

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
    # Primary reward for approaching the goal block
    PrimaryApproachGoalBlockReward = RewTerm(func=primary_approach_goal_block_reward, weight=1.0,
                                             params={"normalise": True, "normaliser_name": "primary_approach_goal_block_reward"})

    # Shaping reward for passing through the doorway
    ShapingDoorwayPassageReward = RewTerm(func=shaping_doorway_passage_reward, weight=0.4,
                                          params={"normalise": True, "normaliser_name": "shaping_doorway_passage_reward"})

    # Shaping reward for avoiding collisions with walls
    ShapingCollisionAvoidanceReward = RewTerm(func=shaping_collision_avoidance_reward, weight=0.6,
                                              params={"normalise": True, "normaliser_name": "shaping_collision_avoidance_reward"})