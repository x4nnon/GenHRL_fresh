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

def main_knock_over_column_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_knock_over_column_reward") -> torch.Tensor:
    """
    Primary reward for knocking over a single cylinder column.
    This reward encourages the robot to first approach the target cylinder column by minimizing the horizontal distance
    between its pelvis and the column. Once the robot is within a certain horizontal proximity, the reward shifts to
    encouraging the column to fall over by minimizing its Z-position. After the column is knocked over, it rewards
    maintaining pelvis stability and proximity to the knocked-over column.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    # Accessing the target column (Object1) as per the object configuration and prompt instructions.
    target_column = env.scene['Object1']
    # Accessing the robot object using approved pattern.
    robot = env.scene["robot"]

    # Getting the index of the pelvis using approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    # Getting the position of the pelvis using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is used as an absolute value for height, which is allowed for this specific purpose.

    # Object dimensions (hardcoded from skill info, as per requirements)
    # This value is hardcoded from the provided object configuration (radius of cylinder column).
    column_radius = 0.3
    # column_height = 2.0 # Not directly used in reward calculation, but useful for context

    # Calculate horizontal distance components between pelvis and column base
    # All distances are relative between robot parts and objects, as required.
    dist_x_pelvis_column = target_column.data.root_pos_w[:, 0] - pelvis_pos_x
    dist_y_pelvis_column = target_column.data.root_pos_w[:, 1] - pelvis_pos_y
    horizontal_dist_pelvis_column = torch.sqrt(dist_x_pelvis_column**2 + dist_y_pelvis_column**2)

    # Calculate column's current Z-position (center)
    # Z-position is used as an absolute value for height, which is allowed for this specific purpose.
    column_z_pos = target_column.data.root_pos_w[:, 2]

    # Threshold for switching from approach to knock-over phase
    # This threshold is relative to the robot's proximity to the column, not an arbitrary world coordinate.
    approach_threshold_xy = 0.5

    # Reward for approaching the column (when column is standing)
    # This is a continuous reward that decreases as the robot gets closer horizontally.
    approach_reward = -horizontal_dist_pelvis_column

    # Reward for knocking over the column (when robot is close and column is still standing)
    # This is a continuous reward that encourages the column's Z-position to decrease towards its radius (on the floor).
    knock_over_reward = -torch.abs(column_z_pos - column_radius)

    # Condition for approach phase: column is standing (Z > radius + small margin) AND robot is far horizontally
    # The margin (0.1) is relative to the column's radius, ensuring it's still considered upright.
    is_column_standing = column_z_pos > (column_radius + 0.1)
    is_far_horizontally = horizontal_dist_pelvis_column > approach_threshold_xy

    # Condition for knock-over phase: column is standing AND robot is close horizontally
    is_close_horizontally = horizontal_dist_pelvis_column <= approach_threshold_xy

    # Combine rewards based on phases using torch.where for continuous transitions
    # Phase 1: Approach the column
    # Phase 2: Knock over the column
    # Phase 3: Column is down, reward stability and proximity for transition to next skill
    primary_reward = torch.where(is_column_standing & is_far_horizontally,
                                 approach_reward,
                                 torch.where(is_column_standing & is_close_horizontally,
                                             knock_over_reward,
                                             # Reward for stability (pelvis height) and maintaining proximity to the knocked-over column
                                             # These are continuous rewards based on relative distances/heights.
                                             -torch.abs(pelvis_pos_z - 0.7) - horizontal_dist_pelvis_column
                                            )
                                )

    # Mandatory reward normalization as per requirements.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def pelvis_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain a stable upright posture by keeping its pelvis at a desired height (0.7m)
    throughout the skill. It also penalizes the robot for falling over.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part using approved pattern.
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_z = pelvis_pos[:, 2] # Z-position is used as an absolute value for height, which is allowed.

    # Reward for maintaining pelvis height around 0.7m
    # This is a continuous reward that penalizes deviation from the target height.
    pelvis_height_reward = -torch.abs(pelvis_pos_z - 0.7)

    # Large negative reward if pelvis falls below a certain threshold (e.g., 0.3m, indicating a fall)
    # This is a conditional penalty, making the reward non-continuous at the threshold, but it's a necessary fall penalty.
    fall_penalty = torch.where(pelvis_pos_z < 0.3, -10.0, 0.0)

    reward = pelvis_height_reward + fall_penalty

    # Mandatory reward normalization as per requirements.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def hand_interaction_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_interaction_reward") -> torch.Tensor:
    """
    This reward encourages the robot to use its hands to interact with the column when it is close enough,
    promoting the desired method of applying force. It rewards minimizing the distance between the robot's hands
    and the column's center, but only when the robot is horizontally close to the column and the column is still standing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns.
    # Accessing the target column (Object1) as per the object configuration and prompt instructions.
    target_column = env.scene['Object1']
    # Accessing the robot object using approved pattern.
    robot = env.scene["robot"]

    # Getting the indices and positions of hands using approved patterns.
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    # Getting the index and position of pelvis for proximity check using approved patterns.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Object dimensions (hardcoded from skill info, as per requirements)
    # This value is hardcoded from the provided object configuration (radius of cylinder column).
    column_radius = 0.3
    # column_height = 2.0 # Not directly used in reward calculation

    # Calculate horizontal distance between pelvis and column base
    # This is a relative distance used for activation condition, as required.
    dist_x_pelvis_column = target_column.data.root_pos_w[:, 0] - pelvis_pos_x
    dist_y_pelvis_column = target_column.data.root_pos_w[:, 1] - pelvis_pos_y
    horizontal_dist_pelvis_column = torch.sqrt(dist_x_pelvis_column**2 + dist_y_pelvis_column**2)

    # Calculate 3D distance between right hand and column center
    # All distances are relative between robot parts and objects, as required.
    dist_x_rh_column = target_column.data.root_pos_w[:, 0] - right_hand_pos[:, 0]
    dist_y_rh_column = target_column.data.root_pos_w[:, 1] - right_hand_pos[:, 1]
    dist_z_rh_column = target_column.data.root_pos_w[:, 2] - right_hand_pos[:, 2]
    dist_rh_column = torch.sqrt(dist_x_rh_column**2 + dist_y_rh_column**2 + dist_z_rh_column**2)

    # Calculate 3D distance between left hand and column center
    dist_x_lh_column = target_column.data.root_pos_w[:, 0] - left_hand_pos[:, 0]
    dist_y_lh_column = target_column.data.root_pos_w[:, 1] - left_hand_pos[:, 1]
    dist_z_lh_column = target_column.data.root_pos_w[:, 2] - left_hand_pos[:, 2]
    dist_lh_column = torch.sqrt(dist_x_lh_column**2 + dist_y_lh_column**2 + dist_z_lh_column**2)

    # Take the minimum distance from either hand to the column
    min_hand_dist_to_column = torch.min(dist_rh_column, dist_lh_column)

    # Condition for activation: robot is close horizontally to the column AND column is still standing
    # The column standing check uses Z-position relative to its radius.
    # The horizontal proximity check uses a relative distance threshold.
    is_column_standing = target_column.data.root_pos_w[:, 2] > (column_radius + 0.1)
    is_close_horizontally = horizontal_dist_pelvis_column <= 0.7 # Slightly larger than primary threshold to activate earlier

    # Reward for hands being close to the column
    # This is a continuous reward that decreases as hands get closer.
    hand_proximity_reward = -min_hand_dist_to_column

    # Apply reward only when conditions are met, otherwise 0.0
    reward = torch.where(is_column_standing & is_close_horizontally, hand_proximity_reward, 0.0)

    # Mandatory reward normalization as per requirements.
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
    Reward terms for the 'knock_over_all_cylinder_columns' skill.
    """
    # Primary reward for approaching and knocking over the column with weight 1.0 as per requirements.
    main_knock_over_column_reward = RewTerm(func=main_knock_over_column_reward, weight=1.0,
                                            params={"normalise": True, "normaliser_name": "main_knock_over_column_reward"})

    # Shaping reward for maintaining pelvis stability and preventing falls with lower weight (0.4) as per requirements.
    pelvis_stability_reward = RewTerm(func=pelvis_stability_reward, weight=0.4,
                                      params={"normalise": True, "normaliser_name": "pelvis_stability_reward"})

    # Shaping reward for encouraging hand interaction with the column with lower weight (0.3) as per requirements.
    hand_interaction_reward = RewTerm(func=hand_interaction_reward, weight=0.3,
                                      params={"normalise": True, "normaliser_name": "hand_interaction_reward"})