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


def walk_to_small_block_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_small_block_primary_reward") -> torch.Tensor:
    """
    Primary reward for guiding the robot's pelvis to the Small Block (Object3).
    It minimizes the Euclidean distance between the robot's pelvis and the center of Object3,
    focusing on x and y for horizontal alignment and z for maintaining a stable standing height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object3 and robot's pelvis using approved patterns
    object_name = env.scene['Object3'] # Accessing Object3 directly as per requirements
    robot = env.scene["robot"]
    robot_partX_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
    robot_partX_pos = robot.data.body_pos_w[:, robot_partX_idx] # Accessing pelvis position using approved pattern

    # Object3 dimensions (0.3m cubed) - hardcoded from object configuration as per requirements
    object3_size_z = 0.3

    # Target positions are relative to Object3's center
    target_x = object_name.data.root_pos_w[:, 0] # Target x-position is the center of Object3
    target_y = object_name.data.root_pos_w[:, 1] # Target y-position is the center of Object3
    # Target z-position for pelvis: 0.7m above the top of the block
    target_z = object_name.data.root_pos_w[:, 2] + (object3_size_z / 2) + 0.7

    # Calculate rewards for each axis, using absolute differences for continuous negative reward
    # This encourages the robot to minimize the distance to the target in all dimensions.
    reward_x = -torch.abs(robot_partX_pos[:, 0] - target_x) # Reward for aligning in x-axis
    reward_y = -torch.abs(robot_partX_pos[:, 1] - target_y) # Reward for approaching the target in y-axis (main progress)
    reward_z = -torch.abs(robot_partX_pos[:, 2] - target_z) # Reward for maintaining stable z-height

    reward = reward_y + reward_x + reward_z # Summing individual axis rewards

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def doorway_navigation_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "doorway_navigation_shaping_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to pass through the doorway (formed by Object1 and Object2)
    without colliding. It rewards maintaining a safe distance from the inner faces of the walls
    while the robot's pelvis is within the y-range of the doorway.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object1 (left wall) and Object2 (right wall) and robot's pelvis
    object1 = env.scene['Object1'] # Accessing Object1 directly
    object2 = env.scene['Object2'] # Accessing Object2 directly
    robot = env.scene["robot"]
    robot_partX_idx = robot.body_names.index('pelvis') # Accessing pelvis index
    robot_partX_pos = robot.data.body_pos_w[:, robot_partX_idx] # Accessing pelvis position

    # Object1 and Object2 dimensions (x=0.5m, y=5m, z=1.5m) - hardcoded from object configuration
    wall_x_dim = 0.5
    wall_y_dim = 5.0

    # Define inner edges of the doorway in x-axis.
    # Assuming Object1 is to the left (lower x) and Object2 to the right (higher x).
    # The robot needs to stay between these inner edges.
    doorway_x_inner_left = object1.data.root_pos_w[:, 0] + (wall_x_dim / 2)
    doorway_x_inner_right = object2.data.root_pos_w[:, 0] - (wall_x_dim / 2)

    # Define a safe clearance distance to avoid touching walls
    clearance = 0.2 # meters, a continuous value to encourage staying away from walls

    # Calculate distance to left wall (Object1) and right wall (Object2) in x-axis
    dist_to_wall1_x = robot_partX_pos[:, 0] - doorway_x_inner_left
    dist_to_wall2_x = doorway_x_inner_right - robot_partX_pos[:, 0]

    # Reward for staying within the x-bounds of the doorway.
    # Maximize the minimum distance to either wall, up to a certain clearance.
    # This creates a positive reward for being centered and away from walls.
    reward_clearance_x = torch.min(dist_to_wall1_x, dist_to_wall2_x)
    # Clamp the reward to be between 0 and 'clearance', then normalize to 0-1.
    # This makes the reward 1.0 when the robot is at least 'clearance' away from both walls,
    # and linearly decreases to 0 as it approaches the walls.
    reward_clearance_x = torch.clamp(reward_clearance_x, min=0, max=clearance) / clearance

    # Activation condition: Robot is within the y-range of the doorway.
    # The walls are 5m long in y. Let's assume the doorway is defined by the y-extent of Object1.
    # This ensures the reward is only active when the robot is actually attempting to pass through the doorway.
    doorway_y_start = object1.data.root_pos_w[:, 1] - (wall_y_dim / 2)
    doorway_y_end = object1.data.root_pos_w[:, 1] + (wall_y_dim / 2)

    activation_condition = (robot_partX_pos[:, 1] > doorway_y_start) & (robot_partX_pos[:, 1] < doorway_y_end)

    # Apply the reward only when the activation condition is met
    reward = torch.where(activation_condition, reward_clearance_x, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_shaping_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain an upright and stable posture
    by keeping its pelvis at a desired height (0.7m above the ground).
    This is continuously active throughout the skill execution.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot's pelvis position
    robot = env.scene["robot"]
    robot_partX_idx = robot.body_names.index('pelvis') # Accessing pelvis index
    robot_partX_pos = robot.data.body_pos_w[:, robot_partX_idx] # Accessing pelvis position

    # Desired pelvis height for stable standing
    desired_pelvis_z = 0.7 # Hardcoded desired height, as per plan

    # Reward is negative absolute difference, encouraging the pelvis to stay close to desired_pelvis_z
    reward = -torch.abs(robot_partX_pos[:, 2] - desired_pelvis_z)

    # Mandatory normalization
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
    Reward terms for the Walk_to_Small_Block skill.
    """
    # Primary reward: Guides the robot to the Small Block (Object3)
    WalkToSmallBlockPrimaryReward = RewTerm(
        func=walk_to_small_block_primary_reward,
        weight=1.0, # Main reward, typically has the highest weight
        params={"normalise": True, "normaliser_name": "walk_to_small_block_primary_reward"}
    )

    # Shaping reward 1: Encourages safe passage through the doorway
    DoorwayNavigationShapingReward = RewTerm(
        func=doorway_navigation_shaping_reward,
        weight=0.6, # Supporting reward, lower weight than primary
        params={"normalise": True, "normaliser_name": "doorway_navigation_shaping_reward"}
    )

    # Shaping reward 2: Encourages maintaining stable pelvis height
    PelvisHeightShapingReward = RewTerm(
        func=pelvis_height_shaping_reward,
        weight=0.4, # Supporting reward, lower weight than primary
        params={"normalise": True, "normaliser_name": "pelvis_height_shaping_reward"}
    )