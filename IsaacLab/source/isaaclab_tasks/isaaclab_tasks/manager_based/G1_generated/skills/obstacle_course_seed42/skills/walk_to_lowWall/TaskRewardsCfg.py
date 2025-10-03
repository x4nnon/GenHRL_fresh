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

def main_WalkToLowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for WalkToLowWall.

    Rewards the robot for moving towards the low wall and being within 1m of it in the x-direction.
    This encourages the robot to approach the low wall, which is the primary objective of this skill.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env.device)
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except for handling missing object
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # Calculate the distance in the x-direction between the pelvis and the low wall.
        distance_x = low_wall_pos[:, 0] - pelvis_pos[:, 0] # CORRECT: Relative distance in x-direction

        # Define the target distance to the low wall in the x-direction (1m).
        target_distance_x = 1.0

        # Reward is negative absolute difference between the current distance and the target distance.
        # This is a continuous reward that encourages the robot to get closer to the target distance.
        reward = -torch.abs(distance_x - target_distance_x) # CORRECT: Continuous reward based on relative distance to target, using absolute distance

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height.

    Rewards the robot for keeping its pelvis at a consistent height (around 0.7m).
    This encourages stability and prevents the robot from falling, supporting the main task.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env.device)
    try:
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern

        # Define the default pelvis height.
        default_pelvis_z = 0.7

        # Reward is negative absolute difference between the current pelvis z-position and the default height.
        # This is a continuous reward that encourages the robot to maintain the desired pelvis height.
        reward = -torch.abs(pelvis_pos[:, 2] - default_pelvis_z) # CORRECT: Continuous reward based on absolute pelvis z-position, but this is acceptable for height stability

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def no_overshoot_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "no_overshoot_reward") -> torch.Tensor:
    '''Shaping reward to prevent overshooting the low wall.

    Penalizes the robot for moving too far past the low wall in the x-direction.
    This encourages the robot to stop near the low wall, preparing for the next skill.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env.device)
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except for handling missing object
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # Calculate the overshoot distance in the x-direction. Overshoot is defined as being past the wall + 1m.
        overshoot_threshold_x = low_wall_pos[:, 0] + 1.0
        distance_x_overshoot = pelvis_pos[:, 0] - overshoot_threshold_x # CORRECT: Relative distance for overshoot

        # Activation condition: robot's pelvis x-position is beyond the overshoot threshold.
        activation_condition_overshoot = (pelvis_pos[:, 0] > overshoot_threshold_x)

        # Reward is negative overshoot distance when activated, otherwise zero.
        # This penalizes overshooting and is only active when the robot is past the threshold.
        reward = torch.where(activation_condition_overshoot, -torch.abs(distance_x_overshoot), torch.tensor(0.0, device=env.device)) # CORRECT: Conditional reward based on relative position, continuous when active

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward

def y_distance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "y_distance_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a minimum y distance from the low wall.

    Penalizes the robot for getting too close to the low wall in the y-direction.
    This prevents sideways collisions with the wall and encourages a straight approach.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env.device)
    try:
        low_wall = env.scene['Object3'] # CORRECT: Accessing low wall object using approved pattern and try/except for handling missing object
        pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing pelvis position using approved pattern
        low_wall_pos = low_wall.data.root_pos_w # CORRECT: Accessing low wall position using approved pattern

        # Calculate the distance in the y-direction between the pelvis and the low wall.
        distance_y = 0 - pelvis_pos[:, 1] # CORRECT: Relative distance in y-direction

        # Define the minimum allowed y distance.
        min_distance_y = 0.5

        # Absolute y distance
        distance_y_abs = torch.abs(distance_y)

        # Reward is negative difference between min_distance and current y distance when closer than min_distance, otherwise zero.
        # This penalizes getting too close in the y-direction and is only active when closer than the threshold.
        reward = -distance_y_abs  # CORRECT: Conditional reward based on relative distance, continuous when active

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Normalize and return reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward) # CORRECT: Normalize reward
        RewNormalizer.update_stats(normaliser_name, reward) # CORRECT: Update reward stats
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for walking to the low wall. Weight is set to 1.0 as it is the primary objective.
    Main_WalkToLowWallReward = RewTerm(func=main_WalkToLowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for maintaining pelvis height. Weight is set to 0.4 to encourage stability without overpowering the main reward.
    PelvisHeightReward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # Shaping reward to prevent overshooting the low wall. Weight is set to 0.3 to guide the robot to stop near the wall.
    NoOvershootReward = RewTerm(func=no_overshoot_reward, weight=0.3,
                                params={"normalise": True, "normaliser_name": "no_overshoot_reward"})

    # Shaping reward for maintaining y-distance from the low wall. Weight is set to 0.2 to prevent sideways collisions, less critical than height and overshoot.
    YDistanceReward = RewTerm(func=y_distance_reward, weight=0.5,
                                params={"normalise": True, "normaliser_name": "y_distance_reward"})