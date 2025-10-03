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

def main_LandStablyAfterLowWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for LandStablyAfterLowWall.

    Reward for being in the target x range just past the low wall and before the large sphere, and close to the ground.
    This encourages the robot to land stably after jumping over the low wall in the desired zone.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env)
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except
        large_sphere = env.scene['Object1'] # Accessing large sphere object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_z = pelvis_pos[:, 2]

        distance_z = pelvis_pos_z # Relative distance in z from ground level (z=0)

        pelvis_default_z = 0.7


        reward_z = -torch.abs(distance_z-pelvis_default_z) # Reward for pelvis being the correct height.

        reward = reward_z # Combining x and z rewards

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_1_increase_pelvis_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    '''Shaping reward for increasing pelvis height as robot approaches the low wall.
    Encourages the robot to jump as it gets closer to the wall.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env)
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]
        pelvis_pos_z = pelvis_pos[:, 2]

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        distance_x_to_wall = low_wall_pos_x - pelvis_pos_x # Relative distance in x to the low wall

        activation_condition = (pelvis_pos_x > low_wall_pos_x) # Activation when robot is past the low wall (after the wall in x direction)

        reward_height = pelvis_pos_z # Reward for increasing pelvis height (absolute z position, but used as relative increase from ground)
        reward = reward_height

        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_2_feet_close_to_ground(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    '''Shaping reward for both feet being close to the ground after passing the low wall in x direction.
    Encourages stable landing on both feet.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env)
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        left_foot_idx = robot.body_names.index('left_ankle_roll_link') # Accessing left foot index using approved pattern
        right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Accessing right foot index using approved pattern
        left_foot_pos = robot.data.body_pos_w[:, left_foot_idx] # Accessing left foot position using approved pattern
        right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Accessing right foot position using approved pattern
        left_foot_pos_z = left_foot_pos[:, 2]
        right_foot_pos_z = right_foot_pos[:, 2]
        pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        activation_condition = (pelvis_pos_x > low_wall_pos_x) # Activation when robot is past the low wall in x direction

        reward_left_foot = -torch.abs(left_foot_pos_z - 0.0) # Reward for left foot being close to ground
        reward_right_foot = -torch.abs(right_foot_pos_z - 0.0) # Reward for right foot being close to ground

        reward = reward_left_foot + reward_right_foot # Combining left and right foot rewards
        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_3_stable_pelvis_height(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_3") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height after landing.
    Encourages body stabilization after the jump. Target pelvis height is set to 0.7m (relative target height).
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env)
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]
        pelvis_pos_z = pelvis_pos[:, 2]
        target_pelvis_z = 0.7 # Target pelvis height (relative to ground)

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern

        activation_condition = (pelvis_pos_x > low_wall_pos_x) # Activation when robot is past the low wall in x direction

        reward_pelvis_z = -torch.abs(pelvis_pos_z - target_pelvis_z) # Reward for pelvis being at a stable height (relative distance to target height)

        reward = reward_pelvis_z
        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_4_center_y_position(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_4") -> torch.Tensor:
    '''Shaping reward for centering the robot at y=0.
    Encourages the robot to stay centered in the y-direction after landing.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env)
    try:
        low_wall = env.scene['Object3'] # Accessing low wall object using approved pattern and try/except
        large_sphere = env.scene['Object1'] # Accessing large sphere object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing pelvis index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing pelvis position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0]
        pelvis_pos_y = pelvis_pos[:, 1]

        low_wall_pos_x = low_wall.data.root_pos_w[:, 0] # Accessing low wall x position using approved pattern
        large_sphere_y = large_sphere.data.root_pos_w[:, 1] # Accessing large sphere y position using approved pattern
        
        # Activation when robot is past the low wall in x direction
        activation_condition = (pelvis_pos_x > low_wall_pos_x)

        # Reward is negative absolute distance from y=0
        reward = -torch.abs(pelvis_pos_y - large_sphere_y)
        reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device)) # Apply activation condition

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward



@configclass
class TaskRewardsCfg:
    Main_LandStablyAfterLowWallReward = RewTerm(func=main_LandStablyAfterLowWall_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    ShapingReward_IncreasePelvisHeight = RewTerm(func=shaping_reward_1_increase_pelvis_height, weight=0.5,
                                params={"normalise": True, "normaliser_name": "shaping_reward_1"})

    ShapingReward_FeetCloseToGround = RewTerm(func=shaping_reward_2_feet_close_to_ground, weight=0.6,
                                params={"normalise": True, "normaliser_name": "shaping_reward_2"})

    ShapingReward_StablePelvisHeight = RewTerm(func=shaping_reward_3_stable_pelvis_height, weight=0.3,
                                params={"normalise": True, "normaliser_name": "shaping_reward_3"})

    ShapingReward_CenterYPosition = RewTerm(func=shaping_reward_4_center_y_position, weight=0.5,
                                params={"normalise": True, "normaliser_name": "shaping_reward_4"})