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

def main_WalkToLargeSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for WalkToLargeSphere.

    Reward for moving the robot's pelvis closer to the large sphere in the horizontal (x-y) plane.
    This encourages the robot to walk towards the large sphere, fulfilling the primary objective of the skill.
    The reward is inversely proportional to the horizontal distance, providing a continuous signal as the robot approaches the sphere.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env)
    try:
        large_sphere = env.scene['Object1'] # Accessing object using approved pattern and try/except

        pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
        pelvis_pos_x = pelvis_pos[:, 0] # Separating x component
        pelvis_pos_y = pelvis_pos[:, 1] # Separating y component

        large_sphere_pos = large_sphere.data.root_pos_w # Accessing object position using approved pattern
        large_sphere_pos_x = large_sphere_pos[:, 0] # Separating x component
        large_sphere_pos_y = large_sphere_pos[:, 1] # Separating y component

        distance_x = (large_sphere_pos_x -0.8) - pelvis_pos_x  # Relative distance in x-direction
        distance_y = large_sphere_pos_y - pelvis_pos_y # Relative distance in y-direction

        horizontal_distance = torch.sqrt(distance_x**2 + distance_y**2) # Euclidean distance in x-y plane
        reward = -horizontal_distance # Negative distance to reward getting closer

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object, return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stable_pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stable_height_reward") -> torch.Tensor:
    '''Shaping reward for maintaining a stable pelvis height.

    Encourages the robot to maintain a pelvis height close to 0.7m, promoting balance and stability during walking.
    This is a shaping reward to prevent the robot from crouching too low or standing too high, which could hinder its movement.
    The reward is based on the negative absolute difference between the pelvis z-position and the target height, providing a continuous signal.
    '''
    robot = env.scene["robot"] # Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env)
    try:
        pelvis_idx = robot.body_names.index('pelvis') # Accessing robot part index using approved pattern
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Accessing robot part position using approved pattern
        pelvis_pos_z = pelvis_pos[:, 2] # Separating z component

        default_pelvis_z = 0.7 # Default pelvis height - not hardcoded position, but a relative target height

        reward = -torch.abs(pelvis_pos_z - default_pelvis_z) # Negative absolute difference from default pelvis height

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # Handle missing object (though pelvis should always exist), return zero reward

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def sphere_contact_penalty(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "sphere_contact_penalty") -> torch.Tensor:
    '''Penalty reward when too close to the large sphere.
    
    Applies a negative reward when the robot's pelvis is within 0.6m of the sphere's center
    to discourage getting too close.
    '''
    RewNormalizer = get_normalizer(env)
    try:
        # Get positions
        robot = env.scene["robot"]
        pelvis_idx = robot.body_names.index('pelvis')
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        
        large_sphere = env.scene['Object1']
        large_sphere_pos = large_sphere.data.root_pos_w

        # Calculate distance to sphere center
        distance_x = (large_sphere_pos[:, 0] - 0.8) - pelvis_pos[:, 0]
        distance_y = large_sphere_pos[:, 1] - pelvis_pos[:, 1]
        distance = torch.sqrt(distance_x**2 + distance_y**2)
        
        # Negative reward when within 0.6m
        too_close = (distance < 0.6)
        reward = torch.where(too_close,
                           torch.ones(env.num_envs, device=env.device) * -1.0,
                           torch.zeros(env.num_envs, device=env.device))

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device)

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward



@configclass
class TaskRewardsCfg:
    Main_WalkToLargeSphereReward = RewTerm(func=main_WalkToLargeSphere_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"}) # Main reward with weight 1.0
    StablePelvisHeightReward = RewTerm(func=stable_pelvis_height_reward, weight=0.6,
                                params={"normalise": True, "normaliser_name": "stable_height_reward"}) # Shaping reward with weight 0.6