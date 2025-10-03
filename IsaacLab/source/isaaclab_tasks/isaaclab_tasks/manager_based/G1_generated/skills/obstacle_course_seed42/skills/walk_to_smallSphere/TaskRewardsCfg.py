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


def main_WalkToSmallSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for WalkToSmallSphere.

    Reward for reducing the horizontal distance to the small sphere until within 1m.
    This encourages the robot to walk towards the small sphere.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env.device)
    try:
        small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere using approved pattern and try/except
        robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
        small_sphere_pos = small_sphere.data.root_pos_w # CORRECT: Accessing object position using approved pattern

        # CORRECT: Calculate relative distance - horizontal distance between pelvis and small sphere
        distance_x = small_sphere_pos[:, 0] - robot_pelvis_pos[:, 0] - 0.5
        distance_y = small_sphere_pos[:, 1] - robot_pelvis_pos[:, 1]
        Distance = torch.sqrt(distance_x**2 + distance_y**2)

        # CORRECT: Reward for reducing horizontal distance, saturated at 1m, continuous reward
        reward = -torch.abs(Distance)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward


    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_forward_progress(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "forward_progress_reward") -> torch.Tensor:
    '''Shaping reward for forward progress towards the small sphere in the x direction.

    Rewards the robot for moving closer to the small sphere in the x direction when behind it.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env.device)
    try:
        small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere using approved pattern and try/except
        robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
        robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
        robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
        small_sphere_pos_x = small_sphere.data.root_pos_w[:, 0] # CORRECT: Accessing object position using approved pattern

        # CORRECT: Calculate relative distance - x distance to sphere
        distance_x = small_sphere_pos_x - robot_pelvis_pos_x - 0.5


        # CORRECT: Reward forward progress in x direction, continuous reward
        reward = -torch.abs(distance_x)

    except KeyError:
        reward = torch.zeros(env.num_envs, device=env.device) # CORRECT: Handle missing object, return zero reward

    # Add clipping before normalization
    reward = torch.clip(reward, min=-3.0, max=3.0) # Choose bounds appropriate for your expected reward scale

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_reward_stability(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stability_reward") -> torch.Tensor:
    '''Shaping reward for maintaining pelvis stability in z direction.

    Rewards the robot for keeping its pelvis at a nominal standing height.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern

    RewNormalizer = get_normalizer(env.device)
    robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # CORRECT: Reward for maintaining pelvis height around 0.7m, continuous reward based on relative height
    target_pelvis_z = 0.7
    reward = -torch.abs(robot_pelvis_pos_z - target_pelvis_z)

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    '''Reward for avoiding collisions with the small sphere.

    Rewards the robot for avoiding collisions with the small sphere.
    '''
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    RewNormalizer = get_normalizer(env.device)
    small_sphere = env.scene['Object2'] # CORRECT: Accessing small sphere using approved pattern and try/except
    robot_pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    small_sphere_pos = small_sphere.data.root_pos_w # CORRECT: Accessing object position using approved pattern

    # CORRECT: Calculate relative distance - x distance to sphere
    distance_x_to_sphere = small_sphere_pos[:, 0] - robot_pelvis_pos[:, 0]
    distance_y_to_sphere = small_sphere_pos[:, 1] - robot_pelvis_pos[:, 1]
    distance_to_sphere = torch.sqrt(distance_x_to_sphere**2 + distance_y_to_sphere**2)

    # CORRECT: Reward for avoiding collisions with the small sphere
    reward = distance_to_sphere**2

    activation_condition = distance_to_sphere < 0.5

    reward = torch.where(activation_condition, reward, torch.tensor(0.0, device=env.device))

    # Add collision avoidance with high wall (Object4) when robot is before it
    high_wall = env.scene['Object4']
    high_wall_pos = high_wall.data.root_pos_w
    
    # Calculate distance to high wall in x-y plane
    distance_x_to_wall = high_wall_pos[:, 0] - robot_pelvis_pos[:, 0] 
    distance_y_to_wall = high_wall_pos[:, 1] - robot_pelvis_pos[:, 1]
    distance_to_wall = torch.sqrt(distance_x_to_wall**2 + distance_y_to_wall**2)
    
    # Only apply wall avoidance when robot is before the wall
    wall_activation = robot_pelvis_pos[:, 0] < high_wall_pos[:, 0]
    wall_reward = distance_to_wall
    wall_reward = torch.where(wall_activation, -1, torch.tensor(0.0, device=env.device))
    
    # Combine sphere and wall avoidance rewards
    reward = reward + wall_reward

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward

    return reward

def jump_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "jump_reward") -> torch.Tensor:
    '''Reward for jumping over the small sphere.
    
    Rewards the robot for jumping over the small sphere.
    '''
    robot = env.scene["robot"]
    RewNormalizer = get_normalizer(env.device)

    # if agent is before the high wall, reward for increasing the feet and pelvis height.
    high_wall = env.scene['Object4']
    high_wall_x = high_wall.data.root_pos_w[:, 0]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    left_foot_pos_z = left_foot_pos[:, 2]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]
    
    activation_condition = pelvis_pos_x < high_wall_x + 0.5

    reward_before = -torch.abs(pelvis_pos_z-1.5) - torch.abs(right_foot_pos_z - 1.0) - torch.abs(left_foot_pos_z - 1.0)
    reward_after = -torch.abs(pelvis_pos_z-0.7) - torch.min(torch.abs(right_foot_pos_z), torch.abs(left_foot_pos_z))

    reward = torch.where(activation_condition, reward_before, reward_after)

    # CORRECT: Reward normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward



@configclass
class TaskRewardsCfg:
    # CORRECT: Main reward with weight 1.0
    Main_WalkToSmallSphereReward = RewTerm(func=main_WalkToSmallSphere_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "main_reward"})

    # CORRECT: Supporting rewards with lower weights
    ForwardProgressReward = RewTerm(func=shaping_reward_forward_progress, weight=0.0,
                                params={"normalise": True, "normaliser_name": "forward_progress_reward"})
    
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.5,
                                params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})

    StabilityReward = RewTerm(func=shaping_reward_stability, weight=0.0,
                                params={"normalise": True, "normaliser_name": "stability_reward"})

    height_reward = RewTerm(func=jump_reward, weight=1.0,
                                params={"normalise": True, "normaliser_name": "jump_reward"})