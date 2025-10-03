from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer, RewardStats
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch

from isaaclab.envs import mdp
import genhrl.generation.mdp.rewards as custom_rewards


def walk_to_cylinderColumn4_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "walk_to_cylinderColumn4_main_reward") -> torch.Tensor:
    '''Main reward for walk_to_cylinderColumn4.

    Mirrors walk_to_cylinderColumn1: encourages stopping in front of Object4 with small lateral offset.
    '''
    RewNormalizer = get_normalizer(env.device)

    # Object4 position
    object4 = env.scene['Object4']
    object4_pos = object4.data.root_pos_w

    # Robot pelvis position
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Relative distances in XY
    abs_diff_x = torch.abs(object4_pos[:, 0] - pelvis_pos[:, 0])
    abs_diff_y = torch.abs(object4_pos[:, 1] - pelvis_pos[:, 1])

    # Same shaping as column1: minimize absolute distances to the target band
    # Use negative absolute distance as continuous shaping (closer is better)
    target_x_center = 0.5  # middle of 0.4..0.6 band used for success
    reward = -torch.abs(abs_diff_x - target_x_center) - torch.abs(abs_diff_y)

    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward_col4") -> torch.Tensor:
    '''Shaping reward for maintaining desired pelvis height (mirrors column1).'''
    RewNormalizer = get_normalizer(env.device)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2]
    desired_pelvis_z = 0.7
    reward = -torch.abs(pelvis_pos_z - desired_pelvis_z)
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward_col4") -> torch.Tensor:
    '''Shaping reward to avoid penetrating Object4 (same pattern as column1).'''
    RewNormalizer = get_normalizer(env.device)

    object4 = env.scene['Object4']
    robot = env.scene["robot"]

    cylinder_radius = 0.3
    robot_parts_to_check = ['pelvis', 'left_knee_link', 'right_knee_link', 'left_ankle_roll_link', 'right_ankle_roll_link']
    collision_reward = torch.zeros(env.num_envs, device=env.device)

    for part_name in robot_parts_to_check:
        part_idx = robot.body_names.index(part_name)
        part_pos = robot.data.body_pos_w[:, part_idx]
        dist_xy = torch.sqrt(
            (part_pos[:, 0] - object4.data.root_pos_w[:, 0])**2 +
            (part_pos[:, 1] - object4.data.root_pos_w[:, 1])**2
        )
        object_base_z = object4.data.root_pos_w[:, 2] - 1.0  # 2.0 height / 2
        object_top_z = object4.data.root_pos_w[:, 2] + 1.0
        horizontal_proximity_condition = (dist_xy < (cylinder_radius + 0.1))
        vertical_overlap_condition = (part_pos[:, 2] > object_base_z) & (part_pos[:, 2] < object_top_z)
        collision_condition = horizontal_proximity_condition & vertical_overlap_condition
        penalty = torch.where(
            collision_condition,
            -torch.exp(-10.0 * (dist_xy - cylinder_radius)),
            torch.tensor(0.0, device=env.device)
        )
        collision_reward += penalty

    reward = collision_reward
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Mirror column1 weights
    WalkToCylinderColumn4MainReward = RewTerm(func=walk_to_cylinderColumn4_main_reward, weight=1.0,
                                              params={"normalise": True, "normaliser_name": "walk_to_cylinderColumn4_main_reward"})

    PelvisHeightReward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "pelvis_height_reward_col4"})

    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward_col4"})