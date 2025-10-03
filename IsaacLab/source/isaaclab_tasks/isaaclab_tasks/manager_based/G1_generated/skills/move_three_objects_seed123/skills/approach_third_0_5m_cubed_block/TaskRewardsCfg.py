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

def approach_third_0_5m_cubed_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'approach_third_0_5m_cubed_block' skill.
    Encourages the robot to position its pelvis directly behind 'Object3' (third 0.5m cubed block),
    facing towards 'Object4' (platform), ready to push.
    """
    # CRITICAL: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CRITICAL: Accessing objects using approved pattern env.scene['ObjectN']
    object3 = env.scene['Object3'] # third 0.5m cubed block
    # object4 = env.scene['Object4'] # platform - used conceptually for direction, not directly for position calculation here

    # CRITICAL: Accessing robot parts using approved pattern robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # CRITICAL: Object dimensions are hardcoded from the object configuration, not accessed from object attributes.
    block_size = 0.5 # 0.5m cubed block, read from object configuration
    
    # Target X: 0.5m behind the block's edge.
    # The block's half-size is block_size / 2.0 = 0.25m.
    # So, 0.25m (half-block) + 0.5m (offset) = 0.75m behind the center of Object3.
    # Assuming platform is in positive X, "behind" means negative X relative to Object3.
    target_x = object3.data.root_pos_w[:, 0] - (block_size / 2.0) - 0.5
    
    # Target Y: Aligned with Object3's center in the Y-axis.
    target_y = object3.data.root_pos_w[:, 1]
    
    # Target Z: A stable pelvis height for standing.
    # CRITICAL: Hardcoded stable pelvis height, as per reward design plan.
    target_pelvis_z = 0.7 

    # CRITICAL: Using relative distances between robot part and object position components.
    distance_x = target_x - pelvis_pos[:, 0]
    distance_y = target_y - pelvis_pos[:, 1]
    distance_z = target_pelvis_z - pelvis_pos[:, 2] # Relative to target Z height

    # CRITICAL: Using negative absolute distance for continuous positive reward as distance decreases.
    # This encourages the robot to minimize the difference in all three dimensions.
    reward = -torch.abs(distance_x) - torch.abs(distance_y) - torch.abs(distance_z)

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain a stable upright posture.
    Penalizes large deviations of the pelvis z-position from a desired stable height.
    """
    # CRITICAL: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CRITICAL: Accessing robot parts using approved pattern robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx][:, 2] # Z-component of pelvis position

    # CRITICAL: Hardcoded desired stable pelvis height, as per reward design plan.
    target_pelvis_z = 0.7 

    # CRITICAL: Using negative absolute difference for continuous positive reward as deviation decreases.
    reward = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to encourage collision avoidance with all other objects in the scene.
    Penalizes the robot if any of its key body parts get too close to any of the objects.
    """
    # CRITICAL: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CRITICAL: Accessing objects using approved pattern env.scene['ObjectN']
    object1 = env.scene['Object1'] # first 0.5m cubed block
    object2 = env.scene['Object2'] # second 0.5m cubed block
    object3 = env.scene['Object3'] # third 0.5m cubed block
    object4 = env.scene['Object4'] # platform

    # CRITICAL: Accessing robot parts using approved pattern robot.body_names.index('part_name')
    robot = env.scene["robot"]
    pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')]
    left_ankle_pos = robot.data.body_pos_w[:, robot.body_names.index('left_ankle_roll_link')]
    right_ankle_pos = robot.data.body_pos_w[:, robot.body_names.index('right_ankle_roll_link')]
    left_palm_pos = robot.data.body_pos_w[:, robot.body_names.index('left_palm_link')]
    right_palm_pos = robot.data.body_pos_w[:, robot.body_names.index('right_palm_link')]

    # Define objects and robot parts to check for collision
    objects_to_check = [object1, object2, object3, object4]
    robot_parts_to_check = [pelvis_pos, left_ankle_pos, right_ankle_pos, left_palm_pos, right_palm_pos]

    collision_penalty = torch.zeros(env.num_envs, device=env.device)
    # CRITICAL: Minimum safe distance from objects, slightly larger than 0.1 to account for body size, as per prompt.
    min_distance_threshold = 0.15 

    for obj in objects_to_check:
        obj_pos = obj.data.root_pos_w
        for part_pos in robot_parts_to_check:
            # CRITICAL: Using torch.norm for Euclidean distance between relative positions.
            distance = torch.norm(obj_pos - part_pos, dim=1)

            # CRITICAL: Using a continuous negative reward. Penalty increases as distance decreases below threshold.
            # clamp ensures penalty is 0 if distance >= threshold, and max at 0 distance.
            penalty_factor = torch.clamp((min_distance_threshold - distance) / min_distance_threshold, min=0.0, max=1.0)
            collision_penalty -= penalty_factor * 1.0 # Scale penalty as needed

    reward = collision_penalty

    # CRITICAL: Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

@configclass
class TaskRewardsCfg:
    # CRITICAL: Main reward weight is 1.0, as per prompt.
    ApproachThirdBlockReward = RewTerm(func=approach_third_0_5m_cubed_block_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "main_reward"})

    # CRITICAL: Shaping reward weights are typically < 1.0, as per prompt.
    PelvisHeightReward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # CRITICAL: Shaping reward weights are typically < 1.0, as per prompt.
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.2,
                                      params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})