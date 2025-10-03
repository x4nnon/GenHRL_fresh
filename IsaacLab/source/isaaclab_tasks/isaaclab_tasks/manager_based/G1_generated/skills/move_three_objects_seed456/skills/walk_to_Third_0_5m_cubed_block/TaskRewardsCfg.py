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


def walk_to_Third_0_5m_cubed_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for walk_to_Third_0_5m_cubed_block.

    This reward encourages the robot's pelvis to be at a specific horizontal distance and height relative to the
    center of Object3 (Third 0.5m cubed block), preparing for a pushing action.
    '''
    # Get normalizer instance, as required.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns.
    object3 = env.scene['Object3'] # Accessing Object3 directly as per requirements.
    robot = env.scene["robot"] # Accessing robot directly as per requirements.

    # Get pelvis position using approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Get Object3 position using approved pattern.
    object3_pos = object3.data.root_pos_w

    # Object3 dimensions (0.5m cubed block) - hardcoded from object configuration as required.
    object3_half_size = 0.25 # 0.5m / 2

    # Define target distances for the robot's pelvis relative to Object3's center.
    # Target horizontal distance (XY plane) from pelvis to Object3 center.
    # This aims for the pelvis to be 0.15m from the block's face (0.4m from center - 0.25m half block size).
    target_xy_distance = 0.4
    # Target Z-height for the pelvis, relative to the ground.
    target_pelvis_z = 0.7

    # Calculate distance vector between Object3 center and robot pelvis.
    # Using component-wise differences for clarity and later for specific axis rewards.
    distance_x = object3_pos[:, 0] - pelvis_pos[:, 0]
    distance_y = object3_pos[:, 1] - pelvis_pos[:, 1]
    # The Z-component of the distance vector is not directly used for the Z-height reward,
    # as the Z-height reward uses the absolute pelvis Z-position.
    # distance_z = object3_pos[:, 2] - pelvis_pos[:, 2]

    # Calculate current horizontal distance (XY plane) using torch.norm for batched environments.
    current_xy_distance = torch.sqrt(distance_x**2 + distance_y**2) # Using relative distances as required.

    # Reward for reaching target horizontal distance.
    # Using a negative absolute difference provides a continuous reward that is maximized when current_xy_distance
    # is equal to target_xy_distance, as required for continuous rewards.
    reward_xy = -torch.abs(current_xy_distance - target_xy_distance)

    # Reward for maintaining target pelvis height.
    # The pelvis_pos[:, 2] is the absolute Z-position of the pelvis, which is allowed for height considerations.
    # Using a negative absolute difference for continuous reward, maximized when pelvis_pos_z is target_pelvis_z.
    reward_z = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # Combine rewards.
    primary_reward = reward_xy + reward_z

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def avoid_other_blocks_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    '''Shaping reward 1: Encourages the robot to avoid colliding with Object1 and Object2.

    This reward applies a negative penalty if the robot's pelvis or feet get too close to Object1 or Object2,
    which are adjacent to Object3 and should be avoided during the approach.
    '''
    # Get normalizer instance, as required.
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns.
    object1 = env.scene['Object1'] # Accessing Object1 directly.
    object2 = env.scene['Object2'] # Accessing Object2 directly.
    robot = env.scene["robot"] # Accessing robot directly.

    # Get positions of relevant robot parts using approved patterns.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Object dimensions (0.5m cubed block) - hardcoded from object configuration.
    object_half_size = 0.25

    # Define a safe distance threshold from object center.
    # This means 0.25m from the object's face (object_half_size + 0.25m clearance).
    safe_distance_threshold = object_half_size + 0.25 # 0.5m from center of the block.

    # Calculate Euclidean distances to Object1 for pelvis and feet using torch.norm for batched environments.
    # All distances are relative as required.
    dist_obj1_pelvis = torch.norm(object1.data.root_pos_w - pelvis_pos, dim=1)
    dist_obj1_left_foot = torch.norm(object1.data.root_pos_w - left_foot_pos, dim=1)
    dist_obj1_right_foot = torch.norm(object1.data.root_pos_w - right_foot_pos, dim=1)

    # Calculate Euclidean distances to Object2 for pelvis and feet using torch.norm for batched environments.
    dist_obj2_pelvis = torch.norm(object2.data.root_pos_w - pelvis_pos, dim=1)
    dist_obj2_left_foot = torch.norm(object2.data.root_pos_w - left_foot_pos, dim=1)
    dist_obj2_right_foot = torch.norm(object2.data.root_pos_w - right_foot_pos, dim=1)

    # Collision penalty for Object1.
    # Applies a negative inverse distance penalty when any part is within the safe_distance_threshold.
    # Adding a small epsilon (0.01) to avoid division by zero, ensuring continuity.
    # The penalty should be applied based on the minimum distance to any part to ensure the strongest penalty.
    min_dist_obj1 = torch.min(torch.stack([dist_obj1_pelvis, dist_obj1_left_foot, dist_obj1_right_foot]), dim=0).values
    collision_penalty_obj1 = torch.where(
        min_dist_obj1 < safe_distance_threshold,
        -1.0 / (min_dist_obj1 + 0.01), # Continuous negative reward.
        0.0
    )

    # Collision penalty for Object2.
    min_dist_obj2 = torch.min(torch.stack([dist_obj2_pelvis, dist_obj2_left_foot, dist_obj2_right_foot]), dim=0).values
    collision_penalty_obj2 = torch.where(
        min_dist_obj2 < safe_distance_threshold,
        -1.0 / (min_dist_obj2 + 0.01), # Continuous negative reward.
        0.0
    )

    shaping_reward1 = collision_penalty_obj1 + collision_penalty_obj2

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def maintain_upright_posture_shaping_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    '''Shaping reward 2: Encourages the robot to maintain an upright and stable posture.

    This reward penalizes large deviations in the pelvis's z-velocity, helping prevent the robot from falling over,
    which is crucial for being ready for the next pushing action.
    '''
    # Get normalizer instance, as required.
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part using approved patterns.
    robot = env.scene["robot"] # Accessing robot directly.

    # Get pelvis velocity using approved pattern.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_vel = robot.data.body_vel_w[:, pelvis_idx] # Accessing robot part velocity.
    pelvis_vel_z = pelvis_vel[:, 2] # Extracting Z-component of velocity.

    # Penalize large vertical velocity of the pelvis.
    # Using -torch.abs(pelvis_vel_z) directly provides a continuous negative reward,
    # which is maximized (closest to zero) when the pelvis vertical velocity is zero,
    # encouraging stability and preventing falling.
    shaping_reward2 = -torch.abs(pelvis_vel_z) # Continuous negative reward.

    # Mandatory reward normalization.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # Primary reward for reaching the target position relative to Object3.
    # Weight is 1.0 as it's the main objective.
    WalkToThirdBlockMainReward = RewTerm(func=walk_to_Third_0_5m_cubed_block_main_reward, weight=1.0,
                                         params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for avoiding collisions with Object1 and Object2.
    # Weight is 0.4, less than the primary reward but significant enough to guide behavior.
    AvoidOtherBlocksShapingReward = RewTerm(func=avoid_other_blocks_shaping_reward, weight=0.4,
                                            params={"normalise": True, "normaliser_name": "shaping_reward_1"})

    # Shaping reward for maintaining an upright and stable posture.
    # Weight is 0.2, a smaller value to encourage stability without dominating other objectives.
    MaintainUprightPostureShapingReward = RewTerm(func=maintain_upright_posture_shaping_reward, weight=0.2,
                                                  params={"normalise": True, "normaliser_name": "shaping_reward_2"})