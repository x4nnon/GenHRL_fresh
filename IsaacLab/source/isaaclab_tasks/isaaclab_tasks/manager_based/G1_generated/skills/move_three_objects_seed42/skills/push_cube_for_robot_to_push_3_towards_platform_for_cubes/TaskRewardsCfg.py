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


def push_object3_towards_object4_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "push_object3_towards_object4_reward") -> torch.Tensor:
    """
    Primary reward for pushing Object3 (cube) towards Object4 (platform).
    This reward encourages Object3's proximity to Object4 using only current relative positions.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']

    # Current position of Object3
    object3_pos = object3.data.root_pos_w
    object3_pos_x = object3_pos[:, 0]
    object3_pos_y = object3_pos[:, 1]
    object3_pos_z = object3_pos[:, 2]

    # Position of Object4 (platform)
    object4_pos = object4.data.root_pos_w
    object4_pos_x = object4_pos[:, 0]
    object4_pos_y = object4_pos[:, 1]
    object4_pos_z = object4_pos[:, 2]

    # Reward for Object3 being close to Object4 (proximity reward)
    # This uses relative distances between Object3 and Object4.
    distance_object3_to_object4_x = object3_pos_x - object4_pos_x
    distance_object3_to_object4_y = object3_pos_y - object4_pos_y
    # Object3 is a 0.5m cube, so its center should be 0.25m above the platform's surface.
    distance_object3_to_object4_z = torch.abs(object3_pos_z - (object4_pos_z + 0.25))

    proximity_distance_xy = torch.sqrt(distance_object3_to_object4_x**2 + distance_object3_to_object4_y**2)

    # Use Gaussian-like rewards for smooth, continuous values, peaking at 0.5.
    proximity_reward_xy = torch.exp(-2.0 * proximity_distance_xy**2) * 0.5
    proximity_reward_z = torch.exp(-10.0 * distance_object3_to_object4_z**2) * 0.5
    proximity_reward = proximity_reward_xy + proximity_reward_z

    # Total primary reward (proximity only)
    reward = proximity_reward

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_to_object3_and_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_to_object3_and_height_reward") -> torch.Tensor:
    """
    Shaping reward encouraging the robot's pelvis to be close to Object3 for pushing
    and maintaining a stable height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object3 = env.scene['Object3']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]
    pelvis_pos_z = pelvis_pos[:, 2]

    # Current position of Object3
    object3_pos_x = object3.data.root_pos_w[:, 0]
    object3_pos_y = object3.data.root_pos_w[:, 1]

    # Distance between pelvis and Object3 (focus on XY plane for approach)
    # This uses relative distances between pelvis and Object3.
    distance_pelvis_to_object3_x = pelvis_pos_x - object3_pos_x
    distance_pelvis_to_object3_y = pelvis_pos_y - object3_pos_y
    distance_pelvis_to_object3_xy = torch.sqrt(distance_pelvis_to_object3_x**2 + distance_pelvis_to_object3_y**2)

    # Reward for pelvis being close to Object3
    # Optimal pushing distance for a 0.5m cube might be around 0.3-0.5m from its center.
    # Using a Gaussian-like reward centered at 0.4m.
    optimal_push_distance = 0.4
    reward_pelvis_proximity = torch.exp(-5.0 * (distance_pelvis_to_object3_xy - optimal_push_distance)**2) * 0.3

    # Reward for pelvis maintaining a stable height
    # This uses the absolute z-position of the pelvis, which is allowed for height.
    target_pelvis_z = 0.7
    reward_pelvis_height = torch.exp(-10.0 * (pelvis_pos_z - target_pelvis_z)**2) * 0.3

    reward = reward_pelvis_proximity + reward_pelvis_height

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_and_overshoot_penalty(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_and_overshoot_penalty") -> torch.Tensor:
    """
    Shaping reward penalizing collisions with Object1, Object2, and overshooting Object4.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']
    robot = env.scene["robot"]

    # Access robot parts for collision checking
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_palm_idx = robot.body_names.index('left_palm_link')
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx]
    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx]
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx]

    # Define a small buffer for collision avoidance. Cube half-size is 0.25m.
    collision_buffer = 0.05 # meters, for a slight margin around the cube's physical boundary

    robot_parts_to_check = [pelvis_pos, left_palm_pos, right_palm_pos, left_ankle_pos, right_ankle_pos]
    objects_to_avoid = [object1, object2] # Avoid other cubes

    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Penalty for robot parts colliding with Object1 or Object2 (other cubes)
    for part_pos in robot_parts_to_check:
        for obj in objects_to_avoid:
            obj_pos_x = obj.data.root_pos_w[:, 0]
            obj_pos_y = obj.data.root_pos_w[:, 1]
            obj_pos_z = obj.data.root_pos_w[:, 2]

            # Calculate relative distances for collision detection
            dist_x = torch.abs(part_pos[:, 0] - obj_pos_x)
            dist_y = torch.abs(part_pos[:, 1] - obj_pos_y)
            dist_z = torch.abs(part_pos[:, 2] - obj_pos_z)

            # Cube half-dimension is 0.25m.
            # Penalty increases as distance decreases below the threshold (0.25 + buffer).
            # Use negative exponential for continuous penalty.
            threshold_distance = 0.25 + collision_buffer

            # Penalty for XY proximity
            # This uses relative distances.
            # The original plan had separate penalties for dist_x and dist_y, which is fine.
            # However, a combined XY distance is often more robust for collision.
            # Let's stick to the original plan's separate checks for x, y, z for consistency with the prompt's guidance
            # "You must consider the x, y and z components of distances seperately"
            penalty_x = torch.where(dist_x < threshold_distance,
                                     -torch.exp(-5.0 * (dist_x - threshold_distance)**2),
                                     torch.zeros_like(dist_x)) * 0.05
            penalty_y = torch.where(dist_y < threshold_distance,
                                      -torch.exp(-5.0 * (dist_y - threshold_distance)**2),
                                      torch.zeros_like(dist_y)) * 0.05

            # Penalty for Z proximity
            # This uses relative distances.
            penalty_z = torch.where(dist_z < threshold_distance,
                                    -torch.exp(-5.0 * (dist_z - threshold_distance)**2),
                                    torch.zeros_like(dist_z)) * 0.05

            collision_penalty += (penalty_x + penalty_y + penalty_z) # Summing individual axis penalties

    # Penalty for Object3 being pushed past Object4 (overshooting)
    # Object4 (platform) is at x=2m, y=2m (center). It's a 2x2m platform.
    # Object3 should ideally land on the platform.
    # Assuming the general push direction is towards positive X and positive Y.
    object3_pos_x = object3.data.root_pos_w[:, 0]
    object3_pos_y = object3.data.root_pos_w[:, 1]
    object4_pos_x = object4.data.root_pos_w[:, 0]
    object4_pos_y = object4.data.root_pos_w[:, 1]

    overshoot_penalty = torch.zeros(env.num_envs, device=env.device)
    # Define the "far edge" of the platform for overshooting.
    # Platform half-size is 1.0m. So, far edge is center + 1.0m. Add a small buffer.
    # The prompt's plan used 0.5m past center, which is within the platform.
    # To penalize overshooting the *edge* of the platform, it should be object4_pos + 1.0 (half-size) + buffer.
    # Let's use the plan's 0.5m past center as it implies a desired landing zone.
    overshoot_threshold_x = object4_pos_x + 0.5 # 0.5m past platform center in X
    overshoot_threshold_y = object4_pos_y + 0.5 # 0.5m past platform center in Y

    # Penalty if Object3's X position is greater than the overshoot threshold
    # This uses relative distances for the penalty calculation.
    overshoot_x_condition = object3_pos_x > overshoot_threshold_x
    overshoot_penalty_x = torch.where(overshoot_x_condition,
                                      -(object3_pos_x - overshoot_threshold_x) * 0.2, # Linear penalty
                                      torch.zeros_like(object3_pos_x))

    # Penalty if Object3's Y position is greater than the overshoot threshold
    # This uses relative distances for the penalty calculation.
    overshoot_y_condition = object3_pos_y > overshoot_threshold_y
    overshoot_penalty_y = torch.where(overshoot_y_condition,
                                      -(object3_pos_y - overshoot_threshold_y) * 0.2, # Linear penalty
                                      torch.zeros_like(object3_pos_y))

    overshoot_penalty = overshoot_penalty_x + overshoot_penalty_y

    reward = collision_penalty + overshoot_penalty

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
    # Primary reward for pushing Object3 towards Object4
    PushObject3TowardsObject4 = RewTerm(func=push_object3_towards_object4_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "push_object3_towards_object4_reward"})

    # Shaping reward for pelvis proximity to Object3 and stable height
    PelvisToObject3AndHeight = RewTerm(func=pelvis_to_object3_and_height_reward, weight=0.6,
                                       params={"normalise": True, "normaliser_name": "pelvis_to_object3_and_height_reward"})

    # Shaping reward for collision avoidance with other cubes and preventing overshooting the platform
    CollisionAndOvershootPenalty = RewTerm(func=collision_and_overshoot_penalty, weight=0.4,
                                           params={"normalise": True, "normaliser_name": "collision_and_overshoot_penalty"})