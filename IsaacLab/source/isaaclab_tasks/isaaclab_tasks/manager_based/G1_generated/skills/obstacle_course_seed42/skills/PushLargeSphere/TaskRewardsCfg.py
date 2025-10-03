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


def push_large_sphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "push_large_sphere_reward") -> torch.Tensor:
    """
    Primary reward for pushing the large sphere (Object1) towards the high wall (Object4).
    This reward encourages Object1's x-position to increase towards Object4's x-position,
    and Object1's y and z positions to align with Object4's y and z positions.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access object positions using approved patterns
    object1 = env.scene['Object1']
    object4 = env.scene['Object4']

    object1_pos = object1.data.root_pos_w
    object4_pos = object4.data.root_pos_w

    # Hardcoded dimensions from object configuration (Object1: radius 1m, Object4: 0.3m x-depth)
    object1_radius = 1.0
    object4_depth = 0.3

    # Calculate target x-position for Object1's center:
    # The sphere's front edge should ideally touch the wall's front edge.
    # Wall's front edge is at object4_pos_x - (object4_depth / 2).
    # Sphere's center should be at (wall's front edge) - sphere_radius.
    target_object1_x = object4_pos[:, 0] - (object4_depth / 2) - object1_radius

    # Reward for Object1's x-position approaching the target_object1_x.
    # We want to minimize the absolute difference between current x and target x.
    # The reward should be higher when Object1 is closer to the target.
    # Normalize by a reasonable maximum distance (e.g., initial separation between Object1 and Object4 is 3m).
    # The effective distance for the sphere's center to the wall's front edge is roughly 3m - 1m (radius) - 0.15m (half wall depth) = 1.85m.
    # Let's use a max_x_dist of 2.0 for normalization, allowing for some initial distance.
    max_x_dist = 2.0
    x_distance = torch.abs(object1_pos[:, 0] - target_object1_x)
    reward_x = -x_distance / max_x_dist # Reward is maximized (closest to 0) when x_distance is 0.

    # Reward for Object1's y-position aligning with Object4's y-position.
    # Object4 (High wall) is 5m wide in y, so a larger tolerance is needed.
    # Object1's y should be close to Object4's y.
    y_distance = torch.abs(object1_pos[:, 1] - object4_pos[:, 1])
    # Normalize by half the wall's width (2.5m) as a reasonable max deviation.
    reward_y = -y_distance / 2.5

    # Reward for Object1's z-position aligning with its natural resting height (its radius).
    # Object1 radius is 1m, so its center z should be 1m when resting on the ground.
    z_distance = torch.abs(object1_pos[:, 2] - object1_radius)
    # Normalize by its radius.
    reward_z = -z_distance / object1_radius

    # Combine rewards with adjusted weights
    reward = reward_x * 0.6 + reward_y * 0.2 + reward_z * 0.2

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_approach_sphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "robot_approach_sphere_reward") -> torch.Tensor:
    """
    Supporting reward for the robot's pelvis moving towards the large sphere (Object1).
    This encourages the robot to get into a position to push the sphere.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot pelvis and Object1 positions using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    object1_pos = object1.data.root_pos_w

    # Hardcoded dimensions from object configuration (Object1: radius 1m)
    object1_radius = 1.0

    # Calculate target x-position for robot's pelvis:
    # The robot's pelvis should be slightly behind the sphere's surface to push it.
    # Let's aim for the pelvis to be 0.5m behind the sphere's surface.
    # Sphere's surface in x is object1_pos_x - object1_radius.
    # So, target pelvis x = (object1_pos_x - object1_radius) - 0.5.
    target_pelvis_x = object1_pos[:, 0] - object1_radius - 0.5
    
    # Distance in x: minimize distance to target pelvis x
    x_distance = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    # Max initial distance from robot to Object1 is roughly 3m.
    # Normalize by a max expected distance, e.g., 3.0m.
    reward_x = -x_distance / 3.0

    # Distance in y: minimize distance to Object1's y.
    # The robot should align with the sphere's y-position.
    y_distance = torch.abs(pelvis_pos[:, 1] - object1_pos[:, 1])
    # Normalize by a reasonable max y-deviation, e.g., 1.0m.
    reward_y = -y_distance / 1.0

    # Combine rewards
    reward = reward_x * 0.7 + reward_y * 0.3

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_hands_near_sphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "robot_hands_near_sphere_reward") -> torch.Tensor:
    """
    Supporting reward for the robot's hands being close to the large sphere (Object1).
    This encourages the robot to use its hands for pushing.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot hand positions and Object1 position using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1']

    left_hand_idx = robot.body_names.index('left_palm_link')
    right_hand_idx = robot.body_names.index('right_palm_link')

    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    object1_pos = object1.data.root_pos_w

    # Hardcoded dimensions from object configuration (Object1: radius 1m)
    object1_radius = 1.0

    # Calculate the average position of the hands
    avg_hand_pos = (left_hand_pos + right_hand_pos) / 2.0

    # Target position for hands relative to sphere surface:
    # Hands should be at or slightly inside the sphere's surface for pushing.
    # Let's aim for hands to be 0.1m inside the sphere's surface in x.
    # Sphere's surface in x is object1_pos_x - object1_radius.
    # So, target hand x = (object1_pos_x - object1_radius) + 0.1.
    target_hand_x = object1_pos[:, 0] - object1_radius + 0.1
    # Hands should align with sphere's y and z center.
    target_hand_y = object1_pos[:, 1]
    target_hand_z = object1_pos[:, 2] # Sphere center z is 1.0 (radius)

    # Calculate distance to target hand position for each dimension.
    # Normalize by a reasonable distance, e.g., 1.0m.
    x_dist_hands = torch.abs(avg_hand_pos[:, 0] - target_hand_x)
    reward_x = -x_dist_hands / 1.0

    y_dist_hands = torch.abs(avg_hand_pos[:, 1] - target_hand_y)
    reward_y = -y_dist_hands / 1.0

    z_dist_hands = torch.abs(avg_hand_pos[:, 2] - target_hand_z)
    reward_z = -z_dist_hands / 1.0

    # Combine rewards
    reward = reward_x * 0.4 + reward_y * 0.3 + reward_z * 0.3

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def wall_toppled_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "wall_toppled_reward") -> torch.Tensor:
    """
    Supporting reward for the high wall (Object4) being pushed over.
    This is a sparse-like reward that becomes active and increases as the wall's z-position changes.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access Object4's position using approved patterns
    object4 = env.scene['Object4']
    object4_pos_z = object4.data.root_pos_w[:, 2]

    # Hardcoded dimensions from object configuration (Object4: 1m z-height, 0.3m x-depth)
    object4_height = 1.0
    object4_depth = 0.3

    # Initial z-position of the wall's center when upright is half its height.
    object4_initial_z_center = object4_height / 2.0 # 0.5m

    # When the wall topples, its center z-position will drop.
    # If it lies flat on its side (along x-axis), its new z-center will be half its depth.
    object4_toppled_z_center = object4_depth / 2.0 # 0.15m

    # Reward for the wall's z-position dropping.
    # The reward should be higher as object4_pos_z decreases from its initial height towards its toppled height.
    # Use a linear reward that maxes out when z is at or below the toppled height.
    # Reward = (initial_z - current_z) / (initial_z - toppled_z)
    # Clamp to 0-1 range to ensure it's only positive when dropping and bounded.
    reward = torch.clamp((object4_initial_z_center - object4_pos_z) / (object4_initial_z_center - object4_toppled_z_center), min=0.0, max=1.0)

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "robot_stability_reward") -> torch.Tensor:
    """
    Supporting reward for maintaining robot stability (pelvis height).
    Encourages the robot to stay upright during the pushing action.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot pelvis position using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos_z = robot.data.body_pos_w[:, pelvis_idx, 2]

    # Target pelvis height for a standing robot (e.g., 0.7m is a common stable height for humanoid robots)
    target_pelvis_height = 0.7

    # Reward for pelvis height being close to the target height.
    # Use a negative absolute difference.
    height_deviation = torch.abs(pelvis_pos_z - target_pelvis_height)
    # Normalize by a reasonable maximum deviation, e.g., 0.5m (if pelvis can drop from 0.7 to 0.2 or rise to 1.2)
    reward = -height_deviation / 0.5

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
    # Primary reward for pushing the large sphere towards the high wall
    PushLargeSphereReward = RewTerm(func=push_large_sphere_reward, weight=1.0,
                                    params={"normalise": True, "normaliser_name": "push_large_sphere_reward"})

    # Supporting reward for the robot approaching the large sphere
    RobotApproachSphereReward = RewTerm(func=robot_approach_sphere_reward, weight=0.3,
                                        params={"normalise": True, "normaliser_name": "robot_approach_sphere_reward"})

    # Supporting reward for robot's hands being near the large sphere for pushing
    RobotHandsNearSphereReward = RewTerm(func=robot_hands_near_sphere_reward, weight=0.2,
                                         params={"normalise": True, "normaliser_name": "robot_hands_near_sphere_reward"})

    # Supporting reward for the high wall being toppled
    WallToppledReward = RewTerm(func=wall_toppled_reward, weight=0.5,
                                params={"normalise": True, "normaliser_name": "wall_toppled_reward"})

    # Supporting reward for maintaining robot stability
    RobotStabilityReward = RewTerm(func=robot_stability_reward, weight=0.1,
                                   params={"normalise": True, "normaliser_name": "robot_stability_reward"})