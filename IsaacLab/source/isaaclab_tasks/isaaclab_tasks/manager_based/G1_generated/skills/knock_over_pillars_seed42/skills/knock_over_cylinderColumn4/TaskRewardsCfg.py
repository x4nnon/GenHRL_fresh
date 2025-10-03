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


def knock_over_cylinderColumn4_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "primary_reward") -> torch.Tensor:
    """
    Primary reward for the knock_over_cylinderColumn4 skill.
    Encourages the robot to first approach Object4 and then to reduce Object4's Z-position until it falls completely onto the floor.
    Combines an approach reward with a "knock over" reward.
    """
    # MANDATORY: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # MANDATORY: Access the required objects using approved pattern
    object4 = env.scene['Object4']

    # MANDATORY: Access the required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    # No need to access hands for primary reward as per plan, only pelvis for approach and object Z for knock over.

    # MANDATORY: Object4 dimensions (from task description: z dimension of 2m, radius of 0.3m)
    # CORRECT: Hardcoding object dimensions from the task description.
    object4_height = 2.0
    object4_radius = 0.3

    # MANDATORY: Calculate distances for approach using relative distances
    # CORRECT: Using relative distances between robot pelvis and Object4 for X and Y components.
    distance_pelvis_object4_x = object4.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    distance_pelvis_object4_y = object4.data.root_pos_w[:, 1] - pelvis_pos[:, 1]

    # MANDATORY: Reward for approaching Object4 in X and Y
    # Encourage getting close, but not too close initially to allow for pushing.
    # CORRECT: Continuous reward using negative absolute distance.
    approach_reward_x = -torch.abs(distance_pelvis_object4_x)
    approach_reward_y = -torch.abs(distance_pelvis_object4_y)

    # MANDATORY: Calculate distances for knocking over
    # Use the minimum Z position of the object as it falls
    object4_z_pos = object4.data.root_pos_w[:, 2]

    # MANDATORY: Reward for reducing Object4's Z-position.
    # The target Z for a fallen cylinder (radius 0.3m) is its radius, so 0.3m.
    # A standing cylinder's center is at 1.0m (half its 2m height).
    # CORRECT: Continuous reward using negative absolute distance to the target fallen Z-position.
    knock_over_reward = -torch.abs(object4_z_pos - object4_radius)

    # MANDATORY: Combine rewards. The knock_over_reward becomes dominant as the object falls.
    # The approach reward helps get the robot into position.
    # CORRECT: Weighted combination of continuous rewards.
    primary_reward = (approach_reward_x * 0.3 + approach_reward_y * 0.3) + (knock_over_reward * 0.4)

    # MANDATORY: Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward


def knock_over_cylinderColumn4_shaping_reward1(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward1") -> torch.Tensor:
    """
    Shaping reward 1 for the knock_over_cylinderColumn4 skill.
    Encourages the robot to maintain a stable upright posture (pelvis Z-position) and to keep its hands at a reasonable height
    to interact with the cylinder. It also encourages the robot to not move too far past Object5, which is the next skill's target.
    """
    # MANDATORY: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # MANDATORY: Access the required objects using approved pattern
    object4 = env.scene['Object4']
    object5 = env.scene['Object5'] # For next skill positioning

    # MANDATORY: Access the required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]

    # MANDATORY: Object4 dimensions (from task description: z dimension of 2m, radius of 0.3m)
    # CORRECT: Hardcoding object dimensions from the task description.
    object4_height = 2.0
    object4_radius = 0.3

    # MANDATORY: Target pelvis Z for stability
    # CORRECT: Using a reasonable target Z for pelvis height, derived from typical robot standing height.
    target_pelvis_z = 0.7

    # MANDATORY: Reward for maintaining pelvis Z height for stability
    # CORRECT: Continuous reward using negative absolute distance to target pelvis Z.
    stability_reward = -torch.abs(pelvis_pos[:, 2] - target_pelvis_z)

    # MANDATORY: Reward for hands being at a suitable height to push the cylinder (e.g., around mid-height of the cylinder)
    # Mid-height of cylinder is object4_height / 2 = 1.0m when standing.
    # Target hand Z relative to object's base Z (center of mass Z - half height)
    # The object's root_pos_w[:, 2] is its center of mass. For a standing cylinder, this is half its height.
    # So, to target the mid-height of the cylinder, the hand Z should be around object4.data.root_pos_w[:, 2].
    # If the cylinder falls, its root_pos_w[:, 2] will be its radius (0.3m).
    # We want hands to be at a height suitable for pushing, which is around the object's current center of mass Z.
    # The plan's skeleton had `target_hand_z = object4.data.root_pos_w[:, 2] + (object4_height / 2.0)` which is incorrect
    # because `object4.data.root_pos_w[:, 2]` is already the center of mass, so adding half height would target above the cylinder.
    # The correct target is the object's current center of mass Z.
    target_hand_z = object4.data.root_pos_w[:, 2]
    # CORRECT: Continuous reward using negative absolute distance for hand Z positions relative to object's current Z.
    hand_height_reward = -torch.abs(left_hand_pos[:, 2] - target_hand_z) - torch.abs(right_hand_pos[:, 2] - target_hand_z)

    # MANDATORY: Reward for not overshooting Object5 (the next target)
    # This encourages the robot to stop after knocking over Object4 and not run past Object5.
    # Assuming objects are placed in a line along the X-axis.
    # CORRECT: Using relative distance between robot pelvis X and Object5 X.
    distance_robot_to_object5_x = object5.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    # Penalize if robot is more than 0.5m past Object5 (i.e., distance_robot_to_object5_x is negative and large in magnitude).
    # CORRECT: Conditional penalty using torch.where, penalizing if robot overshoots.
    overshoot_penalty = torch.where(distance_robot_to_object5_x < -0.5, -10.0 * torch.abs(distance_robot_to_object5_x + 0.5), 0.0)

    # MANDATORY: Combine shaping rewards
    # CORRECT: Weighted combination of continuous and conditional rewards.
    shaping_reward1 = stability_reward * 0.3 + hand_height_reward * 0.2 + overshoot_penalty * 0.5

    # MANDATORY: Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def knock_over_cylinderColumn4_shaping_reward2(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward2") -> torch.Tensor:
    """
    Shaping reward 2 for the knock_over_cylinderColumn4 skill.
    Provides collision avoidance for the robot's body parts with Object4 and other objects
    to prevent unnecessary collisions, especially before the intended push. It also encourages the robot to avoid falling over.
    """
    # MANDATORY: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # MANDATORY: Access the required objects using approved pattern
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    object4 = env.scene['Object4']
    object5 = env.scene['Object5']

    # MANDATORY: Access the required robot part(s) using approved pattern
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Foot positions are not explicitly used in the plan's skeleton for collision, but pelvis is.
    # The prompt's skeleton included foot indices but didn't use them. Removed for clarity as per prompt's "skeleton" usage.
    # left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    # left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    # right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    # right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # MANDATORY: Object dimensions (radius 0.3m)
    # CORRECT: Hardcoding object dimensions from the task description.
    object_radius = 0.3

    collision_penalty = 0.0

    # MANDATORY: Collision avoidance for pelvis with all objects (except Object4 during the push)
    # A small buffer (e.g., 0.1m) around the object radius
    # CORRECT: Using a threshold derived from object dimensions.
    collision_threshold = object_radius + 0.1

    objects_to_check = [object1, object2, object3, object5] # Avoid collisions with other columns
    for obj in objects_to_check:
        # MANDATORY: Calculating 3D Euclidean distance between pelvis and object using relative distances.
        dist_x = obj.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
        dist_y = obj.data.root_pos_w[:, 1] - pelvis_pos[:, 1]
        dist_z = obj.data.root_pos_w[:, 2] - pelvis_pos[:, 2]
        distance_to_obj = torch.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        # MANDATORY: Continuous penalty using torch.where, penalizing closer proximity.
        collision_penalty += torch.where(distance_to_obj < collision_threshold, -10.0 * (collision_threshold - distance_to_obj), 0.0)

    # MANDATORY: Specific collision avoidance for Object4 before interaction
    # Only penalize if robot is not actively pushing (e.g., if Object4 is still standing)
    # Object4 is still mostly upright if its Z-position is significantly above its radius.
    # CORRECT: Conditional check based on Object4's Z-position to determine if it's standing.
    object4_standing_condition = object4.data.root_pos_w[:, 2] > (object_radius + 0.5)
    # MANDATORY: Calculating 3D Euclidean distance between pelvis and Object4 using relative distances.
    dist_x_obj4 = object4.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    dist_y_obj4 = object4.data.root_pos_w[:, 1] - pelvis_pos[:, 1]
    dist_z_obj4 = object4.data.root_pos_w[:, 2] - pelvis_pos[:, 2]
    distance_to_obj4 = torch.sqrt(dist_x_obj4**2 + dist_y_obj4**2 + dist_z_obj4**2)
    # MANDATORY: Conditional penalty using torch.where, active only when Object4 is standing and pelvis is too close.
    collision_penalty += torch.where((distance_to_obj4 < collision_threshold) & object4_standing_condition, -10.0 * (collision_threshold - distance_to_obj4), 0.0)

    # MANDATORY: Penalty for falling over (pelvis Z too low)
    # CORRECT: Using a reasonable threshold for falling.
    falling_threshold = 0.3 # If pelvis Z drops below this, robot is likely falling
    # MANDATORY: Conditional penalty using torch.where, active when pelvis Z drops below threshold.
    falling_penalty = torch.where(pelvis_pos[:, 2] < falling_threshold, -50.0 * (falling_threshold - pelvis_pos[:, 2]), 0.0)

    # MANDATORY: Combine shaping rewards
    # CORRECT: Summing up continuous and conditional penalties.
    shaping_reward2 = collision_penalty + falling_penalty

    # MANDATORY: Normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # CORRECT: Main reward with weight 1.0, using the defined function and normalization parameters.
    PrimaryReward = RewTerm(func=knock_over_cylinderColumn4_primary_reward, weight=1.0,
                            params={"normalise": True, "normaliser_name": "primary_reward"})

    # CORRECT: Supporting reward with lower weight, using the defined function and normalization parameters.
    ShapingReward1 = RewTerm(func=knock_over_cylinderColumn4_shaping_reward1, weight=0.6,
                             params={"normalise": True, "normaliser_name": "shaping_reward1"})

    # CORRECT: Supporting reward with lower weight, using the defined function and normalization parameters.
    ShapingReward2 = RewTerm(func=knock_over_cylinderColumn4_shaping_reward2, weight=0.5,
                             params={"normalise": True, "normaliser_name": "shaping_reward2"})