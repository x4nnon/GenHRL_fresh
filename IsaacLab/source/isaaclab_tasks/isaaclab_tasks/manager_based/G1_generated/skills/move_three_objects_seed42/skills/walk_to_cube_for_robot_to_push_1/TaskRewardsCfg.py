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


def main_walk_to_cube_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the robot to walk to a position near Object1 (Cube for robot to push)
    to prepare for pushing. The goal is for the robot's pelvis to be positioned appropriately
    behind Object1, aligned in Y, and at a stable Z height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object1 = env.scene['Object1'] # Cube for robot to push
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern

    # Get positions
    object1_pos = object1.data.root_pos_w # CORRECT: Accessing object position using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern

    # Cube dimensions (from object configuration: 0.5m cubed block)
    # CORRECT: Hardcoding object dimensions from the configuration, as direct access is not allowed.
    cube_half_size = 0.25 # 0.5m / 2

    # Desired X position relative to cube's center for pushing
    # Robot's pelvis should be slightly behind the cube.
    # Assuming robot pushes in positive X direction, pelvis X should be less than cube X.
    # Target X: 0.4m behind the cube's center.
    target_pelvis_x = object1_pos[:, 0] - 0.5 # CORRECT: Relative target position based on object's X

    # Desired Y position relative to cube's center
    # Target Y: Aligned with the cube's Y center.
    target_pelvis_y = object1_pos[:, 1] # CORRECT: Relative target position based on object's Y

    # Desired Z position for stable pelvis height
    # Target Z: Fixed height for stable posture.
    target_pelvis_z = 0.7 # CORRECT: Absolute Z target for posture, allowed for height control

    # Calculate distances to target position components
    dist_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x) # CORRECT: Relative distance in X
    dist_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y) # CORRECT: Relative distance in Y
    dist_z = torch.abs(pelvis_pos[:, 2] - target_pelvis_z) # CORRECT: Relative distance in Z

    # Combine distances for primary reward. Use negative exponential for continuous positive reward.
    # Smaller distances yield higher rewards.
    # The reward is designed to be continuous and negative, encouraging the agent to minimize the distances.
    # The negative sign ensures that smaller distances result in higher (less negative) rewards.
    # The scaling factor (e.g., 2.0) can be adjusted to control the reward's sensitivity.
    primary_reward = - (dist_x * 2.0 + dist_y * 2.0 + dist_z) # Emphasize Z for stable height

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, primary_reward)
        RewNormalizer.update_stats(normaliser_name, primary_reward)
        return scaled_reward
    return primary_reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    This reward encourages the robot to avoid collisions with Object1, Object2, and Object3.
    It provides a negative reward if any part of the robot (pelvis, hands, feet) gets too close
    or collides with any of the cubes.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object1 = env.scene['Object1']
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern

    # Get indices for robot parts
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    left_palm_idx = robot.body_names.index('left_palm_link') # CORRECT: Accessing robot part index using approved pattern
    right_palm_idx = robot.body_names.index('right_palm_link') # CORRECT: Accessing robot part index using approved pattern
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link') # CORRECT: Accessing robot part index using approved pattern
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link') # CORRECT: Accessing robot part index using approved pattern

    # Get positions for robot parts
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    left_palm_pos = robot.data.body_pos_w[:, left_palm_idx] # CORRECT: Accessing robot part position using approved pattern
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx] # CORRECT: Accessing robot part position using approved pattern
    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx] # CORRECT: Accessing robot part position using approved pattern
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx] # CORRECT: Accessing robot part position using approved pattern

    # Cube dimensions (0.5m cubed block)
    # CORRECT: Hardcoding object dimensions from the configuration.
    cube_half_size = 0.25
    # Robot body part approximate radius/clearance
    robot_part_clearance = 0.15 # A general radius for robot parts

    # Threshold for collision avoidance (sum of cube half size and robot part clearance)
    collision_threshold = cube_half_size + robot_part_clearance # CORRECT: Threshold derived from object size and robot clearance

    # List of objects to check against
    objects_to_check = [object1, object2, object3]
    # List of robot parts to check
    robot_parts_to_check = [pelvis_pos, left_palm_pos, right_palm_pos, left_ankle_pos, right_ankle_pos]

    collision_reward = torch.zeros_like(pelvis_pos[:, 0]) # Initialize reward tensor for batch processing

    for obj in objects_to_check:
        obj_pos = obj.data.root_pos_w # CORRECT: Accessing object position using approved pattern
        for part_pos in robot_parts_to_check:
            # Calculate Euclidean distance between robot part and object center
            distance_to_object = torch.norm(obj_pos - part_pos, dim=1) # CORRECT: Relative Euclidean distance

            # Negative reward if too close or colliding
            # The reward is continuous and gets more negative as the distance decreases below the threshold.
            # (distance_to_object - collision_threshold) will be negative when distance < threshold.
            # Multiplying by a negative scale factor (e.g., 5.0) makes it a penalty.
            collision_penalty = torch.where(
                distance_to_object < collision_threshold,
                (distance_to_object - collision_threshold) * 5.0, # Scale factor to make penalty significant
                torch.tensor(0.0, device=env.device) # No penalty if outside threshold
            )
            collision_reward += collision_penalty # Accumulate penalties from all parts and objects

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, collision_reward)
        RewNormalizer.update_stats(normaliser_name, collision_reward)
        return scaled_reward
    return collision_reward

def upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain an upright and stable posture,
    by penalizing large deviations of the pelvis Z-position from the target height (0.7m).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part using approved patterns
    robot = env.scene["robot"] # CORRECT: Accessing robot using approved pattern
    pelvis_idx = robot.body_names.index('pelvis') # CORRECT: Accessing robot part index using approved pattern
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # CORRECT: Accessing robot part position using approved pattern
    pelvis_pos_z = pelvis_pos[:, 2]

    # Desired Z position for stable pelvis height
    target_pelvis_z = 0.7 # CORRECT: Absolute Z target for posture, allowed for height control

    # Penalty for deviation from target pelvis height
    # Use absolute error for a smooth, increasing penalty as deviation grows.
    # The reward is continuous and negative, getting worse as the pelvis Z deviates from the target.
    height_deviation_penalty = -torch.abs(pelvis_pos_z - target_pelvis_z) * 2.0 # Scale factor for penalty

    # Mandatory normalization implementation
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, height_deviation_penalty)
        RewNormalizer.update_stats(normaliser_name, height_deviation_penalty)
        return scaled_reward
    return height_deviation_penalty


@configclass
class TaskRewardsCfg:
    """
    Configuration for the reward terms used in the walk_to_cube_for_robot_to_push_1 skill.
    """
    # Main reward for reaching the target position relative to Object1
    MainWalkToCubeReward = RewTerm(func=main_walk_to_cube_reward, weight=1.0,
                                   params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for avoiding collisions with cubes
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                       params={"normalise": True, "normaliser_name": "collision_reward"})

    # Shaping reward for maintaining an upright posture
    UprightPostureReward = RewTerm(func=upright_posture_reward, weight=0.2,
                                   params={"normalise": True, "normaliser_name": "posture_reward"})