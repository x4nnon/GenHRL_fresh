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


def position_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "position_large_block_reward") -> torch.Tensor:
    """
    Primary reward for positioning the Large Block (Object3) into its designated final position
    relative to the Medium Block (Object2) to complete the stairs.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    object2 = env.scene['Object2'] # Medium Block
    object3 = env.scene['Object3'] # Large Block

    # Access object positions
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Object dimensions (hardcoded from skill info, as per rules):
    # Object2 (Medium Block): x=1m y=1m z=0.6m
    # Object3 (Large Block): x=1m y=1m z=0.9m

    # Calculate desired relative position for Object3 to be adjacent to Object2 and form a step.
    # Assuming Object3 is placed next to Object2 along the Y-axis, and its top surface is the next step.
    # Desired relative position:
    # x_offset_target = 0.0 (align x-centers)
    # y_offset_target = 1.0m (Object3's center 1m away from Object2's center along y, which is half of Object2's y-dim + half of Object3's y-dim)
    # z_offset_target = 0.0 (Object3's base at same level as Object2's base, assuming both start on ground)

    # Current relative positions (x, y, z components separated as required)
    current_x_offset = object3_pos[:, 0] - object2_pos[:, 0]
    current_y_offset = object3_pos[:, 1] - object2_pos[:, 1]
    current_z_offset = object3_pos[:, 2] - object2_pos[:, 2]

    # Calculate error from target relative position
    # Using torch.abs for continuous, non-negative error
    error_x = torch.abs(current_x_offset - 0.0)
    error_y = torch.abs(current_y_offset - 1.0) # Target 1.0m offset along y-axis
    error_z = torch.abs(current_z_offset - 0.0) # Target 0.0m offset along z-axis (bases aligned)

    # Reward is negative sum of errors, so minimizing error maximizes reward.
    # This creates a continuous reward landscape.
    reward = - (error_x + error_y)

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def approach_large_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_large_block_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach the Large Block (Object3)
    and position its pelvis for an effective push.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object3 = env.scene['Object3'] # Large Block

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    object3_pos = object3.data.root_pos_w

    # Object3 dimensions: x=1m y=1m z=0.9m (hardcoded from skill info)

    # Define target pelvis position relative to Object3 for pushing.
    # Assuming robot pushes from the side where y is smaller than Object3's center y.
    # Target pelvis y: Object3's center y - (half of Object3's y-dimension + a small pushing offset)
    # Half of Object3's y-dimension = 1.0m / 2 = 0.5m
    # Pushing offset = 0.1m (a small clearance for contact)
    target_pelvis_y = object3_pos[:, 1] - (0.5 + 0.1) # 0.6m behind the block's center along y

    # Target pelvis x: Align with Object3's x-center for a straight push.
    target_pelvis_x = object3_pos[:, 0]

    # Target pelvis z: Maintain a reasonable height for pushing and stability.
    # This is an absolute height, allowed sparingly when height is important.
    pelvis_target_z = 0.7 # A common stable pelvis height for the robot

    # Calculate errors for each dimension (x, y, z components separated)
    error_x = torch.abs(pelvis_pos[:, 0] - target_pelvis_x)
    error_y = torch.abs(pelvis_pos[:, 1] - target_pelvis_y)
    error_z = torch.abs(pelvis_pos[:, 2] - pelvis_target_z)

    # Reward is negative sum of errors, providing a continuous gradient for approach.
    reward = - (error_x + error_y + error_z)

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward to penalize the robot for colliding or getting too close to
    Object1 (Small Block) and Object2 (Medium Block) during the task.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    robot = env.scene["robot"]
    object1 = env.scene['Object1'] # Small Block
    object2 = env.scene['Object2'] # Medium Block

    # Access robot part positions
    pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')]
    left_palm_pos = robot.data.body_pos_w[:, robot.body_names.index('left_palm_link')]
    right_palm_pos = robot.data.body_pos_w[:, robot.body_names.index('right_palm_link')]
    left_ankle_pos = robot.data.body_pos_w[:, robot.body_names.index('left_ankle_roll_link')]
    right_ankle_pos = robot.data.body_pos_w[:, robot.body_names.index('right_ankle_roll_link')]

    # Object dimensions (hardcoded from skill info):
    # Object1 (Small Block): x=1m y=1m z=0.3m
    # Object2 (Medium Block): x=1m y=1m z=0.6m

    # Define a safe distance threshold for collision avoidance.
    # This is a relative threshold, representing the minimum allowed distance.
    safe_distance_threshold = 0.5 # meters

    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Iterate through relevant objects and robot parts to check for proximity
    # Using a list of (object, object_half_dimensions) for generalizability
    # Note: object_half_dimensions are not directly used in this simple spherical distance check,
    # but are included for context if a more complex AABB collision check were implemented.
    objects_to_avoid = [
        (object1, torch.tensor([0.5, 0.5, 0.15], device=env.device)), # Half dims for Object1 (1m, 1m, 0.3m)
        (object2, torch.tensor([0.5, 0.5, 0.3], device=env.device))  # Half dims for Object2 (1m, 1m, 0.6m)
    ]
    robot_parts_to_check = [pelvis_pos, left_palm_pos, right_palm_pos, left_ankle_pos, right_ankle_pos]

    for obj, _ in objects_to_avoid:
        obj_pos = obj.data.root_pos_w
        for part_pos in robot_parts_to_check:
            # Calculate Euclidean distance between robot part and object center
            # This is a relative distance calculation.
            distance_to_obj_center = torch.norm(part_pos - obj_pos, dim=1)

            # Penalize if distance is less than the safe_distance_threshold.
            # The penalty is continuous: max penalty when distance is 0, 0 penalty when distance >= threshold.
            # Using a linear penalty based on penetration depth.
            penetration_depth = torch.relu(safe_distance_threshold - distance_to_obj_center)
            collision_penalty -= penetration_depth # Accumulate penalty

    reward = collision_penalty # Reward is negative (penalty)

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
    """
    Reward terms for the Position_Large_Block_for_robot_interaction skill.
    """
    # Primary reward for positioning the large block
    position_large_block_reward = RewTerm(
        func=position_large_block_reward,
        weight=0.0, # High weight for the main objective
        params={"normalise": True, "normaliser_name": "position_large_block_reward"}
    )

    # Shaping reward for approaching the large block
    approach_large_block_reward = RewTerm(
        func=approach_large_block_reward,
        weight=0, # Moderate weight to guide the robot towards the pushing position
        params={"normalise": True, "normaliser_name": "approach_large_block_reward"}
    )

    # Shaping reward for collision avoidance with other blocks
    collision_avoidance_reward = RewTerm(
        func=collision_avoidance_reward,
        weight=0, # Significant weight to prevent unwanted collisions
        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"}
    )