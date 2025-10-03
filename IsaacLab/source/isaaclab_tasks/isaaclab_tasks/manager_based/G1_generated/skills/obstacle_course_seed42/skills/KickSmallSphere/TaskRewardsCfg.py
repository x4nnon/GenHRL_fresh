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


def kick_small_sphere_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "kick_small_sphere_main_reward") -> torch.Tensor:
    """
    Main reward for the KickSmallSphere skill.
    Encourages the robot to approach the small sphere (Object2) and then kick it away
    in the positive X direction, without going past the block cube (Object5).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    robot = env.scene["robot"]
    small_sphere = env.scene['Object2'] # Reasoning: Accessing object using approved pattern.
    block_cube = env.scene['Object5'] # Reasoning: Accessing object using approved pattern.

    pelvis_idx = robot.body_names.index('pelvis') # Reasoning: Accessing robot part index using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Reasoning: Accessing robot part position using approved pattern.
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # Object dimensions (hardcoded from object configuration)
    # Reasoning: Object dimensions are hardcoded from the provided object configuration, as per rules.
    small_sphere_radius = 0.2
    block_cube_x_size = 0.5

    # Phase 1: Approach small sphere (Object2)
    # Reward for reducing X distance between pelvis and small sphere.
    # Target X for approach: slightly behind the sphere's center, to prepare for a kick.
    # Reasoning: Uses relative distance between robot pelvis and sphere. Continuous reward.
    target_approach_x = small_sphere.data.root_pos_w[:, 0] - small_sphere_radius - 0.1 # Target just behind the sphere
    approach_reward_x = -torch.abs(pelvis_pos_x - target_approach_x) # Reasoning: Uses relative distance, continuous.

    # Reward for aligning Y distance between pelvis and small sphere.
    # Reasoning: Uses relative distance between robot pelvis and sphere. Continuous reward.
    approach_reward_y = -torch.abs(pelvis_pos_y - small_sphere.data.root_pos_w[:, 1]) # Reasoning: Uses relative distance, continuous.

    # Phase 2: Kick small sphere (Object2) away
    # Reward for moving the sphere in positive X direction.
    # The reward is for the sphere's X position being greater than its initial X position.
    # Since we cannot store initial positions, we reward for the sphere moving *past* a reference point
    # that is its initial spawn X position (which is fixed in the environment setup).
    # The environment setup states: "small sphere 0.2m radius. and a block cube of 0.5m cubed.
    # These objects should be positioned along the x axis in a line, with 3m separating each.
    # The order should be: 1) Low wall, 2) large sphere, 3) high wall, 4) small sphere and 5) block.
    # The robot will start at 0,0,0 and the first object (low wall) should be 3m away in the x axis."
    # This implies a fixed initial X for the small sphere.
    # Based on the description: Low wall (3m), Large sphere (3m from low wall), High wall (3m from large sphere),
    # Small sphere (3m from high wall), Block (3m from small sphere).
    # So, initial_small_sphere_x = 3 + 3 + 3 + 3 = 12m.
    # Reasoning: This uses a fixed reference point derived from the environment setup, which acts as a proxy for "initial position"
    # without storing state. It's a relative distance from a known world point.
    initial_small_sphere_x_reference = 12.0 # Derived from environment setup description

    sphere_x_pos = small_sphere.data.root_pos_w[:, 0]
    block_x_pos = block_cube.data.root_pos_w[:, 0]

    # Reward for sphere moving past its initial reference X position
    # This encourages positive displacement.
    kick_progress_reward = (sphere_x_pos - initial_small_sphere_x_reference) * 2.0
    # Reasoning: Continuous reward based on displacement from a fixed reference.

    # Penalty if sphere goes too far (past max_kick_x)
    # Max desired kicked X: just before Object5's front edge.
    # Reasoning: Uses relative distance to Object5. Continuous penalty.
    max_kick_x = block_x_pos - block_cube_x_size / 2.0 - small_sphere_radius - 0.1
    overshoot_penalty = torch.where(sphere_x_pos > max_kick_x, -5.0 * (sphere_x_pos - max_kick_x), 0.0)

    # Penalty if sphere is not kicked far enough (before min_kick_x)
    # Min desired kicked X: at least 0.5m from its initial reference position.
    # Reasoning: Uses relative distance to a fixed reference. Continuous penalty.
    min_kick_x = initial_small_sphere_x_reference + 0.5
    underkick_penalty = torch.where(sphere_x_pos < min_kick_x, -2.0 * (min_kick_x - sphere_x_pos), 0.0)

    # Combine kick rewards and penalties
    kick_reward = kick_progress_reward + overshoot_penalty + underkick_penalty

    # Activation condition for kick reward: robot is close to the sphere in X and Y
    # Reasoning: Uses relative distances between robot pelvis and sphere.
    kick_activation_condition = (torch.abs(pelvis_pos_x - small_sphere.data.root_pos_w[:, 0]) < 0.5) & \
                               (torch.abs(pelvis_pos_y - small_sphere.data.root_pos_w[:, 1]) < 0.3)

    # Apply kick reward only when activated, otherwise 0
    kick_reward_activated = torch.where(kick_activation_condition, kick_reward, 0.0)

    # Combine approach and kick rewards
    reward = (approach_reward_x * 0.5) + (approach_reward_y * 0.2) + kick_reward_activated

    # Mandatory normalization
    # Reasoning: Normalization block is included as per mandatory requirements.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_reward") -> torch.Tensor:
    """
    Shaping reward 1: Encourages the robot to maintain a stable, upright posture by keeping its pelvis at a desired height.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required robot part
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis') # Reasoning: Accessing robot part index using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Reasoning: Accessing robot part position using approved pattern.
    pelvis_pos_z = pelvis_pos[:, 2]

    # Desired pelvis height for stability
    desired_pelvis_z = 0.7
    # Reasoning: Uses absolute Z-position of pelvis, which is allowed for height. Continuous reward.
    reward = -torch.abs(pelvis_pos_z - desired_pelvis_z) # Reasoning: Uses relative distance to a target height, continuous.

    # Mandatory normalization
    # Reasoning: Normalization block is included as per mandatory requirements.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def kicking_foot_alignment_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "kicking_foot_alignment_reward") -> torch.Tensor:
    """
    Shaping reward 2: Encourages the robot to position its kicking foot (right foot) close to the small sphere (Object2)
    in X and Y, specifically when in the "pre-kick" phase.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    robot = env.scene["robot"]
    small_sphere = env.scene['Object2'] # Reasoning: Accessing object using approved pattern.

    pelvis_idx = robot.body_names.index('pelvis') # Reasoning: Accessing robot part index using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Reasoning: Accessing robot part position using approved pattern.
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    right_foot_idx = robot.body_names.index('right_ankle_roll_link') # Reasoning: Accessing robot part index using approved pattern.
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx] # Reasoning: Accessing robot part position using approved pattern.
    right_foot_pos_x = right_foot_pos[:, 0]
    right_foot_pos_y = right_foot_pos[:, 1]

    # Object dimensions (hardcoded from object configuration)
    # Reasoning: Object dimensions are hardcoded from the provided object configuration, as per rules.
    small_sphere_radius = 0.2

    # Desired X position for the kicking foot relative to the sphere (e.g., slightly behind)
    # Target X: sphere_x - sphere_radius - 0.1 (just behind the sphere)
    # Reasoning: Uses relative distance between robot foot and sphere. Continuous reward.
    target_foot_x = small_sphere.data.root_pos_w[:, 0] - small_sphere_radius - 0.1
    foot_x_alignment_reward = -torch.abs(right_foot_pos_x - target_foot_x) # Reasoning: Uses relative distance, continuous.

    # Reward for Y alignment of kicking foot
    # Reasoning: Uses relative distance between robot foot and sphere. Continuous reward.
    foot_y_alignment_reward = -torch.abs(right_foot_pos_y - small_sphere.data.root_pos_w[:, 1]) # Reasoning: Uses relative distance, continuous.

    # Activation condition: Robot's pelvis is close to the sphere in X and Y, and still behind it in X.
    # This indicates the pre-kick phase.
    # Reasoning: Uses relative distances between robot pelvis and sphere for activation.
    activation_condition = (torch.abs(pelvis_pos_x - small_sphere.data.root_pos_w[:, 0]) < 0.7) & \
                           (torch.abs(pelvis_pos_y - small_sphere.data.root_pos_w[:, 1]) < 0.5) & \
                           (pelvis_pos_x < small_sphere.data.root_pos_w[:, 0]) # Robot is still behind the sphere

    # Combine foot alignment rewards
    reward = (foot_x_alignment_reward * 0.7) + (foot_y_alignment_reward * 0.3)

    # Apply reward only when activated
    reward = torch.where(activation_condition, reward, 0.0)

    # Mandatory normalization
    # Reasoning: Normalization block is included as per mandatory requirements.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward 3: Penalizes collisions between the robot's body parts (excluding the kicking foot during the kick)
    and the small sphere (Object2), and also between the robot and the block cube (Object5) prematurely.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects and robot parts
    robot = env.scene["robot"]
    small_sphere = env.scene['Object2'] # Reasoning: Accessing object using approved pattern.
    block_cube = env.scene['Object5'] # Reasoning: Accessing object using approved pattern.

    pelvis_idx = robot.body_names.index('pelvis') # Reasoning: Accessing robot part index using approved pattern.
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Reasoning: Accessing robot part position using approved pattern.
    pelvis_pos_x = pelvis_pos[:, 0]

    # Object dimensions (hardcoded from object configuration)
    # Reasoning: Object dimensions are hardcoded from the provided object configuration, as per rules.
    small_sphere_radius = 0.2
    block_cube_x_size = 0.5
    block_cube_y_size = 0.5
    block_cube_z_size = 0.5

    # Robot parts to monitor for collision (excluding the kicking foot during kick)
    # Reasoning: Specific body parts are chosen to avoid unwanted contact.
    body_parts_to_monitor = ['pelvis', 'head_link', 'left_palm_link', 'right_palm_link', 'left_ankle_roll_link']

    collision_penalty = torch.zeros_like(pelvis_pos_x) # Initialize penalty tensor for batch compatibility.

    for part_name in body_parts_to_monitor:
        part_idx = robot.body_names.index(part_name) # Reasoning: Accessing robot part index using approved pattern.
        part_pos = robot.data.body_pos_w[:, part_idx] # Reasoning: Accessing robot part position using approved pattern.
        part_pos_x = part_pos[:, 0]
        part_pos_y = part_pos[:, 1]
        part_pos_z = part_pos[:, 2]

        # Collision with small sphere (Object2)
        # Reasoning: Uses relative distances between robot part and sphere. Continuous penalty.
        dist_x_sphere = torch.abs(part_pos_x - small_sphere.data.root_pos_w[:, 0])
        dist_y_sphere = torch.abs(part_pos_y - small_sphere.data.root_pos_w[:, 1])
        dist_z_sphere = torch.abs(part_pos_z - small_sphere.data.root_pos_w[:, 2])

        # Define a "collision zone" around the sphere
        # These thresholds are relative to the object's size plus a small buffer for robot part radius.
        # Reasoning: Thresholds are derived from object dimensions and a small buffer, not arbitrary.
        sphere_collision_threshold_x = small_sphere_radius + 0.1
        sphere_collision_threshold_y = small_sphere_radius + 0.1
        sphere_collision_threshold_z = small_sphere_radius + 0.1

        # Penalty if part is too close to sphere (not the kicking foot)
        is_colliding_sphere = (dist_x_sphere < sphere_collision_threshold_x) & \
                              (dist_y_sphere < sphere_collision_threshold_y) & \
                              (dist_z_sphere < sphere_collision_threshold_z)

        collision_penalty += torch.where(is_colliding_sphere, -1.0, 0.0) # Small penalty for being too close

        # Collision with block cube (Object5) - only penalize if robot is not yet past the sphere
        # Robot should not touch Object5 until after kicking Object2 and moving past it.
        # Assume robot is still in the "kick small sphere" skill if pelvis_x is before Object5's center.
        # Reasoning: Uses relative distance between robot pelvis and block for activation.
        is_before_block = (pelvis_pos_x < block_cube.data.root_pos_w[:, 0])

        # Define collision zone for block (simplified as a box)
        # These thresholds are relative to the object's size plus a small buffer for robot part radius.
        # Reasoning: Thresholds are derived from object dimensions and a small buffer, not arbitrary.
        block_collision_threshold_x = block_cube_x_size / 2.0 + 0.1
        block_collision_threshold_y = block_cube_y_size / 2.0 + 0.1
        block_collision_threshold_z = block_cube_z_size / 2.0 + 0.1

        # Reasoning: Uses relative distances between robot part and block. Continuous penalty.
        dist_x_block = torch.abs(part_pos_x - block_cube.data.root_pos_w[:, 0])
        dist_y_block = torch.abs(part_pos_y - block_cube.data.root_pos_w[:, 1])
        dist_z_block = torch.abs(part_pos_z - block_cube.data.root_pos_w[:, 2])

        is_colliding_block = (dist_x_block < block_collision_threshold_x) & \
                             (dist_y_block < block_collision_threshold_y) & \
                             (dist_z_block < block_collision_threshold_z)

        collision_penalty += torch.where(is_colliding_block & is_before_block, -2.0, 0.0) # Larger penalty for block collision

    reward = collision_penalty

    # Mandatory normalization
    # Reasoning: Normalization block is included as per mandatory requirements.
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
    Reward terms for the KickSmallSphere skill.
    """
    # Main reward for approaching and kicking the small sphere
    # Reasoning: Main reward weight is 1.0 as per best practices.
    KickSmallSphereMainReward = RewTerm(func=kick_small_sphere_main_reward, weight=1.0,
                                        params={"normalise": True, "normaliser_name": "kick_small_sphere_main_reward"})

    # Shaping reward for maintaining stable pelvis height
    # Reasoning: Shaping reward weight is less than 1.0 as per best practices.
    PelvisHeightReward = RewTerm(func=pelvis_height_reward, weight=0.4,
                                 params={"normalise": True, "normaliser_name": "pelvis_height_reward"})

    # Shaping reward for aligning the kicking foot with the sphere
    # Reasoning: Shaping reward weight is less than 1.0 as per best practices.
    KickingFootAlignmentReward = RewTerm(func=kicking_foot_alignment_reward, weight=0.5,
                                        params={"normalise": True, "normaliser_name": "kicking_foot_alignment_reward"})

    # Shaping reward for avoiding unwanted collisions
    # Reasoning: Shaping reward weight is less than 1.0 as per best practices.
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.3,
                                       params={"normalise": True, "normaliser_name": "collision_avoidance_reward"})