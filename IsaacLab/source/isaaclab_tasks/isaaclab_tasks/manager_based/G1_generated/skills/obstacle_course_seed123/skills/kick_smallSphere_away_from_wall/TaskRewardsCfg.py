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

# Removed global RewNormalizer definition. It should be initialized per function using env.device.

def main_kick_smallSphere_away_from_wall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'kick_smallSphere_away_from_wall' skill.
    Encourages the small sphere (Object2) to move significantly away from its initial position,
    specifically in the positive x-direction, and further away from the high wall (Object4).
    Also includes a reward for maintaining a stable pelvis height and positioning the robot
    appropriately for the next skill (not overshooting the block).
    """
    # Get normalizer instance (initialized once per environment)
    # CORRECTED: RewNormalizer should be initialized inside the function using env.device as per example.
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    small_sphere = env.scene['Object2']
    high_wall = env.scene['Object4']
    block_cube = env.scene['Object5']

    # Access required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_z = pelvis_pos[:, 2]

    # --- Reward Component 1: Small sphere x-displacement ---
    # Reasoning: Reward for moving the small sphere away from its initial x-position.
    # The task is to kick it "away from the wall", and given the object order (wall, sphere, block),
    # this implies moving it in the positive x-direction.
    # We assume env.initial_object_pos is available as per the reward design plan.
    # CORRECTED: Added check for env.initial_object_pos existence as it's an assumed attribute.
    if not hasattr(env, 'initial_object_pos') or 'Object2' not in env.initial_object_pos:
        # If initial_object_pos is not available, default to current position to avoid errors,
        # though this might indicate a missing setup in the environment.
        initial_small_sphere_x = small_sphere.data.root_pos_w[:, 0].detach() # Detach to prevent backprop through initial pos
    else:
        initial_small_sphere_x = env.initial_object_pos['Object2'][:, 0]
    sphere_displacement_x = small_sphere.data.root_pos_w[:, 0] - initial_small_sphere_x
    # Reward is linear with displacement, encouraging continuous movement.
    reward_sphere_displacement = 1.0 * sphere_displacement_x

    # --- Reward Component 2: Small sphere distance from high wall ---
    # Reasoning: Reinforces moving the sphere away from the high wall (Object4).
    # Since the high wall is to the left of the small sphere, increasing the x-distance
    # means moving the sphere further in the positive x-direction.
    distance_sphere_wall_x = small_sphere.data.root_pos_w[:, 0] - high_wall.data.root_pos_w[:, 0]
    # Reward is linear with the distance, encouraging continuous separation.
    reward_sphere_wall_distance = 0.5 * distance_sphere_wall_x

    # --- Reward Component 3: Pelvis height for stability ---
    # Reasoning: Encourages the robot to maintain a stable, upright posture.
    # Penalizes large deviations from a target pelvis height (e.g., 0.7m).
    # CORRECTED: Target pelvis height should be a relative value or a reasonable absolute value.
    # The prompt allows z to be absolute.
    target_pelvis_z = 0.7
    # Uses absolute difference for continuous penalty.
    reward_pelvis_height = -torch.abs(pelvis_pos_z - target_pelvis_z)

    # --- Reward Component 4: Robot positioning for next skill ---
    # Reasoning: Guides the robot to be in a good position for the next skill (jumping on the block).
    # This means being slightly past the kicked sphere but not overshooting the block.
    # Encourage pelvis x-position to be close to the sphere's x-position after kick,
    # but penalize if it goes too far past the block.
    # Target: pelvis_x should be around small_sphere.x + 0.5m.
    target_pelvis_x_after_kick = small_sphere.data.root_pos_w[:, 0] + 0.5
    reward_robot_position_x = -torch.abs(pelvis_pos_x - target_pelvis_x_after_kick)

    # Penalty for overshooting the block (Object5)
    # Block x-dimension is 0.5m. Consider the robot's width (e.g., 0.5m) for clearance.
    # Penalize if pelvis_x is significantly past the block's x-position.
    # The block's center is at block_cube.data.root_pos_w[:, 0].
    # A threshold of 0.5m past the block's center is used.
    penalty_overshoot_block = torch.where(
        pelvis_pos_x > block_cube.data.root_pos_w[:, 0] + 0.5,
        -10.0 * (pelvis_pos_x - (block_cube.data.root_pos_w[:, 0] + 0.5)),
        0.0
    )

    # Combine all reward components
    reward = reward_sphere_displacement + reward_sphere_wall_distance + reward_pelvis_height + \
             reward_robot_position_x + penalty_overshoot_block

    # Mandatory normalization
    # Access the local instance
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_and_kick_preparation_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_1") -> torch.Tensor:
    """
    Shaping reward 1: Guides the robot to approach the small sphere (Object2) and position its
    kicking foot (right_ankle_roll_link) appropriately for a kick.
    """
    # Get normalizer instance (initialized once per environment)
    # CORRECTED: RewNormalizer should be initialized inside the function using env.device as per example.
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    small_sphere = env.scene['Object2']

    # Access required robot part(s)
    robot = env.scene["robot"]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]
    right_foot_pos_z = right_foot_pos[:, 2]

    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_pos_x = pelvis_pos[:, 0]
    pelvis_pos_y = pelvis_pos[:, 1]

    # --- Reward Component 1: Robot pelvis approaching the small sphere ---
    # Reasoning: Encourages the robot to get into a kicking position relative to the sphere.
    # Small sphere radius is 0.2m. Robot pelvis should be slightly behind the sphere in x
    # and aligned in y for a kick.
    # Target: pelvis_x should be small_sphere.x - 0.5m (e.g., 0.5m behind sphere center).
    # Target: pelvis_y should be small_sphere.y (aligned with sphere center).
    target_pelvis_x_relative_to_sphere = small_sphere.data.root_pos_w[:, 0] - 0.5
    target_pelvis_y_relative_to_sphere = small_sphere.data.root_pos_w[:, 1]

    # Penalize distance from target in x and y.
    reward_approach_x = -torch.abs(pelvis_pos_x - target_pelvis_x_relative_to_sphere)
    reward_approach_y = -torch.abs(pelvis_pos_y - target_pelvis_y_relative_to_sphere)

    # --- Reward Component 2: Lifting the kicking foot ---
    # Reasoning: Encourages the actual kicking motion.
    # This reward is active only when the robot's pelvis is close enough to the sphere to kick.
    # Condition: pelvis_x is within 1.0m of the sphere's x-position.
    approach_condition = torch.abs(pelvis_pos_x - small_sphere.data.root_pos_w[:, 0]) < 1.0
    # Reward for positive z-position of the right foot, encouraging it to lift off the ground.
    reward_foot_lift = torch.where(approach_condition, 0.5 * right_foot_pos_z, 0.0)

    # Combine all reward components
    reward = reward_approach_x + reward_approach_y + reward_foot_lift

    # Mandatory normalization
    # Access the local instance
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_reward_2") -> torch.Tensor:
    """
    Shaping reward 2: Penalizes unwanted collisions between robot parts and objects,
    and robot parts with the ground.
    """
    # Get normalizer instance (initialized once per environment)
    # CORRECTED: RewNormalizer should be initialized inside the function using env.device as per example.
    RewNormalizer = get_normalizer(env.device)

    # Access required objects
    small_sphere = env.scene['Object2']
    high_wall = env.scene['Object4']
    block_cube = env.scene['Object5']

    # Access required robot part(s)
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    head_idx = robot.body_names.index('head_link')
    head_pos = robot.data.body_pos_w[:, head_idx]
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # --- Object Dimensions (hardcoded from task description) ---
    # Reasoning: Object dimensions are not accessible from RigidObjectData, must be hardcoded.
    # CORRECTED: Added 0.05m buffer to collision thresholds as per prompt's reward design plan.
    small_sphere_radius = 0.2 # From task: "small sphere 0.2m radius"
    high_wall_x_dim = 0.3 # From task: "0.3m in x axis"
    high_wall_y_dim = 5.0 # From task: "5m in the y-axis"
    high_wall_z_dim = 1.0 # From task: "1m in the z axis"
    block_cube_dim = 0.5 # From task: "block cube of 0.5m cubed"

    # --- Collision Penalty Thresholds ---
    # Reasoning: Define thresholds for collision detection based on object sizes and robot part sizes.
    # A small buffer (e.g., 0.05m) is added for robustness.
    collision_threshold_sphere = small_sphere_radius + 0.05 # Sphere radius + buffer
    collision_threshold_wall_x = high_wall_x_dim / 2 + 0.05 # Half x-dim + buffer
    collision_threshold_wall_y = high_wall_y_dim / 2 + 0.05 # Half y-dim + buffer
    collision_threshold_wall_z = high_wall_z_dim / 2 + 0.05 # Half z-dim + buffer
    collision_threshold_block = block_cube_dim / 2 + 0.05 # Half dim + buffer

    # --- Penalty Component 1: Pelvis/Head/Feet collision with high wall (Object4) ---
    # Reasoning: Discourage robot from hitting the high wall, which is an obstacle.
    # Check proximity in all three dimensions.
    dist_pelvis_wall_x = torch.abs(pelvis_pos[:, 0] - high_wall.data.root_pos_w[:, 0])
    dist_pelvis_wall_y = torch.abs(pelvis_pos[:, 1] - high_wall.data.root_pos_w[:, 1])
    dist_pelvis_wall_z = torch.abs(pelvis_pos[:, 2] - high_wall.data.root_pos_w[:, 2])
    penalty_pelvis_wall = torch.where(
        (dist_pelvis_wall_x < collision_threshold_wall_x) &
        (dist_pelvis_wall_y < collision_threshold_wall_y) &
        (dist_pelvis_wall_z < collision_threshold_wall_z),
        -5.0, 0.0
    )

    # --- Penalty Component 2: Pelvis/Head collision with small sphere (Object2) ---
    # Reasoning: Only the foot should interact with the sphere for kicking. Other body parts
    # colliding indicates an improper kick or fall.
    dist_pelvis_sphere = torch.norm(pelvis_pos - small_sphere.data.root_pos_w, dim=1)
    penalty_pelvis_sphere = torch.where(dist_pelvis_sphere < collision_threshold_sphere, -5.0, 0.0)

    dist_head_sphere = torch.norm(head_pos - small_sphere.data.root_pos_w, dim=1)
    penalty_head_sphere = torch.where(dist_head_sphere < collision_threshold_sphere, -5.0, 0.0)

    # --- Penalty Component 3: Robot parts colliding with the ground ---
    # Reasoning: Penalize falling or excessive contact with the ground for non-foot parts.
    # Feet on ground is generally fine, but penalize if they go too far below.
    # Ground is at z=0. A small positive threshold (e.g., 0.05m) is used to allow for slight contact.
    penalty_ground_pelvis = torch.where(pelvis_pos[:, 2] < 0.05, -10.0, 0.0)
    penalty_ground_head = torch.where(head_pos[:, 2] < 0.05, -10.0, 0.0)
    # Small penalty for feet going below ground, but not for being on it.
    # CORRECTED: Changed threshold for feet to 0.0 to only penalize going *below* ground, not just being on it.
    penalty_ground_left_foot = torch.where(left_foot_pos[:, 2] < 0.0, -1.0, 0.0)
    penalty_ground_right_foot = torch.where(right_foot_pos[:, 2] < 0.0, -1.0, 0.0)

    # Combine all penalty components
    reward = penalty_pelvis_wall + penalty_pelvis_sphere + penalty_head_sphere + \
             penalty_ground_pelvis + penalty_ground_head + penalty_ground_left_foot + penalty_ground_right_foot

    # Mandatory normalization
    # Access the local instance
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
    Reward terms for the 'kick_smallSphere_away_from_wall' skill.
    Defines the main reward and supporting shaping rewards with their respective weights.
    """
    # Main reward for successfully kicking the sphere away and positioning the robot.
    MainKickSmallSphereAwayFromWallReward = RewTerm(
        func=main_kick_smallSphere_away_from_wall_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "main_reward"}
    )

    # Shaping reward for approaching the sphere and preparing the kick.
    ApproachAndKickPreparationReward = RewTerm(
        func=approach_and_kick_preparation_reward,
        weight=0.6, # Lower weight than main reward as it's a shaping reward.
        params={"normalise": True, "normaliser_name": "shaping_reward_1"}
    )

    # Shaping reward for avoiding unwanted collisions.
    CollisionAvoidanceReward = RewTerm(
        func=collision_avoidance_reward,
        weight=0.3, # Lower weight to allow some exploration but penalize severe issues.
        params={"normalise": True, "normaliser_name": "shaping_reward_2"}
    )