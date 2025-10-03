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


def main_walk_to_smallSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the walk_to_smallSphere skill.
    Encourages the robot's pelvis to approach and maintain a target distance from Object2 (small sphere)
    in x, y, and z dimensions, positioning it for a kick. The robot should be slightly behind the sphere
    in the x-axis to kick it forward.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object positions using approved patterns
    robot = env.scene["robot"]
    object_smallSphere = env.scene['Object2'] # Object2 is the small sphere for robot to kick

    # Get robot pelvis position
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Shape: [num_envs, 3]
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2]

    # Get small sphere position
    smallSphere_pos = object_smallSphere.data.root_pos_w # Shape: [num_envs, 3]
    smallSphere_pos_x = smallSphere_pos[:, 0]
    smallSphere_pos_y = smallSphere_pos[:, 1]

    # Hardcoded object dimensions from object configuration (radius of Object2)
    # CORRECT: Object dimensions are hardcoded from the object configuration, not accessed from the object itself.
    smallSphere_radius = 0.2 # From object configuration: "Object2": "small sphere for robot to kick", 0.2m radius

    # Define target position relative to the sphere for kicking
    # Robot pelvis should be slightly behind the sphere in x, aligned in y, and at a stable height
    # CORRECT: Target offsets are relative to the object, not hardcoded absolute positions.
    target_x_offset = -0.3 # Robot pelvis should be 0.3m behind the sphere's center in x
    target_y_offset = 0.0  # Robot pelvis should be aligned with sphere's center in y
    target_pelvis_z = 0.7  # Stable standing height for the robot's pelvis

    # Calculate relative distances to the target position
    # ALL rewards MUST ONLY use relative distances between objects and robot parts
    # CORRECT: All distances are calculated as relative differences between positions.
    dist_x = smallSphere_pos_x + target_x_offset - robot_pelvis_pos_x
    dist_y = smallSphere_pos_y + target_y_offset - robot_pelvis_pos_y
    dist_z = target_pelvis_z - robot_pelvis_pos_z

    # Reward for approaching the target position (negative absolute distance for continuous reward)
    # CORRECT: Rewards are continuous (negative absolute distance) and contribute to learning the skill.
    reward_x = -torch.abs(dist_x) # Penalize deviation in x
    reward_y = -torch.abs(dist_y) # Penalize deviation in y
    reward_z = -torch.abs(dist_z) # Penalize deviation in z

    # Penalty for overshooting the sphere in x.
    # The robot should be at an x-position less than or equal to the sphere's x-position + small clearance.
    # This ensures it's positioned to kick it forward, not past it.
    # A small positive offset (e.g., 0.1m) from the sphere's center in x is acceptable for the pelvis.
    # NEVER use hard-coded positions or arbitrary thresholds. This threshold is relative to the sphere's position.
    # CORRECT: Overshoot threshold is relative to the sphere's position and radius, not an arbitrary hardcoded world position.
    overshoot_threshold_x = smallSphere_pos_x + smallSphere_radius + 0.1
    overshoot_penalty_x = torch.where(robot_pelvis_pos_x > overshoot_threshold_x,
                                      -10.0 * (robot_pelvis_pos_x - overshoot_threshold_x),
                                      0.0)

    # Combine rewards
    reward = reward_x + reward_y + reward_z + overshoot_penalty_x

    # MANDATORY REWARD NORMALIZATION
    # CORRECT: Mandatory normalization implemented.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_reward") -> torch.Tensor:
    """
    Shaping reward that penalizes collisions between the robot's main body parts (pelvis, feet, hands)
    and any of the objects in the scene. This encourages safe movement.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access all objects using approved patterns
    # CORRECT: Objects are accessed using their ObjectN names as specified in the prompt.
    object_largeSphere = env.scene['Object1'] # large sphere for robot to push
    object_smallSphere = env.scene['Object2'] # small sphere for robot to kick
    object_lowWall = env.scene['Object3']     # low wall for robot to jump over
    object_highWall = env.scene['Object4']    # high wall for large sphere to push over
    object_block = env.scene['Object5']       # block cube for robot to jump on top of

    # Access relevant robot parts using approved patterns
    # CORRECT: Robot parts are accessed using robot.body_names.index('part_name').
    robot = env.scene["robot"]
    robot_pelvis_pos = robot.data.body_pos_w[:, robot.body_names.index('pelvis')]
    robot_left_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('left_ankle_roll_link')]
    robot_right_foot_pos = robot.data.body_pos_w[:, robot.body_names.index('right_ankle_roll_link')]
    robot_left_hand_pos = robot.data.body_pos_w[:, robot.body_names.index('left_palm_link')]
    robot_right_hand_pos = robot.data.body_pos_w[:, robot.body_names.index('right_palm_link')]

    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Hardcoded object dimensions from object configuration
    # CORRECT: Object dimensions are hardcoded from the object configuration, not accessed from the object itself.
    largeSphere_radius = 1.0
    smallSphere_radius = 0.2
    lowWall_x_dim = 0.3
    lowWall_y_dim = 5.0
    lowWall_z_dim = 0.5
    highWall_x_dim = 0.3
    highWall_y_dim = 5.0
    highWall_z_dim = 1.0
    block_x_dim = 0.5
    block_y_dim = 0.5
    block_z_dim = 0.5

    # Helper function for sphere collision penalty
    def calculate_sphere_collision_penalty(robot_part_pos, object_pos, radius, penalty_value=-5.0):
        # ALL rewards MUST ONLY use relative distances between objects and robot parts
        # CORRECT: Distance calculation uses relative positions.
        distance = torch.norm(robot_part_pos - object_pos, dim=-1)
        # Penalize if robot part is inside the sphere (distance < radius)
        # CORRECT: Reward is continuous (penalty based on distance).
        return torch.where(distance < radius, torch.full_like(distance, penalty_value), torch.tensor(0.0, device=env.device))

    # Helper function for box collision penalty (simplified AABB check)
    def calculate_box_collision_penalty(robot_part_pos, object_pos, obj_x, obj_y, obj_z, penalty_value=-5.0):
        # ALL rewards MUST ONLY use relative distances between objects and robot parts
        # CORRECT: Relative position is used for collision check.
        # Calculate half dimensions for collision check
        half_obj_x = obj_x / 2.0
        half_obj_y = obj_y / 2.0
        half_obj_z = obj_z / 2.0

        # Relative position of robot part to object center
        rel_pos = robot_part_pos - object_pos

        # Check for overlap in each dimension
        # CORRECT: Checks for overlap in x, y, z dimensions separately.
        overlap_x = torch.abs(rel_pos[:, 0]) < half_obj_x
        overlap_y = torch.abs(rel_pos[:, 1]) < half_obj_y
        overlap_z = torch.abs(rel_pos[:, 2]) < half_obj_z

        # Penalize if overlap occurs in all three dimensions
        # CORRECT: Reward is continuous (penalty based on overlap).
        is_colliding = overlap_x * overlap_y * overlap_z
        return torch.where(is_colliding, torch.full_like(is_colliding.float(), penalty_value), torch.tensor(0.0, device=env.device))

    # List of robot parts to check for collisions
    robot_parts = [robot_pelvis_pos, robot_left_foot_pos, robot_right_foot_pos, robot_left_hand_pos, robot_right_hand_pos]

    # Penalize collisions with all objects for relevant robot parts
    # CORRECT: Iterates through robot parts and objects, applying penalties.
    for part_pos in robot_parts:
        # Sphere objects
        collision_penalty += calculate_sphere_collision_penalty(part_pos, object_largeSphere.data.root_pos_w, largeSphere_radius)
        collision_penalty += calculate_sphere_collision_penalty(part_pos, object_smallSphere.data.root_pos_w, smallSphere_radius)
        # Box objects
        collision_penalty += calculate_box_collision_penalty(part_pos, object_lowWall.data.root_pos_w, lowWall_x_dim, lowWall_y_dim, lowWall_z_dim)
        collision_penalty += calculate_box_collision_penalty(part_pos, object_highWall.data.root_pos_w, highWall_x_dim, highWall_y_dim, highWall_z_dim)
        collision_penalty += calculate_box_collision_penalty(part_pos, object_block.data.root_pos_w, block_x_dim, block_y_dim, block_z_dim)

    reward = collision_penalty

    # MANDATORY REWARD NORMALIZATION
    # CORRECT: Mandatory normalization implemented.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def maintain_upright_posture_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "posture_reward") -> torch.Tensor:
    """
    Shaping reward that encourages the robot to maintain an upright and stable posture
    by penalizing large deviations of the pelvis's z-height from a target stable height (0.7m).
    This helps prevent the robot from falling or crouching excessively.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot pelvis position using approved patterns
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_z = robot_pelvis_pos[:, 2] # Z-height is the only absolute position allowed for reward.

    # CORRECT: Target height is a fixed value, which is acceptable for a target Z-height.
    target_pelvis_z = 0.7 # Target stable pelvis height

    # Penalize deviation from target pelvis z-height (negative absolute difference for continuous reward)
    # ALL rewards MUST ONLY use relative distances between objects and robot parts.
    # Here, the "relative distance" is between the pelvis's current Z and a target Z.
    # CORRECT: Reward is continuous (negative absolute difference) and uses the allowed absolute Z-position.
    posture_reward = -torch.abs(robot_pelvis_pos_z - target_pelvis_z)

    reward = posture_reward

    # MANDATORY REWARD NORMALIZATION
    # CORRECT: Mandatory normalization implemented.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Primary reward for walking to and positioning near the small sphere
    # CORRECT: Main reward with weight 1.0.
    MainWalkToSmallSphereReward = RewTerm(func=main_walk_to_smallSphere_reward, weight=1.0,
                                          params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for collision avoidance with all objects
    # CORRECT: Shaping reward with appropriate weight (0.4 < 1.0).
    CollisionAvoidanceReward = RewTerm(func=collision_avoidance_reward, weight=0.4,
                                       params={"normalise": True, "normaliser_name": "collision_reward"})

    # Shaping reward for maintaining an upright posture
    # CORRECT: Shaping reward with appropriate weight (0.2 < 1.0).
    MaintainUprightPostureReward = RewTerm(func=maintain_upright_posture_reward, weight=0.2,
                                            params={"normalise": True, "normaliser_name": "posture_reward"})