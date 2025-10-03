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


def main_push_largeSphere_towards_highWall_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for the 'push_largeSphere_towards_highWall' skill.
    This reward encourages the robot to push the large sphere (Object1) towards and through the high wall (Object4),
    ensuring the wall is toppled, and the robot maintains stability and good final positioning.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1']  # large sphere
    object4 = env.scene['Object4']  # high wall
    object2 = env.scene['Object2']  # small sphere (for next skill positioning)

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object dimensions (hardcoded from object configuration as per rules)
    # Object1: large sphere, radius 1.0m
    object1_radius = 1.0
    # Object4: high wall, 0.3m in x, 5m in y, 1m in z
    object4_x_dim = 0.3
    # Object2: small sphere, radius 0.2m
    object2_radius = 0.2

    # 1. Reward for pushing Object1 towards Object4 (push_progress_reward)
    # This rewards the reduction of the gap between the front of the sphere and the front of the wall.
    # A negative distance means the sphere has passed the wall's initial position.
    # We want to minimize this positive distance, or maximize the negative distance.
    # Object1's front edge: object1.data.root_pos_w[:, 0] + object1_radius
    # Object4's front edge: object4.data.root_pos_w[:, 0] - object4_x_dim / 2.0
    # Using relative distances as required.
    distance_sphere_wall_x = (object4.data.root_pos_w[:, 0] - object4_x_dim / 2.0) - (object1.data.root_pos_w[:, 0] + object1_radius)
    # Reward is higher as distance_sphere_wall_x becomes smaller (more negative).
    # This is a linear, continuous reward that encourages pushing past the wall.
    push_progress_reward = -distance_sphere_wall_x

    # 2. Reward for Object4 being toppled (wall_fallen_reward)
    # The high wall's initial Z is 1m. If it falls, its Z-position will be significantly lower.
    # We use a threshold of 0.5m for its root_pos_w[:, 2] to indicate it has fallen.
    # This is a step function, providing a large bonus for the goal state, as per the reward design plan.
    wall_fallen_reward = torch.where(object4.data.root_pos_w[:, 2] < 0.5, 10.0, 0.0)

    # 3. Robot stability (stability_reward)
    # Encourage the robot's pelvis to be at a stable height, e.g., 0.7m.
    # This penalizes deviation from the target height. Using absolute Z position for height is allowed sparingly.
    stability_reward = -torch.abs(pelvis_pos[:, 2] - 0.7)

    # 4. Robot final positioning (positioning_reward)
    # Encourage the robot to be positioned slightly past Object4 but before Object2,
    # preparing for the next skill.
    # Target x-range: [Object4's back edge, Object2's front edge]
    # Object4's back edge: object4.data.root_pos_w[:, 0] + object4_x_dim / 2.0
    # Object2's front edge: object2.data.root_pos_w[:, 0] - object2_radius
    # Penalize the absolute distance of the pelvis's x-position from this target center.
    # All calculations are based on relative object positions and hardcoded dimensions.
    target_x_center = (object4.data.root_pos_w[:, 0] + object4_x_dim / 2.0 + object2.data.root_pos_w[:, 0] - object2_radius) / 2.0
    positioning_reward = -torch.abs(pelvis_pos[:, 0] - target_x_center)

    # Combine all reward components
    reward = push_progress_reward + wall_fallen_reward + stability_reward + positioning_reward

    # Normalization as per mandatory requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_hand_proximity_to_largeSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_proximity_reward") -> torch.Tensor:
    """
    Shaping reward: Rewards the robot for keeping its hands close to the back of the large sphere (Object1).
    This encourages continuous contact and effective force application during the push.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1']  # large sphere
    object4 = env.scene['Object4']  # high wall

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    right_hand_idx = robot.body_names.index('right_palm_link')
    right_hand_pos = robot.data.body_pos_w[:, right_hand_idx]
    left_hand_idx = robot.body_names.index('left_palm_link')
    left_hand_pos = robot.data.body_pos_w[:, left_hand_idx]

    # Object dimensions (hardcoded from object configuration as per rules)
    object1_radius = 1.0  # large sphere radius
    object4_x_dim = 0.3   # high wall x dimension

    # Calculate target point on the sphere for hands to be near (back of the sphere with a small offset)
    # This encourages pushing from behind. Using relative positions and hardcoded dimensions.
    target_x_sphere_back = object1.data.root_pos_w[:, 0] - object1_radius - 0.1 # 0.1m offset behind the sphere's back edge
    target_y_sphere_center = object1.data.root_pos_w[:, 1]
    target_z_sphere_center = object1.data.root_pos_w[:, 2]

    # Calculate distances for right hand to the target point on the sphere
    # Using absolute differences for each dimension as per rules for relative distances.
    dist_right_hand_x = torch.abs(right_hand_pos[:, 0] - target_x_sphere_back)
    dist_right_hand_y = torch.abs(right_hand_pos[:, 1] - target_y_sphere_center)
    dist_right_hand_z = torch.abs(right_hand_pos[:, 2] - target_z_sphere_center)

    # Calculate distances for left hand to the target point on the sphere
    dist_left_hand_x = torch.abs(left_hand_pos[:, 0] - target_x_sphere_back)
    dist_left_hand_y = torch.abs(left_hand_pos[:, 1] - target_y_sphere_center)
    dist_left_hand_z = torch.abs(left_hand_pos[:, 2] - target_z_sphere_center)

    # Reward is a penalty for the sum of absolute distances in x, y, z for both hands.
    # This encourages hands to be very close to the target point. This is a continuous reward.
    hand_proximity_reward = - (dist_right_hand_x + dist_right_hand_y + dist_right_hand_z +
                               dist_left_hand_x + dist_left_hand_y + dist_left_hand_z)

    # Activation condition: Only active when Object1 is still in front of Object4 (not yet pushed past it).
    # Object1's front edge: object1.data.root_pos_w[:, 0] + object1_radius
    # Object4's front edge: object4.data.root_pos_w[:, 0] - object4_x_dim / 2.0
    # Condition: Object1's front edge is still behind Object4's front edge.
    # Using relative positions and hardcoded dimensions.
    activation_condition = (object1.data.root_pos_w[:, 0] + object1_radius) < (object4.data.root_pos_w[:, 0] - object4_x_dim / 2.0)

    # Apply the reward only when the activation condition is met, otherwise reward is 0.
    reward = torch.where(activation_condition, hand_proximity_reward, torch.tensor(0.0, device=env.device))

    # Normalization as per mandatory requirements
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward: Penalizes the robot for collisions between its main body parts (pelvis, knees)
    and Object1 (large sphere) or Object4 (high wall), except for the intended pushing contact with Object1.
    This encourages a controlled push and prevents the robot from getting stuck or damaging itself.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access required objects using approved patterns
    object1 = env.scene['Object1']  # large sphere
    object4 = env.scene['Object4']  # high wall

    # Access required robot part(s) using approved patterns
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_knee_idx = robot.body_names.index('left_knee_link')
    left_knee_pos = robot.data.body_pos_w[:, left_knee_idx]
    right_knee_idx = robot.body_names.index('right_knee_link')
    right_knee_pos = robot.data.body_pos_w[:, right_knee_idx]

    # Object dimensions (hardcoded from object configuration as per rules)
    object1_radius = 1.0  # large sphere radius
    object4_x_dim = 0.3   # high wall x dimension
    object4_y_dim = 5.0   # high wall y dimension
    object4_z_dim = 1.0   # high wall z dimension

    # Define a small buffer for collision detection.
    # If a robot part is within this distance from an object's surface, it's considered a collision.
    collision_buffer = 0.1

    collision_penalty = torch.zeros(env.num_envs, device=env.device)

    # Collision with Object1 (large sphere)
    # Penalize if pelvis or knees are too close to Object1.
    # Using torch.norm for Euclidean distance for sphere collision check.
    dist_pelvis_obj1 = torch.norm(pelvis_pos - object1.data.root_pos_w, dim=1)
    dist_left_knee_obj1 = torch.norm(left_knee_pos - object1.data.root_pos_w, dim=1)
    dist_right_knee_obj1 = torch.norm(right_knee_pos - object1.data.root_pos_w, dim=1)

    # Check if pelvis or knees are "colliding" with Object1 (within radius + buffer)
    is_colliding_obj1_pelvis = dist_pelvis_obj1 < (object1_radius + collision_buffer)
    is_colliding_obj1_left_knee = dist_left_knee_obj1 < (object1_radius + collision_buffer)
    is_colliding_obj1_right_knee = dist_right_knee_obj1 < (object1_radius + collision_buffer)

    # Apply penalty if any of these conditions are met. This is a binary penalty.
    collision_penalty -= torch.where(is_colliding_obj1_pelvis | is_colliding_obj1_left_knee | is_colliding_obj1_right_knee, 5.0, 0.0)

    # Collision with Object4 (high wall)
    # Penalize if pelvis or knees are too close to Object4.
    # For a box, we check proximity along each axis. Using absolute differences for each dimension.
    # Pelvis to Object4
    dist_pelvis_obj4_x = torch.abs(pelvis_pos[:, 0] - object4.data.root_pos_w[:, 0])
    dist_pelvis_obj4_y = torch.abs(pelvis_pos[:, 1] - object4.data.root_pos_w[:, 1])
    dist_pelvis_obj4_z = torch.abs(pelvis_pos[:, 2] - object4.data.root_pos_w[:, 2])

    # Left knee to Object4
    dist_left_knee_obj4_x = torch.abs(left_knee_pos[:, 0] - object4.data.root_pos_w[:, 0])
    dist_left_knee_obj4_y = torch.abs(left_knee_pos[:, 1] - object4.data.root_pos_w[:, 1])
    dist_left_knee_obj4_z = torch.abs(left_knee_pos[:, 2] - object4.data.root_pos_w[:, 2])

    # Right knee to Object4
    dist_right_knee_obj4_x = torch.abs(right_knee_pos[:, 0] - object4.data.root_pos_w[:, 0])
    dist_right_knee_obj4_y = torch.abs(right_knee_pos[:, 1] - object4.data.root_pos_w[:, 1])
    dist_right_knee_obj4_z = torch.abs(right_knee_pos[:, 2] - object4.data.root_pos_w[:, 2])

    # Check if pelvis is "colliding" with Object4 (within half-dimension + buffer for each axis)
    is_colliding_obj4_pelvis = (dist_pelvis_obj4_x < (object4_x_dim / 2.0 + collision_buffer)) & \
                               (dist_pelvis_obj4_y < (object4_y_dim / 2.0 + collision_buffer)) & \
                               (dist_pelvis_obj4_z < (object4_z_dim / 2.0 + collision_buffer))

    # Check if left knee is "colliding" with Object4
    is_colliding_obj4_left_knee = (dist_left_knee_obj4_x < (object4_x_dim / 2.0 + collision_buffer)) & \
                                  (dist_left_knee_obj4_y < (object4_y_dim / 2.0 + collision_buffer)) & \
                                  (dist_left_knee_obj4_z < (object4_z_dim / 2.0 + collision_buffer))

    # Check if right knee is "colliding" with Object4
    is_colliding_obj4_right_knee = (dist_right_knee_obj4_x < (object4_x_dim / 2.0 + collision_buffer)) & \
                                   (dist_right_knee_obj4_y < (object4_y_dim / 2.0 + collision_buffer)) & \
                                   (dist_right_knee_obj4_z < (object4_z_dim / 2.0 + collision_buffer))

    # Apply penalty if any of these conditions are met. This is a binary penalty.
    collision_penalty -= torch.where(is_colliding_obj4_pelvis | is_colliding_obj4_left_knee | is_colliding_obj4_right_knee, 5.0, 0.0)

    reward = collision_penalty

    # Normalization as per mandatory requirements
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
    Configuration for the reward terms used in the 'push_largeSphere_towards_highWall' skill.
    """
    # Main reward for pushing the sphere and toppling the wall, with robot stability and positioning.
    # Weight is 1.0 as per requirements for main rewards.
    MainPushLargeSphereTowardsHighWallReward = RewTerm(
        func=main_push_largeSphere_towards_highWall_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "main_reward"}
    )

    # Shaping reward for keeping robot hands close to the large sphere for effective pushing.
    # Weight is 0.5, lower as it's a shaping reward.
    RobotHandProximityToLargeSphereReward = RewTerm(
        func=robot_hand_proximity_to_largeSphere_reward,
        weight=0.5,
        params={"normalise": True, "normaliser_name": "hand_proximity_reward"}
    )

    # Shaping reward for avoiding unintended collisions with objects.
    # Weight is 0.2, lower to prevent over-penalization, but still discourage collisions.
    CollisionAvoidanceReward = RewTerm(
        func=collision_avoidance_reward,
        weight=0.2,
        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"}
    )