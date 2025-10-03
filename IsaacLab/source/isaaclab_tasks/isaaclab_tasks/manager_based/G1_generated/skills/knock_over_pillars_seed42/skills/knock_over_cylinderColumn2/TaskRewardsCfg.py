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


def knock_over_cylinderColumn2_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    '''Main reward for knock_over_cylinderColumn2.

    Rewards the robot for knocking over Cylinder Column 2 by encouraging its Z-position to approach the fallen state.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required object: Object2 (Cylinder Column 2)
    object2 = env.scene['Object2']
    # Access Object2's current root position in world coordinates
    current_z_object2 = object2.data.root_pos_w[:, 2] # Accessing Z-component of object's position

    # Cylinder column dimensions (hardcoded from object configuration): height 2m, radius 0.3m
    # When standing, the center Z is 1.0m (half height).
    # When fallen, the center Z is 0.3m (its radius).
    target_z_fallen = 0.3 # Target Z for the object's center when it has fallen completely
    standing_z_center = 1.0 # Z for the object's center when it is perfectly standing

    # Reward is higher when current_z_object2 is closer to target_z_fallen.
    # We use a negative absolute difference to create a continuous reward that is maximized at the target.
    # The reward is scaled to be 0 when standing (Z=1.0) and 1 when fallen (Z=0.3).
    # This creates a normalized range for the reward.
    reward = (standing_z_center - current_z_object2) / (standing_z_center - target_z_fallen)
    # Clamp the reward to ensure it stays within a reasonable range, e.g., 0 to 1.
    reward = torch.clamp(reward, min=0.0, max=1.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def approach_and_position_object2_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_reward") -> torch.Tensor:
    '''Shaping reward for approaching and positioning the robot's pelvis relative to Object2.

    Encourages the robot to reduce the X-Y distance between its pelvis and Object2's center,
    and to be slightly past the center of Object2 in the X-direction for pushing.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required object: Object2
    object2 = env.scene['Object2']
    object2_pos_x = object2.data.root_pos_w[:, 0] # X-component of Object2's position
    object2_pos_y = object2.data.root_pos_w[:, 1] # Y-component of Object2's position

    # Access the required robot part: pelvis
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis') # Get index of pelvis
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx] # Get pelvis position
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0] # X-component of pelvis position
    robot_pelvis_pos_y = robot_pelvis_pos[:, 1] # Y-component of pelvis position

    # Calculate relative distances between pelvis and Object2's center
    # This is the distance from the robot's pelvis to the object's center.
    # We want the robot to be slightly past the object in X for pushing.
    # Assuming the robot pushes from the negative X side towards positive X.
    # A target X offset of +0.5m means the robot's X should be 0.5m greater than object's X.
    # So, robot_pelvis_pos_x - object2_pos_x should be 0.5.
    # Therefore, (robot_pelvis_pos_x - object2_pos_x) - 0.5 should be 0.
    target_x_offset_relative_to_object = 0.5 # Robot's X should be 0.5m past Object2's X for pushing
    
    # Reward for X-position: closer to the target offset
    # We use negative absolute difference for a continuous reward that is maximized at the target.
    reward_x = -torch.abs((robot_pelvis_pos_x - object2_pos_x) - target_x_offset_relative_to_object)

    # Reward for Y-position: closer to Object2's Y-center (i.e., minimize Y-distance)
    reward_y = -torch.abs(robot_pelvis_pos_y - object2_pos_y)

    # Combine X and Y rewards
    reward = reward_x + reward_y

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def hand_contact_and_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "hand_contact_reward") -> torch.Tensor:
    '''Shaping reward for encouraging hand contact with Object2 and preventing collisions/overshoot.

    Rewards the robot for having its hands close to Object2's surface at a suitable height for pushing.
    Includes a penalty for pelvis collision with Object2 and for overshooting Object2 too much in X.
    '''
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required object: Object2
    object2 = env.scene['Object2']
    object2_pos_x = object2.data.root_pos_w[:, 0]
    object2_pos_y = object2.data.root_pos_w[:, 1]
    object2_pos_z = object2.data.root_pos_w[:, 2] # Current Z of object2's center

    # Access required robot parts: hands and pelvis
    robot = env.scene["robot"]
    robot_left_hand_idx = robot.body_names.index('left_palm_link')
    robot_right_hand_idx = robot.body_names.index('right_palm_link')
    robot_pelvis_idx = robot.body_names.index('pelvis')

    robot_left_hand_pos = robot.data.body_pos_w[:, robot_left_hand_idx]
    robot_right_hand_pos = robot.data.body_pos_w[:, robot_right_hand_idx]
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]
    robot_pelvis_pos_x = robot_pelvis_pos[:, 0]

    # Cylinder column dimensions (hardcoded from object configuration): height 2m, radius 0.3m
    column_radius = 0.3
    column_height = 2.0

    # --- Hand Contact Reward ---
    # Target Z for hand contact: Aim for mid-height of the column when standing.
    # The object's center Z is object2_pos_z. The top of the column is object2_pos_z + column_height/2.
    # A good pushing height might be slightly below the top, e.g., 0.5m below the top.
    target_hand_z = object2_pos_z + (column_height / 2.0) - 0.5

    # Calculate distance from hands to object's center in X-Y plane
    dist_left_hand_xy_to_center = torch.sqrt(torch.square(robot_left_hand_pos[:, 0] - object2_pos_x) + torch.square(robot_left_hand_pos[:, 1] - object2_pos_y))
    dist_right_hand_xy_to_center = torch.sqrt(torch.square(robot_right_hand_pos[:, 0] - object2_pos_x) + torch.square(robot_right_hand_pos[:, 1] - object2_pos_y))

    # Reward for hands being close to the object's surface (within radius) in X-Y.
    # We want the distance to the center to be approximately the column_radius.
    reward_left_hand_xy = -torch.abs(dist_left_hand_xy_to_center - column_radius)
    reward_right_hand_xy = -torch.abs(dist_right_hand_xy_to_center - column_radius)

    # Reward for hands being at the target Z height.
    reward_left_hand_z = -torch.abs(robot_left_hand_pos[:, 2] - target_hand_z)
    reward_right_hand_z = -torch.abs(robot_right_hand_pos[:, 2] - target_hand_z)

    # Combine hand rewards, averaged
    reward_hands = (reward_left_hand_xy + reward_right_hand_xy + reward_left_hand_z + reward_right_hand_z) / 4.0

    # --- Collision Avoidance for Pelvis with Object2 ---
    # Prevent robot from going through Object2.
    # Calculate X-Y distance from pelvis to Object2's center.
    pelvis_object2_dist_xy = torch.sqrt(torch.square(robot_pelvis_pos[:, 0] - object2_pos_x) + torch.square(robot_pelvis_pos[:, 1] - object2_pos_y))
    
    # Define a collision threshold: object radius + a small buffer.
    collision_threshold_pelvis_object2 = column_radius + 0.2 # 0.3m radius + 0.2m buffer = 0.5m
    
    # Apply a negative reward if pelvis is too close (inside the threshold).
    # The penalty increases linearly as the pelvis gets closer than the threshold.
    reward_pelvis_collision = torch.where(pelvis_object2_dist_xy < collision_threshold_pelvis_object2,
                                          -10.0 * (collision_threshold_pelvis_object2 - pelvis_object2_dist_xy),
                                          0.0)

    # --- Overshoot Penalty ---
    # Ensure robot does not overshoot Object2 too much in X, to be ready for the next skill (Object3).
    # Assuming Object3 is further in positive X from Object2.
    # Robot's X position should not be too far past Object2's X.
    # Let's say the robot should not go more than 1.0m past Object2's center in X.
    max_x_offset_past_object2 = 1.0 # Max X distance robot can be past Object2's center
    
    # Condition for overshoot: robot's X position is greater than Object2's X + max_x_offset.
    overshoot_condition = robot_pelvis_pos_x > (object2_pos_x + max_x_offset_past_object2)
    
    # Apply a negative reward if overshooting, proportional to how much it overshoots.
    reward_overshoot = torch.where(overshoot_condition,
                                   -5.0 * (robot_pelvis_pos_x - (object2_pos_x + max_x_offset_past_object2)),
                                   0.0)

    # Combine all components of this shaping reward
    reward = reward_hands + reward_pelvis_collision + reward_overshoot

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for knocking over Cylinder Column 2
    MainKnockOverCylinderColumn2Reward = RewTerm(func=knock_over_cylinderColumn2_main_reward, weight=1.0,
                                                 params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for approaching and positioning the robot's pelvis relative to Object2
    ApproachAndPositionObject2Reward = RewTerm(func=approach_and_position_object2_reward, weight=0.5,
                                               params={"normalise": True, "normaliser_name": "approach_reward"})

    # Shaping reward for encouraging hand contact and preventing collisions/overshoot
    HandContactAndCollisionAvoidanceReward = RewTerm(func=hand_contact_and_collision_avoidance_reward, weight=0.3,
                                                     params={"normalise": True, "normaliser_name": "hand_contact_reward"})