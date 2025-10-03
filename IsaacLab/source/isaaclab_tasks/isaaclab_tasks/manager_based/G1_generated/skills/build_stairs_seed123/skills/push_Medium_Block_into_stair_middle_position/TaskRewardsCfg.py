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


def push_medium_block_primary_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "push_medium_block_primary_reward") -> torch.Tensor:
    """
    Primary reward for pushing the Medium Block (Object2) to its target position adjacent to the Large Block (Object3).
    The target position is defined such that Object2 is 1.0m along the X-axis from Object3's center,
    and their Y and Z coordinates are aligned.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects using approved patterns
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access object positions using approved patterns
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Define target relative position for Object2 based on Object3's position.
    # Object2 (Medium Block) is 1m x 1m x 0.6m. Object3 (Large Block) is 1m x 1m x 0.9m.
    # As per plan: Target x for Object2: object3.x + 1.0m (assuming pushing along +X axis to form stairs)
    # Target y for Object2: object3.y (aligned)
    # Target z for Object2: object3.z (aligned on the ground)
    target_x_obj2 = object3_pos[:, 0] + 1.0  # Relative distance: 1.0m offset from Object3's X
    target_y_obj2 = object3_pos[:, 1]        # Relative distance: Aligned with Object3's Y
    target_z_obj2 = object3_pos[:, 2]        # Relative distance: Aligned with Object3's Z (on the ground)

    # Calculate distances between Object2's current position and its target relative position
    dist_x_obj2_target = object2_pos[:, 0] - target_x_obj2
    dist_y_obj2_target = object2_pos[:, 1] - target_y_obj2
    dist_z_obj2_target = object2_pos[:, 2] - target_z_obj2

    # Reward is negative absolute distance, encouraging the block to reach the target.
    # This is a continuous reward.
    reward = -torch.abs(dist_x_obj2_target) * 1.0 \
             -torch.abs(dist_y_obj2_target) * 1.0 \
             -torch.abs(dist_z_obj2_target) * 1.0

    # Add a small positive exponential reward for being very close to the target.
    # This encourages fine-tuning once the block is near the goal.
    reward += 0.5 * torch.exp(-5.0 * (torch.abs(dist_x_obj2_target) + torch.abs(dist_y_obj2_target) + torch.abs(dist_z_obj2_target)))

    # Apply normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def robot_hand_contact_and_push_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "robot_hand_contact_and_push_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot's right hand to be in a good position to push Object2.
    It encourages the hand to be close to the pushing side of Object2 (e.g., -X side if pushing in +X).
    This reward is active only when Object2 has not yet reached its final target position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and object using approved patterns
    robot = env.scene["robot"]
    object2 = env.scene['Object2']

    # Access robot part position using approved patterns
    right_palm_link_idx = robot.body_names.index('right_palm_link')
    right_palm_link_pos = robot.data.body_pos_w[:, right_palm_link_idx]
    object2_pos = object2.data.root_pos_w

    # Hardcode Object2 dimensions from configuration (1m x 1m x 0.6m)
    object2_size_x = 1.0
    object2_size_y = 1.0
    object2_size_z = 0.6

    # Define target hand position relative to Object2's center for pushing.
    # Assuming pushing along +X axis, hand should be slightly behind Object2's -X face, centered on Y, at mid-height.
    target_palm_x_rel_obj2 = -object2_size_x / 2.0 - 0.1  # 0.1m behind the -X face of Object2
    target_palm_y_rel_obj2 = 0.0                          # Centered on Y
    target_palm_z_rel_obj2 = object2_size_z / 2.0         # Mid-height of the block

    # Calculate absolute target hand position relative to Object2's current position
    target_palm_x = object2_pos[:, 0] + target_palm_x_rel_obj2
    target_palm_y = object2_pos[:, 1] + target_palm_y_rel_obj2
    target_palm_z = object2_pos[:, 2] + target_palm_z_rel_obj2

    # Calculate distances between right_palm_link and its target position relative to Object2
    reward_palm_x = -torch.abs(right_palm_link_pos[:, 0] - target_palm_x)
    reward_palm_y = -torch.abs(right_palm_link_pos[:, 1] - target_palm_y)
    reward_palm_z = -torch.abs(right_palm_link_pos[:, 2] - target_palm_z)

    # Sum the individual component rewards
    shaping_reward1 = (reward_palm_x + reward_palm_y + reward_palm_z)

    # Condition: This reward is only active when Object2 is not yet at its final position.
    # We use a threshold based on the primary reward's target distance.
    object3 = env.scene['Object3']
    object3_pos = object3.data.root_pos_w
    # Re-calculate the target for Object2 for the condition check
    target_x_obj2_for_cond = object3_pos[:, 0] + 1.0
    target_y_obj2_for_cond = object3_pos[:, 1]
    target_z_obj2_for_cond = object3_pos[:, 2]

    # Check if Object2 is outside a 0.1m radius of its target
    object2_not_at_target_condition = (torch.abs(object2_pos[:, 0] - target_x_obj2_for_cond) > 0.1) | \
                                      (torch.abs(object2_pos[:, 1] - target_y_obj2_for_cond) > 0.1) | \
                                      (torch.abs(object2_pos[:, 2] - target_z_obj2_for_cond) > 0.1)

    # Apply the condition: reward is 0.0 if Object2 is at target, otherwise it's the calculated shaping reward.
    shaping_reward1 = torch.where(object2_not_at_target_condition, shaping_reward1, torch.tensor(0.0, device=env.device))

    # Apply normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward1)
        RewNormalizer.update_stats(normaliser_name, shaping_reward1)
        return scaled_reward
    return shaping_reward1


def robot_stability_and_collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "robot_stability_and_collision_avoidance_reward") -> torch.Tensor:
    """
    Shaping reward for robot stability (pelvis height), collision avoidance with blocks,
    and positioning for the next skill (climbing stairs).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access robot and objects using approved patterns
    robot = env.scene["robot"]
    object2 = env.scene['Object2']
    object3 = env.scene['Object3']

    # Access robot part positions using approved patterns
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    object2_pos = object2.data.root_pos_w
    object3_pos = object3.data.root_pos_w

    # Hardcode Object dimensions from configuration
    object2_size_x, object2_size_y, object2_size_z = 1.0, 1.0, 0.6
    object3_size_x, object3_size_y, object3_size_z = 1.0, 1.0, 0.9

    # 1. Pelvis height reward for stability
    pelvis_target_z = 0.7  # Target standing height for the robot's pelvis
    reward_pelvis_z = -torch.abs(pelvis_pos[:, 2] - pelvis_target_z) # Continuous reward for maintaining target height

    # 2. Collision avoidance with Object2 and Object3
    # Penalize if pelvis is inside the object's bounding box + a small margin.
    # This is a simplified collision check using pelvis as a proxy for the robot body.
    collision_margin = 0.2 # Margin around robot and object for collision detection

    # Calculate distances from pelvis to Object2's center
    dist_x_pelvis_obj2 = torch.abs(pelvis_pos[:, 0] - object2_pos[:, 0])
    dist_y_pelvis_obj2 = torch.abs(pelvis_pos[:, 1] - object2_pos[:, 1])
    dist_z_pelvis_obj2 = torch.abs(pelvis_pos[:, 2] - object2_pos[:, 2])

    # Define penalty thresholds based on object dimensions and collision margin
    penalty_threshold_x_obj2 = (object2_size_x / 2.0) + collision_margin
    penalty_threshold_y_obj2 = (object2_size_y / 2.0) + collision_margin
    penalty_threshold_z_obj2 = (object2_size_z / 2.0) + collision_margin

    # Apply large negative penalty if pelvis is "colliding" with Object2
    collision_penalty_obj2 = torch.where(
        (dist_x_pelvis_obj2 < penalty_threshold_x_obj2) &
        (dist_y_pelvis_obj2 < penalty_threshold_y_obj2) &
        (dist_z_pelvis_obj2 < penalty_threshold_z_obj2),
        -10.0, # Large penalty for collision
        0.0
    )

    # Calculate distances from pelvis to Object3's center
    dist_x_pelvis_obj3 = torch.abs(pelvis_pos[:, 0] - object3_pos[:, 0])
    dist_y_pelvis_obj3 = torch.abs(pelvis_pos[:, 1] - object3_pos[:, 1])
    dist_z_pelvis_obj3 = torch.abs(pelvis_pos[:, 2] - object3_pos[:, 2])

    # Define penalty thresholds based on object dimensions and collision margin
    penalty_threshold_x_obj3 = (object3_size_x / 2.0) + collision_margin
    penalty_threshold_y_obj3 = (object3_size_y / 2.0) + collision_margin
    penalty_threshold_z_obj3 = (object3_size_z / 2.0) + collision_margin

    # Apply large negative penalty if pelvis is "colliding" with Object3
    collision_penalty_obj3 = torch.where(
        (dist_x_pelvis_obj3 < penalty_threshold_x_obj3) &
        (dist_y_pelvis_obj3 < penalty_threshold_y_obj3) &
        (dist_z_pelvis_obj3 < penalty_threshold_z_obj3),
        -10.0, # Large penalty for collision
        0.0
    )

    # 3. Reward for robot being in a good position for the next skill (climbing stairs)
    # This reward encourages the robot to be near the newly formed stairs, ready to climb,
    # and not too far away or on top of the blocks.
    # Target robot position: roughly 0.5m behind Object2's final position (relative to push direction)
    # and aligned with the stairs.
    # Object2's final X position is object3.x + 1.0.
    # So, robot's target X is (object3.x + 1.0) + (object2_size_x / 2.0) + 0.5m (behind Object2's face)
    target_robot_x = object3_pos[:, 0] + 1.0 + (object2_size_x / 2.0) + 0.5
    target_robot_y = object3_pos[:, 1] # Aligned with stairs Y
    target_robot_z = pelvis_target_z   # Standing height

    # Calculate distances from pelvis to the final target robot position
    dist_x_pelvis_final_pos = pelvis_pos[:, 0] - target_robot_x
    dist_y_pelvis_final_pos = pelvis_pos[:, 1] - target_robot_y
    dist_z_pelvis_final_pos = pelvis_pos[:, 2] - target_robot_z

    # Reward is negative absolute distance, encouraging the robot to reach the target final position.
    reward_final_robot_pos = -torch.abs(dist_x_pelvis_final_pos) \
                             -torch.abs(dist_y_pelvis_final_pos) \
                             -torch.abs(dist_z_pelvis_final_pos)

    # Condition: This reward is more active when Object2 is close to its final position.
    # Use a threshold for the primary reward's target distance.
    target_x_obj2_for_cond = object3_pos[:, 0] + 1.0
    target_y_obj2_for_cond = object3_pos[:, 1]
    target_z_obj2_for_cond = object3_pos[:, 2]

    # Check if Object2 is within a 0.2m radius of its target
    object2_at_target_condition = (torch.abs(object2_pos[:, 0] - target_x_obj2_for_cond) < 0.2) & \
                                  (torch.abs(object2_pos[:, 1] - target_y_obj2_for_cond) < 0.2) & \
                                  (torch.abs(object2_pos[:, 2] - target_z_obj2_for_cond) < 0.2)

    # Combine all shaping reward components
    shaping_reward2 = reward_pelvis_z * 0.2 + collision_penalty_obj2 + collision_penalty_obj3
    # The final robot position reward is only added when Object2 is at its target
    shaping_reward2 += torch.where(object2_at_target_condition, reward_final_robot_pos * 0.3, torch.tensor(0.0, device=env.device))

    # Apply normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()
    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, shaping_reward2)
        RewNormalizer.update_stats(normaliser_name, shaping_reward2)
        return scaled_reward
    return shaping_reward2


@configclass
class TaskRewardsCfg:
    # Primary reward for the Medium Block reaching its target position relative to the Large Block
    PushMediumBlockPrimaryReward = RewTerm(func=push_medium_block_primary_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "push_medium_block_primary_reward"})

    # Shaping reward for robot hand contact and pushing posture
    RobotHandContactAndPushReward = RewTerm(func=robot_hand_contact_and_push_reward, weight=0.6,
                                            params={"normalise": True, "normaliser_name": "robot_hand_contact_and_push_reward"})

    # Shaping reward for robot stability, collision avoidance, and final positioning
    RobotStabilityAndCollisionAvoidanceReward = RewTerm(func=robot_stability_and_collision_avoidance_reward, weight=0.5,
                                                        params={"normalise": True, "normaliser_name": "robot_stability_and_collision_avoidance_reward"})