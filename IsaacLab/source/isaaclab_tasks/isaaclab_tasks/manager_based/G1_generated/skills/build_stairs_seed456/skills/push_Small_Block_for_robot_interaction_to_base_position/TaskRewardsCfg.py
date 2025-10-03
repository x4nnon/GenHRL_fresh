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


def push_small_block_main_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Primary reward for pushing the Small Block (Object1) to its designated base position.
    The 'designated base position' is defined relative to the robot's current pelvis position
    to adhere to the 'NEVER use hard-coded positions' constraint.
    The target is 1.5m in front of the robot's pelvis on the y-axis, centered on the x-axis,
    and at the correct height for the block (half its height).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts using approved patterns
    object1 = env.scene['Object1']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')

    # Get positions
    object1_pos = object1.data.root_pos_w
    robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Object1 dimensions: x=1m, y=1m, z=0.3m. Half height is 0.15m.
    # This value is hardcoded from the object configuration as per rules.
    object1_half_height = 0.15

    # Define the target position for Object1 relative to the robot's current pelvis position.
    # This ensures the target is always relative to a robot part, avoiding hard-coded world coordinates.
    # Target X: Aligned with robot pelvis X.
    # Target Y: 1.5m in front of robot pelvis Y.
    # Target Z: At the block's half height (on the ground).
    target_object1_x = robot_pelvis_pos[:, 0]
    target_object1_y = robot_pelvis_pos[:, 1] + 1.5
    target_object1_z = object1_half_height

    # Calculate the absolute distances in each dimension between Object1 and its target.
    # This uses relative distances between object and robot part, and then defines a target relative to that.
    distance_x = torch.abs(object1_pos[:, 0] - target_object1_x)
    distance_y = torch.abs(object1_pos[:, 1] - target_object1_y)
    distance_z = torch.abs(object1_pos[:, 2] - target_object1_z)

    # The reward is the negative sum of these distances.
    # This creates a continuous reward that is maximized (becomes 0) when all distances are 0.
    reward = -distance_x - distance_y - distance_z

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def push_small_block_approach_align_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_align_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach Object1 and align its pelvis with the block's x-position.
    This reward is active when the robot is still far from the final pushing position.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    object1 = env.scene['Object1']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')

    # Get positions
    object1_pos = object1.data.root_pos_w
    robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Calculate relative distances between robot pelvis and Object1
    distance_pelvis_object1_x = torch.abs(object1_pos[:, 0] - robot_pelvis_pos[:, 0])
    distance_pelvis_object1_y = torch.abs(object1_pos[:, 1] - robot_pelvis_pos[:, 1])

    # Define the target for Object1 relative to robot's current pelvis position (as in primary reward)
    # This is used to determine if Object1 is still far from its final target.
    object1_half_height = 0.15
    target_object1_x = robot_pelvis_pos[:, 0]
    target_object1_y = robot_pelvis_pos[:, 1] + 1.5
    target_object1_z = object1_half_height

    # Condition: Check if Object1 is far from its final target.
    # This prevents this shaping reward from conflicting with the primary reward when the block is near its goal.
    object1_far_from_target_condition = (torch.abs(object1_pos[:, 0] - target_object1_x) > 0.1) | \
                                        (torch.abs(object1_pos[:, 1] - target_object1_y) > 0.1) | \
                                        (torch.abs(object1_pos[:, 2] - target_object1_z) > 0.1)

    # Condition for approaching: Robot is behind the block on the y-axis (assuming approach from negative y)
    # and not yet in the close pushing zone (e.g., more than 0.6m behind the block).
    # This ensures the reward is active during the approach phase.
    approach_condition = (robot_pelvis_pos[:, 1] < object1_pos[:, 1] - 0.6) & object1_far_from_target_condition

    # Reward for reducing y-distance to block (encourages moving towards the block)
    # The reward is negative, so minimizing distance increases the reward.
    reward_approach_y = -distance_pelvis_object1_y

    # Reward for aligning x-axis (encourages centering with the block)
    reward_align_x = -distance_pelvis_object1_x

    # Combine rewards and apply activation condition
    reward = reward_approach_y + reward_align_x
    reward = torch.where(approach_condition, reward, torch.tensor(0.0, device=env.device))

    # Mandatory normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def push_small_block_pushing_contact_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pushing_contact_reward") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to maintain contact with Object1 while pushing it
    and to keep its pelvis at a stable height.
    This reward is active once the robot is in the pushing zone.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects and robot parts
    object1 = env.scene['Object1']
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')

    # Get positions
    object1_pos = object1.data.root_pos_w
    robot_pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    # Define the target for Object1 relative to robot's current pelvis position (as in primary reward)
    # This is used to determine if Object1 is still far from its final target.
    object1_half_height = 0.15
    target_object1_x = robot_pelvis_pos[:, 0]
    target_object1_y = robot_pelvis_pos[:, 1] + 1.5
    target_object1_z = object1_half_height

    # Condition: Check if Object1 is far from its final target.
    # This prevents this shaping reward from conflicting with the primary reward when the block is near its goal.
    object1_far_from_target_condition = (torch.abs(object1_pos[:, 0] - target_object1_x) > 0.1) | \
                                        (torch.abs(object1_pos[:, 1] - target_object1_y) > 0.1) | \
                                        (torch.abs(object1_pos[:, 2] - target_object1_z) > 0.1)

    # Pushing condition: Robot pelvis is within a certain range behind the block on y-axis
    # and reasonably aligned on x-axis.
    # This defines the "pushing zone".
    pushing_condition = (robot_pelvis_pos[:, 1] >= object1_pos[:, 1] - 0.6) & \
                        (robot_pelvis_pos[:, 1] < object1_pos[:, 1] + 0.1) & \
                        (torch.abs(robot_pelvis_pos[:, 0] - object1_pos[:, 0]) < 0.3) & \
                        object1_far_from_target_condition

    # Reward for maintaining contact: Encourages the robot to stay at an optimal pushing distance (e.g., 0.5m behind block center).
    # This is a continuous reward that is maximized when the distance is close to the target.
    # A small positive constant (0.1) is added to make the reward positive when close to target.
    contact_reward = 0.1 - torch.abs(object1_pos[:, 1] - robot_pelvis_pos[:, 1] - 0.5)

    # Reward for maintaining stable pelvis height: Encourages the robot to keep its pelvis around a typical standing height (e.g., 0.7m).
    # This is a continuous reward that is maximized when the pelvis height is 0.7m.
    stable_pelvis_height_reward = -torch.abs(robot_pelvis_pos[:, 2] - 0.7)

    # Combine rewards and apply activation condition
    reward = contact_reward + stable_pelvis_height_reward
    reward = torch.where(pushing_condition, reward, torch.tensor(0.0, device=env.device))

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
    Reward terms for the push_Small_Block_for_robot_interaction_to_base_position skill.
    """
    # Primary reward for the block reaching its designated base position.
    # Weight is 1.0 as it's the main goal.
    PushSmallBlockMainReward = RewTerm(func=push_small_block_main_reward, weight=1.0,
                                       params={"normalise": True, "normaliser_name": "main_reward"})

    # Shaping reward for approaching and aligning with the block.
    # Lower weight (0.4) as it's a guiding reward for an early phase.
    PushSmallBlockApproachAlignReward = RewTerm(func=push_small_block_approach_align_reward, weight=0.4,
                                                params={"normalise": True, "normaliser_name": "approach_align_reward"})

    # Shaping reward for maintaining contact and stable height during pushing.
    # Lower weight (0.3) as it's a guiding reward for the pushing phase.
    PushSmallBlockPushingContactReward = RewTerm(func=push_small_block_pushing_contact_reward, weight=0.3,
                                                 params={"normalise": True, "normaliser_name": "pushing_contact_reward"})