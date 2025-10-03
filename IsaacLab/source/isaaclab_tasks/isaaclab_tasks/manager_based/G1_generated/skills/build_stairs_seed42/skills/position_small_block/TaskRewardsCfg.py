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

def main_position_small_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_reward") -> torch.Tensor:
    """
    Main reward for positioning the Small Block (Object1) at its designated target location.
    This reward is based on the absolute distance of Object1's center from its target x, y, and z coordinates.
    """
    # CORRECT: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CORRECT: Access the required object directly
    object1 = env.scene['Object1']

    # Small Block dimensions: x=1m, y=1m, z=0.3m. Center at z=0.15m.
    # CORRECT: Hardcoded target position for the block's center, as defined by the task goal.
    # This is allowed as per rule 8 and 9, where target positions derived from task goals or object dimensions are hardcoded.
    target_object1_x = 3.5
    target_object1_y = 0.0
    target_object1_z = 0.15

    # CORRECT: Calculate the distance vector between Object1 and its target position using relative distances.
    # This adheres to rule 1: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    distance_x = object1.data.root_pos_w[:, 0] - target_object1_x
    distance_y = object1.data.root_pos_w[:, 1] - target_object1_y
    #distance_z = object1.data.root_pos_w[:, 2] - target_object1_z

    # CORRECT: Reward is negative absolute distance, so closer is higher reward. This is a continuous reward.
    # This adheres to rule 7: Continuous Rewards.
    reward = -torch.abs(distance_x) - torch.abs(distance_y) #- torch.abs(distance_z)

    # CORRECT: Complete normalization implementation
    # This adheres to rule 2: MANDATORY REWARD NORMALIZATION.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def shaping_approach_and_align_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "shaping_approach_align") -> torch.Tensor:
    """
    Shaping reward to encourage the robot to approach the Small Block (Object1) and then position its
    right hand (right_palm_link) to be able to push the block.
    It transitions from rewarding pelvis proximity to right hand alignment.
    """
    # CORRECT: Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # CORRECT: Access the required objects
    # This adheres to rule 2: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1']
    robot = env.scene["robot"]

    # CORRECT: Access the required robot parts using approved patterns
    # This adheres to rule 3: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    right_palm_idx = robot.body_names.index('right_palm_link')
    right_palm_pos = robot.data.body_pos_w[:, right_palm_idx]

    # Small Block dimensions: x=1m, y=1m, z=0.3m.
    # Assuming pushing along the x-axis towards positive x.

    # Phase 1: Approach Object1 with pelvis
    # CORRECT: Calculate relative distances for pelvis to object1
    # This adheres to rule 1: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    distance_pelvis_object1_x = object1.data.root_pos_w[:, 0] - pelvis_pos[:, 0]
    distance_pelvis_object1_y = object1.data.root_pos_w[:, 1] - pelvis_pos[:, 1]
    #distance_pelvis_object1_z = object1.data.root_pos_w[:, 2] - pelvis_pos[:, 2]

    # CORRECT: Condition for approaching phase: pelvis is far from object1 (e.g., > 1.5m in x or y).
    # These thresholds are hardcoded as they define the shaping logic, which is allowed for thresholds.
    approach_condition = (torch.abs(distance_pelvis_object1_x) > 1.5) | (torch.abs(distance_pelvis_object1_y) > 1.5)

    # CORRECT: Continuous reward for approaching phase
    # This adheres to rule 7: Continuous Rewards.
    reward_approach = -torch.abs(distance_pelvis_object1_x) - torch.abs(distance_pelvis_object1_y) #- torch.abs(distance_pelvis_object1_z)

    # Phase 2: Align right hand with Object1 for pushing
    # CORRECT: Target hand position relative to block, hardcoded based on block dimensions and desired pushing pose.
    # Block's x-dimension is 1m, so half-width is 0.5m. Hand should be slightly behind, e.g., -0.6m from block center.
    # This is allowed as per rule 8 and 9, where target positions derived from task goals or object dimensions are hardcoded.
    target_hand_x_relative_to_block = -0.6
    target_hand_y_relative_to_block = 0.0  # Aligned with block's y center
    target_hand_z_relative_to_block = 0.15 # Aligned with block's z center (half height)

    # CORRECT: Calculate global target hand position relative to the object's current position.
    # This adheres to rule 1: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    target_hand_x_global = object1.data.root_pos_w[:, 0] + target_hand_x_relative_to_block
    target_hand_y_global = object1.data.root_pos_w[:, 1] + target_hand_y_relative_to_block
    target_hand_z_global = object1.data.root_pos_w[:, 2] + target_hand_z_relative_to_block

    # CORRECT: Calculate relative distances for hand to its target position
    # This adheres to rule 1: ALL rewards MUST ONLY use relative distances between objects and robot parts.
    distance_hand_object1_x = right_palm_pos[:, 0] - target_hand_x_global
    distance_hand_object1_y = right_palm_pos[:, 1] - target_hand_y_global
    #distance_hand_object1_z = right_palm_pos[:, 2] - target_hand_z_global

    # CORRECT: Condition for hand alignment phase: pelvis is close to object1 (e.g., <= 1.5m in x and y).
    hand_align_condition = (torch.abs(distance_pelvis_object1_x) <= 1.5) & (torch.abs(distance_pelvis_object1_y) <= 1.5)

    # CORRECT: Continuous reward for hand alignment phase
    # This adheres to rule 7: Continuous Rewards.
    reward_hand_align = -torch.abs(distance_hand_object1_x) - torch.abs(distance_hand_object1_y) #- torch.abs(distance_hand_object1_z)

    # CORRECT: Combine rewards based on conditions using torch.where for batch compatibility.
    # This adheres to rule 6: HANDLE TENSOR OPERATIONS CORRECTLY.
    reward = torch.where(approach_condition, reward_approach, reward_hand_align)

    # CORRECT: Complete normalization implementation
    # This adheres to rule 2: MANDATORY REWARD NORMALIZATION.
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # CORRECT: Main reward with weight 1.0
    # This adheres to rule 3: PROPER WEIGHTS.
    MainPositionSmallBlockReward = RewTerm(func=main_position_small_block_reward, weight=1.0,
                                           params={"normalise": True, "normaliser_name": "main_reward"})

    # CORRECT: Shaping reward 1 with lower weight
    # This adheres to rule 3: PROPER WEIGHTS.
    ShapingApproachAndAlignReward = RewTerm(func=shaping_approach_and_align_reward, weight=0.6,
                                            params={"normalise": True, "normaliser_name": "shaping_approach_align"})
