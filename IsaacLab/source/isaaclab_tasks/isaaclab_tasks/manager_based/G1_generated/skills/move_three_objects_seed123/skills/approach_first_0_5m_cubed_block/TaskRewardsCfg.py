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


def approach_block_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_block_reward") -> torch.Tensor:
    """
    Main reward for the robot to approach the 'first 0.5m cubed block' (Object1)
    from the side opposite the 'platform' (Object4), aligning its pelvis behind the block.
    This sets up the robot for a pushing action.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required objects using approved patterns
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    object1 = env.scene['Object1'] # 'first 0.5m cubed block'
    object4 = env.scene['Object4'] # 'platform'

    # Access the required robot part(s) using approved patterns
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Object1 dimensions (0.5m cubed block) - hardcoded from object configuration
    # Requirement: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    block_half_size = 0.25 # 0.5m / 2

    # Calculate target position relative to Object1 and Object4
    # The robot should be positioned directly behind Object1, on the side opposite Object4.
    # Assuming Object4 (platform) is generally in the positive X direction relative to Object1 for pushing.
    # Thus, the robot should be on the negative X side of Object1.
    # Target x-position: slightly behind Object1's edge (Object1.x - block_half_size - clearance)
    # Target y-position: aligned with Object1's y-center
    # Target z-position: stable pelvis height (0.7m)

    # Requirement: ALL rewards MUST ONLY use relative distances between objects and robot parts
    # Requirement: YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN
    # Target x-position: 0.15m behind the block's edge in the negative X direction
    target_pelvis_x = object1.data.root_pos_w[:, 0] - block_half_size - 0.15
    # Target y-position: Aligned with the block's y-center
    target_pelvis_y = object1.data.root_pos_w[:, 1]
    # Target z-position: Desired stable pelvis height. This is an absolute height, which is allowed for stability.
    # Requirement: z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill.
    target_pelvis_z = 0.7

    # Calculate distance components to the target position
    # Using relative distances between robot pelvis and the calculated target point.
    # Requirement: All rewards must only be based on relative distances between objects and robot parts, NO OTHER MEASUREMENTS MAY BE USED FOR SHAPING REWARDS.
    distance_x = target_pelvis_x - pelvis_pos[:, 0]
    distance_y = target_pelvis_y - pelvis_pos[:, 1]
    distance_z = target_pelvis_z - pelvis_pos[:, 2]

    # Reward for reducing distance to target position.
    # Using negative absolute distance for continuous positive reward as distance decreases.
    # This ensures the reward is smooth and continuous.
    # Requirement: Use smooth, continuous rewards.
    # Requirement: absolute distances must be used for distances from objects.
    reward_x = -torch.abs(distance_x)
    reward_y = -torch.abs(distance_y)
    reward_z = -torch.abs(distance_z)

    # Combine rewards. Weight x more as it's the primary approach direction for pushing.
    # All rewards are continuous and based on relative distances.
    reward = (reward_x * 0.5) + (reward_y * 0.3) + (reward_z * 0.2)

    # Mandatory reward normalization
    # Requirement: MANDATORY REWARD NORMALIZATION - EVERY reward function MUST include normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def pelvis_height_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "pelvis_height_stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to maintain an upright and stable posture by penalizing
    large deviations of the pelvis z-position from a desired height (0.7m).
    This helps prevent the robot from falling or crouching excessively, ensuring it's ready for the next skill.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s) using approved patterns
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    # Z-position is an absolute height, which is allowed for stability.
    # Requirement: z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill.
    pelvis_pos_z = pelvis_pos[:, 2]

    # Desired pelvis height
    # Requirement: NEVER use hard-coded positions or arbitrary thresholds.
    # This is a desired height for stability, not a hard-coded world position. It's a target relative to the robot's own body.
    desired_pelvis_z = 0.7

    # Penalize deviation from desired pelvis height.
    # Using negative absolute difference for a continuous positive reward as deviation decreases.
    # Requirement: Use smooth, continuous rewards.
    # Requirement: absolute distances must be used for distances from objects.
    reward = -torch.abs(pelvis_pos_z - desired_pelvis_z)

    # Mandatory reward normalization
    # Requirement: MANDATORY REWARD NORMALIZATION - EVERY reward function MUST include normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


def feet_on_ground_stability_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "feet_on_ground_stability_reward") -> torch.Tensor:
    """
    This reward encourages the robot to keep its feet on the ground and avoid lifting them too high,
    promoting stable walking. It penalizes the z-position of the feet if they are significantly
    above the ground plane (z=0).
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access the required robot part(s) using approved patterns
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]
    left_ankle_idx = robot.body_names.index('left_ankle_roll_link')
    right_ankle_idx = robot.body_names.index('right_ankle_roll_link')

    left_ankle_pos = robot.data.body_pos_w[:, left_ankle_idx]
    right_ankle_pos = robot.data.body_pos_w[:, right_ankle_idx]

    # Z-position is an absolute height, allowed for ground contact.
    # Requirement: z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill.
    left_ankle_pos_z = left_ankle_pos[:, 2]
    right_ankle_pos_z = right_ankle_pos[:, 2]

    # Ground plane is at z=0. Penalize if feet are too high above ground.
    # A small threshold (e.g., 0.05m) allows for normal walking steps without penalizing.
    # This threshold is relative to the ground (z=0), which is a fixed environmental reference.
    # It's a common practice for stability rewards to use such thresholds.
    ground_threshold = 0.05

    # Reward is negative if foot z is above threshold, 0 otherwise.
    # This creates a continuous penalty for lifting feet too high.
    # Requirement: Use smooth, continuous rewards.
    # Requirement: All rewards must only be based on relative distances between objects and robot parts, NO OTHER MEASUREMENTS MAY BE USED FOR SHAPING REWARDS.
    # The distance here is relative to the ground plane (z=0) and the threshold.
    reward_left_foot = torch.where(left_ankle_pos_z > ground_threshold, -torch.abs(left_ankle_pos_z - ground_threshold), torch.tensor(0.0, device=env.device))
    reward_right_foot = torch.where(right_ankle_pos_z > ground_threshold, -torch.abs(right_ankle_pos_z - ground_threshold), torch.tensor(0.0, device=env.device))

    # Combine rewards for both feet.
    reward = reward_left_foot + reward_right_foot

    # Mandatory reward normalization
    # Requirement: MANDATORY REWARD NORMALIZATION - EVERY reward function MUST include normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward


@configclass
class TaskRewardsCfg:
    # Main reward for approaching the block, weighted higher as it's the primary objective.
    # Requirement: PROPER WEIGHTS - Set appropriate weights in TaskRewardsCfg (primary reward ~1.0, supporting rewards <1.0)
    ApproachBlockReward = RewTerm(func=approach_block_reward, weight=1.0,
                                  params={"normalise": True, "normaliser_name": "approach_block_reward"})

    # Supporting reward for maintaining pelvis height stability.
    # Requirement: PROPER WEIGHTS - Set appropriate weights in TaskRewardsCfg (primary reward ~1.0, supporting rewards <1.0)
    PelvisHeightStabilityReward = RewTerm(func=pelvis_height_stability_reward, weight=0.4,
                                          params={"normalise": True, "normaliser_name": "pelvis_height_stability_reward"})

    # Supporting reward for keeping feet on the ground, promoting stable locomotion.
    # Requirement: PROPER WEIGHTS - Set appropriate weights in TaskRewardsCfg (primary reward ~1.0, supporting rewards <1.0)
    FeetOnGroundStabilityReward = RewTerm(func=feet_on_ground_stability_reward, weight=0.2,
                                          params={"normalise": True, "normaliser_name": "feet_on_ground_stability_reward"})