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

def main_kick_smallSphere_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "main_kick_smallSphere_reward") -> torch.Tensor:
    """
    Primary reward for the kick_smallSphere skill.
    Simple reward: distance of small sphere - distance of block.
    Negative when sphere is before block, positive when past block.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects using approved patterns
    object2 = env.scene['Object2'] # Small sphere
    object5 = env.scene['Object5'] # Block cube

    # Simple distance-based reward: sphere_x - block_x
    sphere_x = object2.data.root_pos_w[:, 0]
    block_x = object5.data.root_pos_w[:, 0]
    

    # Reward: distance from sphere to block (xy norm), clipped at 3m
    sphere_pos = object2.data.root_pos_w
    block_pos = object5.data.root_pos_w
    sphere_block_xy_dist = torch.norm(sphere_pos[:, :2] - block_pos[:, :2], dim=1)
    reward = torch.clamp(sphere_block_xy_dist, max=4.0)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def stay_near_block_y_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "stay_near_block_y_reward") -> torch.Tensor:
    """
    Reward for the robot's pelvis staying near the y-coordinate of the block (Object5).
    Encourages alignment in the y-direction with the block.
    Uses negative absolute distance as the reward.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access block and robot pelvis positions
    object5 = env.scene['Object5']  # Block cube
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    block_y = object5.data.root_pos_w[:, 1]
    pelvis_y = robot_pelvis_pos[:, 1]

    # Reward is negative absolute y-distance (higher when closer)
    y_distance = torch.abs(pelvis_y - block_y)
    reward = -y_distance

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward



def approach_and_alignment_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "approach_and_alignment_reward") -> torch.Tensor:
    """
    Shaping reward to help robot approach the small sphere and maintain proper pelvis height.
    Simplified to focus on approach distance and pelvis height only.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects and robot parts using approved patterns
    object2 = env.scene['Object2'] # Small sphere
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Approach reward: encourage robot to get closer to the sphere
    sphere_pos = object2.data.root_pos_w
    robot_pos = robot_pelvis_pos
    approach_distance = torch.norm(sphere_pos - robot_pos, dim=1)
    
    # Reward for reducing distance to sphere (exponential decay)
    approach_reward = torch.exp(-approach_distance * 2.0)

    # Pelvis height reward: encourage stable height around 0.7m
    pelvis_target_z = 0.7
    pelvis_height_reward = torch.exp(-torch.abs(robot_pelvis_pos[:, 2] - pelvis_target_z) * 5.0)

    # Combine rewards
    reward = (approach_reward * 0.6) + (pelvis_height_reward * 0.4)

    # Mandatory reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, reward)
        RewNormalizer.update_stats(normaliser_name, reward)
        return scaled_reward
    return reward

def collision_avoidance_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "collision_avoidance_reward") -> torch.Tensor:
    """
    Simple collision avoidance reward - minimal penalty for major collisions only.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access objects and robot parts using approved patterns
    object2 = env.scene['Object2'] # Small sphere
    robot = env.scene["robot"]
    robot_pelvis_idx = robot.body_names.index('pelvis')
    robot_pelvis_pos = robot.data.body_pos_w[:, robot_pelvis_idx]

    # Simple collision penalty: only penalize if robot pelvis gets too close to sphere
    sphere_pos = object2.data.root_pos_w
    collision_distance = torch.norm(sphere_pos - robot_pelvis_pos, dim=1)
    
    # Only penalize if very close (within 0.3m)
    collision_threshold = 0.3
    collision_penalty = torch.where(
        collision_distance < collision_threshold,
        -1.0 * (collision_threshold - collision_distance), # Linear penalty
        torch.tensor(0.0, device=env.device)
    )

    reward = collision_penalty

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
    """
    Reward terms for the kick_smallSphere skill.
    Simplified reward structure with main distance-based reward and minimal shaping rewards.
    """
    # Primary reward: simple distance-based (sphere_x - block_x)
    MainKickSmallSphereReward = RewTerm(
        func=main_kick_smallSphere_reward,
        weight=1.0,
        params={"normalise": True, "normaliser_name": "main_kick_smallSphere_reward"}
    )

    # Shaping reward for approach and pelvis height
    ApproachAndAlignmentReward = RewTerm(
        func=approach_and_alignment_reward,
        weight=0.3, # Reduced weight since main reward is now simpler
        params={"normalise": True, "normaliser_name": "approach_and_alignment_reward"}
    )

    # Shaping reward for y-alignment
    StayNearBlockYReward = RewTerm(
        func=stay_near_block_y_reward,
        weight=0.5, # Reduced weight since main reward is now simpler
        params={"normalise": True, "normaliser_name": "stay_near_block_y_reward"}
    )

    # Minimal collision avoidance
    CollisionAvoidanceReward = RewTerm(
        func=collision_avoidance_reward,
        weight=0.0, # Very low weight for minimal collision penalty
        params={"normalise": True, "normaliser_name": "collision_avoidance_reward"}
    )