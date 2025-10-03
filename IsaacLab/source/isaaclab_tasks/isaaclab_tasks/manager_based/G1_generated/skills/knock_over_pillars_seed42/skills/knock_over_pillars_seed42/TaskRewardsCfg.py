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


def knock_over_pillars_reward(env: ManagerBasedRLEnv, normalise: bool = True, normaliser_name: str = "knock_over_pillars_reward") -> torch.Tensor:
    """
    Primary reward for knocking over all 5 pillars.
    Gives +1 the FIRST time each pillar is detected as fallen within an episode.
    """
    # Get normalizer instance
    RewNormalizer = get_normalizer(env.device)

    # Access all 5 pillars
    # CRITICAL RULE: ALWAYS access object positions using env.scene['ObjectName'].data.root_pos_w
    total_reward = torch.zeros(env.num_envs, device=env.device)
    
    # From object configuration: Cylinder Column has radius of 0.3m
    # A pillar is considered fallen if its center Z is close to its radius
    pillar_fallen_threshold_z = 0.8  # 0.3m (radius) + 0.5m clearance
    
    # Build current fallen mask for all pillars: shape [num_envs, 5]
    fallen_now_list = []
    for i in range(1, 6):
        pillar = env.scene[f'Object{i}']
        pillar_z = pillar.data.root_pos_w[:, 2]
        fallen_i = pillar_z < pillar_fallen_threshold_z  # [num_envs]
        fallen_now_list.append(fallen_i.unsqueeze(-1))
    fallen_now = torch.cat(fallen_now_list, dim=-1)  # [num_envs, 5]

    # Initialize or reset per-episode previous fallen mask
    # We reset rows for envs at the start of an episode (episode_length_buf == 0)
    if not hasattr(env, "_pillars_fallen_prev"):
        env._pillars_fallen_prev = torch.zeros((env.num_envs, 5), dtype=torch.bool, device=env.device)
    else:
        if hasattr(env, "episode_length_buf"):
            at_start = (env.episode_length_buf < 2)
            if at_start.any():
                env._pillars_fallen_prev[at_start] = False
        else:
            print("No episode length buffer found")

    # Compute first-time events: fallen now AND not fallen before
    first_time_fall = fallen_now & (~env._pillars_fallen_prev)
    # Reward is number of first-time falls this step per env
    total_reward = first_time_fall.sum(dim=1).float()*10

    # Update memory so we don't reward again this episode
    env._pillars_fallen_prev |= fallen_now

    # CRITICAL RULE: ALWAYS implement proper reward normalization
    if normaliser_name not in RewNormalizer.stats:
        RewNormalizer.stats[normaliser_name] = RewardStats()

    if normalise:
        scaled_reward = RewNormalizer.normalize(normaliser_name, total_reward)
        RewNormalizer.update_stats(normaliser_name, total_reward)
        return scaled_reward
    return total_reward


@configclass
class TaskRewardsCfg:
    """
    Reward terms for the knock_over_pillars_seed42 skill.
    Single reward term for knocking over all pillars.
    """
    knock_over_pillars_reward = RewTerm(
        func=knock_over_pillars_reward,
        weight=1.0,
        params={"normalise": False, "normaliser_name": "knock_over_pillars_reward"}
    )