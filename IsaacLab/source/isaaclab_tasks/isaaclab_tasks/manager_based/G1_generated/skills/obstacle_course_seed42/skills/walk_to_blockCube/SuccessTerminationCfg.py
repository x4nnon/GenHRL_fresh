
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_blockCube_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_blockCube skill has been successfully completed.

    Args:
        env: The environment instance

    Returns:
        Tensor of booleans indicating success for each environment
    '''
    # Access robot pelvis position
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]  # [num_envs, 3]

    # Access block cube position
    object5 = env.scene['Object5']
    object5_pos = object5.data.root_pos_w  # [num_envs, 3]

    # Success: abs(pelvis_x - block_x) < 1 and abs(pelvis_y - block_y) < 1
    dist_x = torch.abs(pelvis_pos[:, 0] - object5_pos[:, 0])
    dist_y = torch.abs(pelvis_pos[:, 1] - object5_pos[:, 1])
    condition = (dist_x < 1.0) & (dist_y < 1.0)

    # Check if the success condition is maintained for a duration of 0.5 seconds.
    success = check_success_duration(env, condition, "walk_to_blockCube", duration=0.5)

    # Save success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_blockCube")

    return success

class SuccessTerminationCfg:
    # Define the success termination term using the implemented success function.
    success = DoneTerm(func=walk_to_blockCube_success)
