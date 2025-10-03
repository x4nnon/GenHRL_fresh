
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def walk_to_cylinderColumn4_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the walk_to_cylinderColumn4 skill has been successfully completed.
    Mirrors the column1 success logic but against Object4.
    '''
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]

    object4 = env.scene['Object4']
    object4_pos = object4.data.root_pos_w

    # Relative distances
    abs_diff_x = torch.abs(object4_pos[:, 0] - pelvis_pos[:, 0])
    abs_diff_y = torch.abs(object4_pos[:, 1] - pelvis_pos[:, 1])

    # Same target band as column1: X in [0.4, 0.6], Y < 0.2
    condition_x = abs_diff_x <= 0.6
    condition_y = (abs_diff_y < 0.5)

    success_condition = condition_x & condition_y

    success = check_success_duration(env, success_condition, "walk_to_cylinderColumn4", duration=0.2)
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "walk_to_cylinderColumn4")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=walk_to_cylinderColumn4_success)
