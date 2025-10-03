
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def knock_over_cylinderColumn2_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the knock_over_cylinderColumn2 skill has been successfully completed.'''
    # Access Object2 (Cylinder Column 2) as specified in the object configuration.
    # This directly accesses the object from the scene, adhering to rule 2.
    object2 = env.scene['Object2']
    
    # Get the Z-component of Object2's root position in world coordinates.
    # This is an allowed absolute position check as Z-height is critical for determining if the object has fallen,
    # as per the rule "z is the only absolute position allowed. Use this sparingly, only when height is important to the skill."
    object2_z_pos = object2.data.root_pos_w[:, 2]
    
    # Define the success threshold for Object2's Z-position.
    # The object's radius is 0.3m (from object configuration), so when it's fully fallen, its center Z-position should be 0.3m.
    # A threshold of 0.4m allows for slight variations and ensures it's clearly on the floor.
    # This threshold is derived from the object's physical properties, not an arbitrary value, adhering to rule 4 and 6.
    success_z_threshold = 0.4
    
    # Check the success condition: Object2's Z-position is at or below the threshold.
    # This operation works correctly for batched environments, producing a boolean tensor, adhering to rule 3.
    condition = object2_z_pos <= success_z_threshold
    
    # Add pelvis height condition to ensure robot hasn't collapsed while knocking
    robot = env.scene["robot"]
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    min_pelvis_height = 0.6
    pelvis_height_condition = (pelvis_pos[:, 2] > min_pelvis_height)

    final_condition = condition & pelvis_height_condition

    # Check duration and save success states (0.5s duration)
    success = check_success_duration(env, final_condition, "knock_over_cylinderColumn2", duration=0.5)
    
    # If any environment has succeeded, save its success state, adhering to rule 5.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "knock_over_cylinderColumn2")
    
    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=knock_over_cylinderColumn2_success)
