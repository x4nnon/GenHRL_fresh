
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

def kick_smallSphere_away_from_wall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the kick_smallSphere_away_from_wall skill has been successfully completed.'''

    # Reasoning: Accessing the robot object to get its body part positions.
    robot = env.scene["robot"]

    # Reasoning: Accessing the 'pelvis' body part as specified in the success criteria plan.
    pelvis_idx = robot.body_names.index('pelvis')
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx] # Shape: [num_envs, 3]

    # Reasoning: Accessing Object2 (small sphere) as specified in the success criteria plan.
    small_sphere = env.scene['Object2']
    small_sphere_pos = small_sphere.data.root_pos_w # Shape: [num_envs, 3]

    # Reasoning: Accessing Object4 (high wall) to measure sphere displacement relative to it.
    high_wall = env.scene['Object4']
    high_wall_pos = high_wall.data.root_pos_w # Shape: [num_envs, 3]

    # Reasoning: Accessing Object5 (block cube) to position the robot for the next skill.
    block_cube = env.scene['Object5']
    block_cube_pos = block_cube.data.root_pos_w # Shape: [num_envs, 3]

    # --- Success Condition 1: Small sphere moved significantly away from the high wall ---
    # Reasoning: The task is to kick the sphere away from the wall. Given the object order,
    # this means increasing the x-distance between the sphere and the high wall (Object4).
    # The initial setup places the high wall (Object4) before the small sphere (Object2) in the x-axis.
    # A successful kick should increase the x-distance between them.
    # Object dimensions: small sphere radius = 0.2m. High wall x-dimension = 0.3m.
    # The initial separation between object centers is 3m.
    # A displacement of 1.5m further away from the wall is desired.
    # So, the current x-distance between sphere and wall should be at least (3m + 1.5m) = 4.5m.
    # This is a relative distance check and does not rely on initial positions, adhering to rules.
    sphere_wall_x_distance = small_sphere_pos[:, 0] - high_wall_pos[:, 0]
    sphere_moved_away = sphere_wall_x_distance > 4.5 # Target: 3m initial separation + 1.5m displacement

    # --- Success Condition 2: Robot's pelvis is past the kicked sphere ---
    # Reasoning: The robot should have completed the kick and moved past the sphere.
    # This is a relative x-position check between the robot's pelvis and the small sphere.
    # Small sphere radius is 0.2m. The robot's pelvis should be slightly past the sphere's center.
    # A threshold of 0.2m past the sphere's center ensures it's clearly past it, accounting for sphere size.
    robot_past_sphere = pelvis_pos[:, 0] > (small_sphere_pos[:, 0] + 0.2)

    # --- Success Condition 3: Robot's pelvis is before the block cube ---
    # Reasoning: The robot should be positioned for the next skill (jumping on the block),
    # meaning it should not have overshot the block.
    # This is a relative x-position check between the robot's pelvis and the block cube.
    # Block cube dimension is 0.5m. Robot pelvis should be before the block's center,
    # allowing for some clearance (e.g., 0.5m before the block's center).
    # This threshold accounts for the block's half-size (0.25m) and an additional clearance.
    robot_before_block = pelvis_pos[:, 0] < (block_cube_pos[:, 0] - 0.5)

    # --- Combine all success conditions ---
    # Reasoning: All conditions must be met for a successful skill completion.
    condition = sphere_moved_away & robot_past_sphere & robot_before_block

    # Reasoning: Using check_success_duration to ensure the conditions are met for a sustained period.
    # Duration of 0.5 seconds is specified in the plan.
    success = check_success_duration(env, condition, "kick_smallSphere_away_from_wall", duration=0.5)

    # Reasoning: Saving success states for environments that have successfully completed the skill.
    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "kick_smallSphere_away_from_wall")

    return success

class SuccessTerminationCfg:
    # Reasoning: Registering the success function as a termination condition.
    success = DoneTerm(func=kick_smallSphere_away_from_wall_success)
