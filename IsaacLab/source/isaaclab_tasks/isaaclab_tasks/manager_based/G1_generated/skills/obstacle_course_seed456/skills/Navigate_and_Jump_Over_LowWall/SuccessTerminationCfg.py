
from .base_success import save_success_state, check_success_duration
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.mdp import * 
import torch
from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from genhrl.generation.reward_normalizer import get_normalizer

from genhrl.generation.objects import get_object_volume

from isaaclab.envs import mdp
import genhrl.generation.mdp.rewards as custom_rewards
import genhrl.generation.mdp.terminations as custom_terminations
import genhrl.generation.mdp.observations as custom_observations
import genhrl.generation.mdp.events as custom_events
import genhrl.generation.mdp.curriculums as custom_curriculums

def Navigate_and_Jump_Over_LowWall_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''Determine if the Navigate_and_Jump_Over_LowWall skill has been successfully completed.'''

    # Access the robot object.
    # Requirement: ALWAYS access robot parts using: robot.data.body_pos_w[:, robot.body_names.index('part_name')]
    robot = env.scene["robot"]

    # Get indices and positions for required robot parts.
    # Requirement: ALWAYS access robot parts using robot.body_names.index('part_name')
    pelvis_idx = robot.body_names.index('pelvis')
    left_foot_idx = robot.body_names.index('left_ankle_roll_link')
    right_foot_idx = robot.body_names.index('right_ankle_roll_link')

    # Requirement: All operations must work with batched environments
    pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]

    # Access the required objects.
    # Requirement: ALWAYS access object positions using: env.scene['ObjectName'].data.root_pos_w
    # Requirement: Access objects directly - objects should always exist in the scene
    # Requirement: ONLY USE Object1...Object5 based on the object configuration
    low_wall = env.scene['Object3']  # Low wall for robot to jump over
    large_sphere = env.scene['Object1']  # Large sphere for robot to push

    # Requirement: YOU MUST ACCESS OBJECT LOCATIONS (instead of hard coding)USING THE APPROVED PATTERN
    low_wall_pos = low_wall.data.root_pos_w
    large_sphere_pos = large_sphere.data.root_pos_w

    # Hardcode object dimensions from the task description/object configuration.
    # Requirement: THERE IS NO way to access the SIZE of an object - if you need this. You must read this from the object config and hard code the value for it.
    low_wall_x_dim = 0.3  # x-dimension of low wall from task description
    large_sphere_radius = 1.0  # radius of large sphere from task description

    # Calculate success conditions based on relative distances.
    # Requirement: SUCCESS CRITERIA MUST ONLY use relative distances between objects and robot parts
    # Requirement: NEVER use hard-coded positions or arbitrary thresholds (except for object dimensions read from config)
    # Requirement: YOU MUST CONSIDER THE X, Y AND Z COMPONENTS OF DISTANCES SEPERATELY, INCLUDING THEIR THRESHOLDS.

    # Condition 1: Robot's pelvis is past the low wall's far edge in the x-axis.
    # Low wall's far edge x-position (relative to wall center)
    # This is a relative calculation based on wall position and its dimension.
    low_wall_far_edge_x = low_wall_pos[:, 0] + low_wall_x_dim / 2.0
    # Robot pelvis x-position must be greater than the far edge of the wall plus a small clearance.
    # Using a lenient threshold of 0.1m clearance.
    pelvis_past_wall_condition = pelvis_pos[:, 0] > (low_wall_far_edge_x + 0.1)

    # Condition 2: Robot's pelvis is within a reasonable distance of the large sphere's near edge in the x-axis.
    # Large sphere's near edge x-position (relative to sphere center)
    # This is a relative calculation based on sphere position and its radius.
    large_sphere_near_edge_x = large_sphere_pos[:, 0] - large_sphere_radius
    # Robot pelvis x-position must be less than the near edge of the sphere plus a tolerance.
    # This ensures the robot is positioned before the sphere, ready for the next skill.
    # Using a lenient threshold of 0.5m tolerance.
    pelvis_before_sphere_condition = pelvis_pos[:, 0] < (large_sphere_near_edge_x + 0.5)

    # Condition 3: Robot's feet are on the ground, indicating a stable landing.
    # Assuming ground is at z=0. A small tolerance (e.g., 0.1m) for feet contact.
    # Requirement: z_height = torch.abs(pos1[:, 2]) # z is the only absolute position allowed. Use this sparingly, only when height is important to the skill.
    left_foot_on_ground_condition = left_foot_pos[:, 2] < 0.1
    right_foot_on_ground_condition = right_foot_pos[:, 2] < 0.1
    feet_on_ground_condition = left_foot_on_ground_condition & right_foot_on_ground_condition

    # Combine all conditions for overall success.
    # Requirement: All operations must work with batched environments
    overall_success_condition = pelvis_past_wall_condition & pelvis_before_sphere_condition & feet_on_ground_condition

    # Check success duration and save success states.
    # Requirement: ALWAYS use check_success_duration and save_success_state
    success = check_success_duration(env, overall_success_condition, "Navigate_and_Jump_Over_LowWall", duration=0.5)

    if success.any():
        for env_id in torch.where(success)[0]:
            save_success_state(env, env_id, "Navigate_and_Jump_Over_LowWall")

    return success

class SuccessTerminationCfg:
    success = DoneTerm(func=Navigate_and_Jump_Over_LowWall_success)
