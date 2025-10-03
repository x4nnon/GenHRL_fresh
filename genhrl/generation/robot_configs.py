"""
Robot configurations for different robots available in IsaacLab.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class RobotConfig:
    """Configuration for a specific robot."""
    name: str
    asset_cfg: str  # The Isaac Lab asset configuration import
    spawn_cfg: str  # The spawn configuration
    joint_names: Dict[str, List[str]]  # Joint name patterns by body part
    default_joint_pos: Dict[str, float]  # Default joint positions
    actuator_configs: Dict[str, Dict[str, Any]]  # Actuator configurations
    prim_path_template: str = "{ENV_REGEX_NS}/Robot"  # Default prim path
    height_scanner_path: str = "{ENV_REGEX_NS}/Robot/base"  # Default height scanner path
    contact_sensor_path: str = "{ENV_REGEX_NS}/Robot/.*"  # Default contact sensor path


# Robot configurations for different robots
ROBOT_CONFIGS = {
    "G1": RobotConfig(
        name="G1",
        asset_cfg="G1_CFG",
        spawn_cfg="G1_CFG",
        joint_names={
            "legs": [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
            "torso": ["torso_joint"],
            "arms": [".*_shoulder_.*", ".*_elbow_.*"],
            "fingers": [".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint", 
                       ".*_zero_joint", ".*_one_joint", ".*_two_joint"]
        },
        default_joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.28,
            ".*_knee": 0.79,
            ".*_ankle": -0.52,
            "torso": 0.0,
            ".*_shoulder_pitch": 0.28,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.52,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
                "effort_limit": 300,
                "velocity_limit": 100.0,
                "stiffness": {
                    ".*_hip_yaw": 150.0,
                    ".*_hip_roll": 150.0,
                    ".*_hip_pitch": 200.0,
                    ".*_knee": 200.0,
                    "torso": 200.0
                },
                "damping": {
                    ".*_hip_yaw": 5.0,
                    ".*_hip_roll": 5.0,
                    ".*_hip_pitch": 5.0,
                    ".*_knee": 5.0,
                    "torso": 5.0
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/torso_link"
    ),
    
    "H1": RobotConfig(
        name="H1",
        asset_cfg="UNITREE_H1_CFG",
        spawn_cfg="UNITREE_H1_CFG", 
        joint_names={
            "legs": [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
            "torso": ["torso_joint"],
            "arms": [".*_shoulder_.*", ".*_elbow_.*"]
        },
        default_joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0, 
            ".*_hip_pitch": -0.4,
            ".*_knee": 0.8,
            ".*_ankle": -0.4,
            "torso": 0.0,
            ".*_shoulder_pitch": 0.0,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.0,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
                "effort_limit": 300,
                "velocity_limit": 100.0,
                "stiffness": {
                    ".*_hip_yaw": 200.0,
                    ".*_hip_roll": 200.0,
                    ".*_hip_pitch": 300.0,
                    ".*_knee": 300.0,
                    ".*_ankle": 40.0
                },
                "damping": {
                    ".*_hip_yaw": 5.0,
                    ".*_hip_roll": 5.0,
                    ".*_hip_pitch": 6.0,
                    ".*_knee": 6.0,
                    ".*_ankle": 1.0
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/torso_link"
    ),
    
    "Anymal_C": RobotConfig(
        name="Anymal_C",
        asset_cfg="ANYMAL_C_CFG",
        spawn_cfg="ANYMAL_C_CFG",
        joint_names={
            "legs": [".*_HAA", ".*_HFE", ".*_KFE"]
        },
        default_joint_pos={
            ".*_HAA": 0.0,
            ".*_HFE": 0.4,
            ".*_KFE": -0.8,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_HAA", ".*_HFE", ".*_KFE"],
                "effort_limit": 80.0,
                "velocity_limit": 7.5,
                "stiffness": {
                    ".*_HAA": 40.0,
                    ".*_HFE": 40.0,
                    ".*_KFE": 40.0
                },
                "damping": {
                    ".*_HAA": 1.0,
                    ".*_HFE": 1.0,
                    ".*_KFE": 1.0
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/base"
    ),
    
    "Anymal_B": RobotConfig(
        name="Anymal_B", 
        asset_cfg="ANYMAL_B_CFG",
        spawn_cfg="ANYMAL_B_CFG",
        joint_names={
            "legs": [".*_HAA", ".*_HFE", ".*_KFE"]
        },
        default_joint_pos={
            ".*_HAA": 0.0,
            ".*_HFE": 0.4,
            ".*_KFE": -0.8,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_HAA", ".*_HFE", ".*_KFE"],
                "effort_limit": 80.0,
                "velocity_limit": 7.5,
                "stiffness": {
                    ".*_HAA": 40.0,
                    ".*_HFE": 40.0,
                    ".*_KFE": 40.0
                },
                "damping": {
                    ".*_HAA": 1.0,
                    ".*_HFE": 1.0,
                    ".*_KFE": 1.0
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/base"
    ),
    
    "Anymal_D": RobotConfig(
        name="Anymal_D",
        asset_cfg="ANYMAL_D_CFG", 
        spawn_cfg="ANYMAL_D_CFG",
        joint_names={
            "legs": [".*_HAA", ".*_HFE", ".*_KFE"]
        },
        default_joint_pos={
            ".*_HAA": 0.0,
            ".*_HFE": 0.4,
            ".*_KFE": -0.8,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_HAA", ".*_HFE", ".*_KFE"],
                "effort_limit": 80.0,
                "velocity_limit": 7.5,
                "stiffness": {
                    ".*_HAA": 40.0,
                    ".*_HFE": 40.0,
                    ".*_KFE": 40.0
                },
                "damping": {
                    ".*_HAA": 1.0,
                    ".*_HFE": 1.0,
                    ".*_KFE": 1.0
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/base"
    ),
    
    "A1": RobotConfig(
        name="A1",
        asset_cfg="UNITREE_A1_CFG",
        spawn_cfg="UNITREE_A1_CFG",
        joint_names={
            "legs": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
        },
        default_joint_pos={
            ".*_hip_joint": 0.1,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                "effort_limit": 33.5,
                "velocity_limit": 21.0,
                "stiffness": {
                    ".*_hip_joint": 20.0,
                    ".*_thigh_joint": 20.0,
                    ".*_calf_joint": 20.0
                },
                "damping": {
                    ".*_hip_joint": 0.5,
                    ".*_thigh_joint": 0.5,
                    ".*_calf_joint": 0.5
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/base"
    ),
    
    "Go1": RobotConfig(
        name="Go1",
        asset_cfg="UNITREE_GO1_CFG",
        spawn_cfg="UNITREE_GO1_CFG",
        joint_names={
            "legs": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
        },
        default_joint_pos={
            ".*_hip_joint": 0.1,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                "effort_limit": 23.5,
                "velocity_limit": 30.0,
                "stiffness": {
                    ".*_hip_joint": 20.0,
                    ".*_thigh_joint": 20.0,
                    ".*_calf_joint": 20.0
                },
                "damping": {
                    ".*_hip_joint": 0.5,
                    ".*_thigh_joint": 0.5,
                    ".*_calf_joint": 0.5
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/base"
    ),
    
    "Go2": RobotConfig(
        name="Go2",
        asset_cfg="UNITREE_GO2_CFG",
        spawn_cfg="UNITREE_GO2_CFG",
        joint_names={
            "legs": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
        },
        default_joint_pos={
            ".*_hip_joint": 0.1,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                "effort_limit": 23.5,
                "velocity_limit": 30.0,
                "stiffness": {
                    ".*_hip_joint": 20.0,
                    ".*_thigh_joint": 20.0,
                    ".*_calf_joint": 20.0
                },
                "damping": {
                    ".*_hip_joint": 0.5,
                    ".*_thigh_joint": 0.5,
                    ".*_calf_joint": 0.5
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/base"
    ),
    
    "Spot": RobotConfig(
        name="Spot",
        asset_cfg="SPOT_CFG",
        spawn_cfg="SPOT_CFG",
        joint_names={
            "legs": [".*_hip_.*", ".*_upper_leg_joint", ".*_lower_leg_joint"]
        },
        default_joint_pos={
            "fl_hx": 0.0,
            "fl_hy": 0.9,
            "fl_kn": -1.8,
            "fr_hx": 0.0,
            "fr_hy": 0.9,
            "fr_kn": -1.8,
            "rl_hx": 0.0,
            "rl_hy": 0.9,
            "rl_kn": -1.8,
            "rr_hx": 0.0,
            "rr_hy": 0.9,
            "rr_kn": -1.8,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_hip_.*", ".*_upper_leg_joint", ".*_lower_leg_joint"],
                "effort_limit": 138.0,
                "velocity_limit": 8.5,
                "stiffness": {
                    ".*_hip_.*": 200.0,
                    ".*_upper_leg_joint": 200.0,
                    ".*_lower_leg_joint": 200.0
                },
                "damping": {
                    ".*_hip_.*": 5.0,
                    ".*_upper_leg_joint": 5.0,
                    ".*_lower_leg_joint": 5.0
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/base"
    ),
    
    "Digit": RobotConfig(
        name="Digit",
        asset_cfg="DIGIT_CFG",
        spawn_cfg="DIGIT_CFG",
        joint_names={
            "legs": [".*_hip_.*", ".*_knee_joint", ".*_ankle_joint"],
            "arms": [".*_shoulder_.*", ".*_elbow_joint"]
        },
        default_joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.5,
            ".*_knee": 1.0,
            ".*_ankle": -0.5,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_pitch": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.0,
        },
        actuator_configs={
            "legs": {
                "joint_names_expr": [".*_hip_.*", ".*_knee_joint", ".*_ankle_joint"],
                "effort_limit": 150.0,
                "velocity_limit": 20.0,
                "stiffness": {
                    ".*_hip_.*": 200.0,
                    ".*_knee": 200.0,
                    ".*_ankle": 200.0
                },
                "damping": {
                    ".*_hip_.*": 5.0,
                    ".*_knee": 5.0,
                    ".*_ankle": 5.0
                }
            }
        },
        height_scanner_path="{ENV_REGEX_NS}/Robot/torso"
    ),
    
    "Franka": RobotConfig(
        name="Franka",
        asset_cfg="FRANKA_PANDA_CFG",
        spawn_cfg="FRANKA_PANDA_CFG",
        joint_names={
            "arm": ["panda_joint[1-7]"],
            "hand": ["panda_finger_.*"]
        },
        default_joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.81,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.035,
        },
        actuator_configs={
            "arm": {
                "joint_names_expr": ["panda_joint[1-7]"],
                "effort_limit": 87.0,
                "velocity_limit": 2.175,
                "stiffness": 800.0,
                "damping": 40.0
            },
            "hand": {
                "joint_names_expr": ["panda_finger_.*"],
                "effort_limit": 200.0,
                "velocity_limit": 0.2,
                "stiffness": 400.0,
                "damping": 40.0
            }
        },
        prim_path_template="{ENV_REGEX_NS}/Robot",
        height_scanner_path="{ENV_REGEX_NS}/Robot/panda_link0",
        contact_sensor_path="{ENV_REGEX_NS}/Robot/panda_hand"
    ),
    
    "UR10": RobotConfig(
        name="UR10",
        asset_cfg="UR10_CFG",
        spawn_cfg="UR10_CFG",
        joint_names={
            "arm": [".*_shoulder_pan_joint", ".*_shoulder_lift_joint", ".*_elbow_joint", 
                   ".*_wrist_1_joint", ".*_wrist_2_joint", ".*_wrist_3_joint"]
        },
        default_joint_pos={
            ".*_shoulder_pan_joint": 0.0,
            ".*_shoulder_lift_joint": -1.712,
            ".*_elbow_joint": 1.712,
            ".*_wrist_1_joint": 0.0,
            ".*_wrist_2_joint": 0.0,
            ".*_wrist_3_joint": 0.0,
        },
        actuator_configs={
            "arm": {
                "joint_names_expr": [".*_shoulder_pan_joint", ".*_shoulder_lift_joint", ".*_elbow_joint", 
                                   ".*_wrist_1_joint", ".*_wrist_2_joint", ".*_wrist_3_joint"],
                "effort_limit": 150.0,
                "velocity_limit": 3.15,
                "stiffness": 400.0,
                "damping": 40.0
            }
        },
        prim_path_template="{ENV_REGEX_NS}/Robot",
        height_scanner_path="{ENV_REGEX_NS}/Robot/base_link"
    )
}


def get_robot_config(robot_name: str) -> RobotConfig:
    """Get robot configuration by name."""
    if robot_name not in ROBOT_CONFIGS:
        available_robots = ", ".join(ROBOT_CONFIGS.keys())
        raise ValueError(f"Robot '{robot_name}' not found. Available robots: {available_robots}")
    
    return ROBOT_CONFIGS[robot_name]


def get_available_robots() -> List[str]:
    """Get list of available robot names."""
    return list(ROBOT_CONFIGS.keys())


def get_robot_folder_name(robot_name: str) -> str:
    """Get the folder name for a robot (e.g., 'G1' -> 'G1_generated')."""
    return f"{robot_name}_generated"