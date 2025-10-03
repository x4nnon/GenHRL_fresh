# Hierarchical Training Scripts

This directory contains the scripts for training the hierarchical policies.

## `train_l1.py`

This script is used to train a Level 1 (L1) composite skill, which is a policy that learns to select from a set of pre-trained Level 0 (L0) primitive skills.

### Execution Flow

1.  **Initialization and Configuration:**
    *   The script starts by parsing command-line arguments, including the task name, robot name, and other training parameters.
    *   It then sets up the necessary paths for the task, skills, and skill library.
    *   The Isaac Sim application is launched.

2.  **Environment Creation:**
    *   The script creates an Isaac Lab environment using `gym.make`. The environment is determined by the `--task` argument.
    *   The environment is then wrapped with `SkrlVecEnvWrapper`, which is a standard wrapper for using Isaac Lab environments with `skrl`.

3.  **Hierarchical Wrapper Application:**
    *   The script loads the skill library to identify the sub-skills of the composite skill being trained.
    *   It then collects the checkpoint paths and registered names of the sub-skills.
    *   If sub-skill checkpoints are found, the environment is wrapped with `HierarchicalVecActionWrapper`. This wrapper is the core of the hierarchical training setup. It takes the following key arguments:
        *   `sub_policy_checkpoint_paths`: A list of paths to the pre-trained sub-skill policies.
        *   `sub_policy_registered_names`: A list of the registered names of the sub-skills.
        *   `steps_per_l0_policy`: The number of steps to execute a selected sub-skill policy before the higher-level policy can select a new one.
        *   `l1_action_frequency`: The frequency at which the higher-level policy makes decisions.

4.  **Custom `skrl` Component Registration:**
    *   The script calls `register_hppo_components()` to register the custom `HPPO` agent and `DecisionPointMemory` with `skrl`.

5.  **Agent Configuration:**
    *   The agent configuration is updated to use the custom `HPPO` agent and `DecisionPointMemory`.
    *   The `DecisionPointMemory` is configured to filter out non-decision steps, so that the `HPPO` agent only learns from the steps where it makes a decision.

6.  **`skrl` Runner Instantiation:**
    *   The `skrl` `HPPORunner` is instantiated with the wrapped environment and the modified agent configuration.

7.  **Training:**
    *   The `runner.run()` method is called to start the training process.

### How `skrl` Handles Hierarchical Training

The `skrl` framework is not inherently designed for hierarchical reinforcement learning. However, the `HierarchicalVecActionWrapper` and `DecisionPointMemory` classes provide a clever workaround to enable hierarchical training with `skrl`.

Here's how it works:

1.  **Action Space Modification:** The `HierarchicalVecActionWrapper` modifies the action space of the environment to be a discrete space where each action corresponds to selecting a sub-skill. The higher-level `HPPO` agent learns to select the best sub-skill to execute at each decision point.
2.  **Sub-skill Execution:** When the `HPPO` agent selects a sub-skill, the `HierarchicalVecActionWrapper` loads the corresponding pre-trained policy and executes it for a fixed number of steps (`steps_per_l0_policy`).
3.  **Decision Point Filtering:** The `DecisionPointMemory` ensures that the `HPPO` agent only learns from the steps where it makes a decision. This is done by filtering out the steps where the sub-skill policies are being executed.
4.  **Reward and Observation Handling:** The `HierarchicalVecActionWrapper` is responsible for handling the rewards and observations at each level of the hierarchy. It passes the appropriate rewards and observations to the `HPPO` agent at each decision point.

## `train_l2.py`

This script is used to train a Level 2 (L2) composite skill, which is a policy that learns to select from a set of pre-trained Level 1 (L1) composite skills.

### Execution Flow

The execution flow of `train_l2.py` is very similar to `train_l1.py`, with the main difference being that it uses the `HierarchicalVecActionWrapperL2` to wrap the environment. This wrapper is responsible for loading and managing the pre-trained L1 policies.

The `train_l2.py` script also uses the custom `HPPORunner` to ensure that the correct observations are passed to the wrappers.
