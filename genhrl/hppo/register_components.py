"""
Register custom HPPO components (memory and agent) with SKRL.
"""

def register_hppo_components():
    """
    Register both DecisionPointMemory and HPPO agent with SKRL by adding them
    directly to SKRL's component registry.
    
    This makes both components available to be used in agent configurations.
    """
    print("Registering HPPO components (memory and agent) with SKRL")
    
    try:
        # Import Runner and our custom components
        from skrl.utils.runner.torch import Runner
        from genhrl.hppo.hppo import HPPO
        from genhrl.hppo.decision_point_memory import DecisionPointMemory
        
        # Store the original methods we need to patch
        original_component = Runner._component
        original_generate_agent = Runner._generate_agent
        
        # Create a patched version that handles our custom components
        def patched_component(self, name):
            # Accept both class objects and strings
            if not isinstance(name, str):
                return name
            name_lower = name.lower()
            
            # Check for our custom components
            if name_lower == "hppo":
                return HPPO
            elif name_lower == "decisionpointmemory":
                return DecisionPointMemory
            
            # Otherwise, use the original method
            return original_component(self, name)
        
        # Create a patched _generate_agent that handles the agent_cfg bug
        def patched_generate_agent(self, env, cfg, models):
            """
            Fixed version of _generate_agent that properly handles agent_cfg.
            This fixes the UnboundLocalError in SKRL's original implementation.
            """
            import copy
            from skrl import logger
            
            multi_agent = hasattr(env, 'possible_agents')
            device = env.device if hasattr(env, "device") else "cuda:0"
            num_envs = env.num_envs if hasattr(env, "num_envs") else 1
            possible_agents = env.possible_agents if multi_agent else ["agent"]
            observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
            action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}
            
            agent_cfg = copy.deepcopy(cfg.get("agent", {}))
            agent_class = agent_cfg.get("class", "").lower()
            
            if not agent_class:
                raise ValueError(f"No 'class' field defined in 'agent' cfg")
            
            # Remove class from config to avoid passing it as parameter
            if "class" in agent_cfg:
                del agent_cfg["class"]
            
            # Handle memory generation (replicate SKRL's logic)
            memory_cfg = copy.deepcopy(cfg.get("memory", {}))
            if not memory_cfg:
                logger.warning("No 'memory' field defined in cfg. Using the default generated configuration")
                memory_cfg = {"class": "RandomMemory", "memory_size": -1}
            
            # Get memory class and remove 'class' field
            try:
                memory_class = self._component(memory_cfg["class"])
                if "class" in memory_cfg:
                    del memory_cfg["class"]
            except (KeyError, ValueError):
                memory_class = self._component("RandomMemory")
                logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
            
            # Set memory size if not specified
            if memory_cfg.get("memory_size", -1) < 0:
                memory_cfg["memory_size"] = agent_cfg.get("rollouts", 16)
            
            # Create memory for single agent
            agent_id = possible_agents[0]
            memory = memory_class(num_envs=num_envs, device=device, **self._process_cfg(memory_cfg))
            
            # Handle agent configuration for standard agents (including our HPPO)
            if agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo", "hppo"]:
                # For HPPO, use PPO's default config as the base since they're compatible
                config_name = "ppo" if agent_class == "hppo" else agent_class
                default_config = self._component(f"{config_name}_DEFAULT_CONFIG").copy()
                default_config.update(self._process_cfg(agent_cfg))
                
                default_config.get("state_preprocessor_kwargs", {}).update(
                    {"size": observation_spaces[agent_id], "device": device}
                )
                default_config.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
                
                if default_config.get("exploration", {}).get("noise", None):
                    default_config["exploration"]["noise"] = default_config["exploration"]["noise"](
                        **default_config["exploration"].get("noise_kwargs", {})
                    )
                if default_config.get("smooth_regularization_noise", None):
                    default_config["smooth_regularization_noise"] = default_config["smooth_regularization_noise"](
                        **default_config.get("smooth_regularization_noise_kwargs", {})
                    )
                
                agent_kwargs = {
                    "models": models[agent_id],
                    "memory": memory,
                    "observation_space": observation_spaces[agent_id],
                    "action_space": action_spaces[agent_id],
                }
                
                final_agent_cfg = default_config
            else:
                # Fallback for other agent types not explicitly handled
                agent_kwargs = {
                    "models": models[agent_id],
                    "memory": memory,
                    "observation_space": observation_spaces[agent_id],
                    "action_space": action_spaces[agent_id],
                }
                final_agent_cfg = agent_cfg
            
            # Get the agent class using our patched component method
            try:
                AgentClass = self._component(agent_class)
            except ValueError as e:
                print(f"Error finding agent class '{agent_class}': {e}")
                raise
            
            # Create and return the agent
            return AgentClass(cfg=final_agent_cfg, device=device, **agent_kwargs)
        
        # Apply the monkey patches
        Runner._component = patched_component
        Runner._generate_agent = patched_generate_agent
        
        print("HPPO components successfully registered with SKRL")
        print("Applied fixes for SKRL Runner bugs")
        return True
    except Exception as e:
        print(f"Error registering HPPO components: {e}")
        import traceback
        traceback.print_exc()
        return False 