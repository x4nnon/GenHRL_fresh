"""
Register HPPO agent with SKRL.
"""

def register_hppo_agent():
    """
    Register the HPPO agent with SKRL by monkey patching the Runner._component method.
    
    This makes the HPPO agent available to be used in agent configurations.
    """
    print("Registering HPPO agent with SKRL")
    
    try:
        # Import Runner and HPPO to patch the _component method
        from skrl.utils.runner.torch import Runner
        from genhrl.hppo.hppo import HPPO
        
        # Store the original _component method
        original_component = Runner._component
        
        # Create a patched version that handles our custom agent
        def patched_component(self, name):
            # Accept both class objects and strings
            if not isinstance(name, str):
                return name
            # Convert name to lowercase as SKRL does
            name_lower = name.lower()
            
            # Check if it's our custom agent
            if name_lower == "hppo":
                return HPPO
            
            # Otherwise, use the original method
            return original_component(self, name)
        
        # Apply the monkey patch
        Runner._component = patched_component
        
        print("HPPO agent successfully registered via monkey patch")
        return True
    except Exception as e:
        print(f"Error registering HPPO agent: {e}")
        return False 