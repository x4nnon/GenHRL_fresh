"""
Register custom memories for hierarchical RL with SKRL.
"""
from genhrl.hppo.decision_point_memory import DecisionPointMemory

def register_decision_point_memory():
    """
    Register the DecisionPointMemory with SKRL by monkey patching the Runner._component method.
    
    This makes the memory type available to be used in agent configurations.
    """
    # Print confirmation
    print("Registering DecisionPointMemory with SKRL")
    
    try:
        # Import Runner to patch its _component method
        from skrl.utils.runner.torch import Runner
        
        # Store the original _component method
        original_component = Runner._component
        
        # Create a patched version that handles our custom memory
        def patched_component(self, name):
            # Accept both class objects and strings
            if not isinstance(name, str):
                # If it's already a class/type, return as-is
                return name
            
            name_lower = name.lower()
            
            # Check if it's our custom memory (support a few variants)
            if name_lower in ("decisionpointmemory", "decision_point_memory", "decision-point-memory"):
                return DecisionPointMemory
            
            # Otherwise, use the original method
            return original_component(self, name)
        
        # Apply the monkey patch
        Runner._component = patched_component
        
        print("DecisionPointMemory successfully registered via monkey patch")
        return True
    except Exception as e:
        print(f"Error registering DecisionPointMemory: {e}")
        return False 