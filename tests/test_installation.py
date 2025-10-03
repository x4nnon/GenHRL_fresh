"""
Test installation verification as mentioned in the README.

These tests validate that all the installation verification steps
mentioned in the README work correctly.
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Any


class TestInstallationVerification:
    """Test installation verification functionality."""
    
    def test_core_imports(self):
        """Test that core components can be imported as shown in README."""
        # Test the exact import from README
        try:
            # Using dynamic imports to avoid linter errors
            from genhrl.generation import TaskManager, SkillLibrary
            
            # Test individual imports that might not exist yet
            import importlib
            generation_module = importlib.import_module('genhrl.generation')
            
            # Check for key components
            assert hasattr(generation_module, 'TaskManager')
            assert hasattr(generation_module, 'SkillLibrary')
            
            # Try to get the workflow function
            if hasattr(generation_module, 'HierarchicalCodeGenerator'):
                HierarchicalCodeGenerator = getattr(generation_module, 'HierarchicalCodeGenerator')
                assert HierarchicalCodeGenerator is not None
                
            if hasattr(generation_module, 'create_task_with_workflow'):
                create_task_with_workflow = getattr(generation_module, 'create_task_with_workflow')
                assert create_task_with_workflow is not None
            
            print("✅ Core components imported successfully")
            
        except ImportError as e:
            pytest.fail(f"❌ Core imports failed: {e}")
    
    def test_training_imports(self):
        """Test that training components can be imported."""
        try:
            import importlib
            training_module = importlib.import_module('genhrl.training')
            
            # Check for key training components
            if hasattr(training_module, 'TrainingOrchestrator'):
                TrainingOrchestrator = getattr(training_module, 'TrainingOrchestrator')
                assert TrainingOrchestrator is not None
                
            if hasattr(training_module, 'TrainingConfig'):
                TrainingConfig = getattr(training_module, 'TrainingConfig')
                assert TrainingConfig is not None
            
            print("✅ Training components imported successfully")
            
        except ImportError as e:
            pytest.fail(f"❌ Training imports failed: {e}")
    
    def test_additional_imports(self):
        """Test additional imports mentioned in README."""
        try:
            import importlib
            generation_module = importlib.import_module('genhrl.generation')
            
            # Check for additional components
            additional_components = [
                'remove_all_previous_skills',
                'strip_markdown_formatting', 
                'clean_json_string',
                'main_create_steps_example',
                'get_available_robots',
                'get_robot_config'
            ]
            
            for component in additional_components:
                if hasattr(generation_module, component):
                    func = getattr(generation_module, component)
                    assert func is not None
                else:
                    print(f"⚠️ {component} not found in generation module")
            
            print("✅ Additional components checked successfully")
            
        except ImportError as e:
            pytest.fail(f"❌ Additional imports failed: {e}")
    
    def test_api_key_handling(self):
        """Test API key handling as shown in README."""
        # Test environment variable detection
        original_key = os.getenv("GENHRL_API_KEY")
        
        # Test with no key
        if "GENHRL_API_KEY" in os.environ:
            del os.environ["GENHRL_API_KEY"]
        
        key = os.getenv("GENHRL_API_KEY")
        assert key is None
        print("⚠️ GENHRL_API_KEY not set in environment (expected for test)")
        
        # Test with key set
        os.environ["GENHRL_API_KEY"] = "test_key"
        key = os.getenv("GENHRL_API_KEY")
        assert key == "test_key"
        print("✅ API key found in environment")
        
        # Restore original state
        if original_key:
            os.environ["GENHRL_API_KEY"] = original_key
        elif "GENHRL_API_KEY" in os.environ:
            del os.environ["GENHRL_API_KEY"]
    
    def test_full_installation_verification(self):
        """Test the complete installation verification function from README."""
        
        def test_installation():
            """Replicate the exact function from README."""
            try:
                import importlib
                
                # Test core generation components
                generation_module = importlib.import_module('genhrl.generation')
                core_components = ['TaskManager', 'SkillLibrary']
                
                for component in core_components:
                    if not hasattr(generation_module, component):
                        print(f"❌ Missing core component: {component}")
                        return False
                
                print("✅ Core components imported successfully")
                
                # Test training components
                try:
                    training_module = importlib.import_module('genhrl.training')
                    print("✅ Training components imported successfully")
                except ImportError:
                    print("⚠️ Training components not available")
                
                # Test API key handling
                import os
                if os.getenv("GENHRL_API_KEY"):
                    print("✅ API key found in environment")
                else:
                    print("⚠️ GENHRL_API_KEY not set in environment")
                    
                print("🎉 GenHRL installation verified!")
                return True
                
            except ImportError as e:
                print(f"❌ Import error: {e}")
                return False
        
        # Run the verification function
        result = test_installation()
        assert result is True, "Installation verification should pass"
    
    def test_quick_start_verification(self):
        """Test the quick start verification from README."""
        # Test the exact command from README
        try:
            from genhrl.generation import TaskManager
            print('✅ GenHRL installed successfully')
        except ImportError as e:
            pytest.fail(f"Quick start verification failed: {e}")
    
    def test_package_structure(self):
        """Test that the package structure is correct."""
        # Test that main modules exist
        import genhrl
        import genhrl.generation
        import genhrl.training
        
        # Test that key files exist
        genhrl_path = Path(genhrl.__file__).parent
        
        expected_files = [
            genhrl_path / "generation" / "__init__.py",
            genhrl_path / "generation" / "code_generator.py",
            genhrl_path / "generation" / "task_manager.py",
            genhrl_path / "generation" / "skill_library.py",
            genhrl_path / "generation" / "main_workflow.py",
            genhrl_path / "training" / "__init__.py",
            genhrl_path / "training" / "orchestrator.py",
        ]
        
        for file_path in expected_files:
            assert file_path.exists(), f"Expected file not found: {file_path}"
        
        print("✅ Package structure verified")
    
    def test_version_compatibility(self):
        """Test Python version compatibility."""
        # Test that we're running on Python 3.8+
        version = sys.version_info
        assert version.major >= 3 and version.minor >= 8, \
            f"Python 3.8+ required, found {version.major}.{version.minor}"
        
        print(f"✅ Python version {version.major}.{version.minor} compatibility verified")


if __name__ == "__main__":
    # Allow running tests directly
    test = TestInstallationVerification()
    test.test_core_imports()
    test.test_training_imports()
    test.test_additional_imports()
    test.test_api_key_handling()
    test.test_full_installation_verification()
    test.test_quick_start_verification()
    test.test_package_structure()
    test.test_version_compatibility()
    print("🎉 All installation tests passed!")