"""
Test Runner for GenHRL README Validation

This module runs all tests to validate that the functionality described
in the README works correctly after installation.
"""

import sys
import traceback
from pathlib import Path

def run_installation_tests():
    """Run installation verification tests."""
    print("🔍 Running installation verification tests...")
    try:
        from .test_installation import TestInstallationVerification
        
        test = TestInstallationVerification()
        test.test_core_imports()
        test.test_training_imports()
        test.test_additional_imports()
        test.test_api_key_handling()
        test.test_full_installation_verification()
        test.test_quick_start_verification()
        test.test_package_structure()
        test.test_version_compatibility()
        
        print("✅ Installation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Installation tests failed: {e}")
        traceback.print_exc()
        return False

def run_workflow_tests():
    """Run workflow functionality tests."""
    print("🔍 Running workflow functionality tests...")
    try:
        from .test_workflows import TestWorkflowFunctionality, TestREADMEExamples
        
        # Test workflow functionality
        test_workflows = TestWorkflowFunctionality()
        test_workflows.setup_method()
        
        try:
            test_workflows.test_create_task_with_workflow_signature()
            test_workflows.test_main_create_steps_example_signature()
            test_workflows.test_remove_all_previous_skills_signature()
            test_workflows.test_task_manager_initialization_from_readme()
            test_workflows.test_hierarchical_code_generator_configurations()
            test_workflows.test_skill_library_management_from_readme()
            test_workflows.test_robot_configuration_from_readme()
            test_workflows.test_utility_functions_from_readme()
            test_workflows.test_training_integration_from_readme()
            test_workflows.test_verification_settings_from_readme()
            
            # Test README examples
            test_examples = TestREADMEExamples()
            test_examples.setup_method()
            
            try:
                test_examples.test_readme_programmatic_api_example()
                test_examples.test_readme_step_by_step_example()
                test_examples.test_readme_debugging_workflow()
                
            finally:
                test_examples.teardown_method()
            
        finally:
            test_workflows.teardown_method()
        
        print("✅ Workflow tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Workflow tests failed: {e}")
        traceback.print_exc()
        return False

def run_configuration_tests():
    """Run configuration and structure tests."""
    print("🔍 Running configuration and structure tests...")
    try:
        from .test_config_and_structure import (
            TestFileStructureGeneration,
            TestConfigurationSettings,
            TestEnvironmentVariables,
            TestTaskConfigStructure
        )
        
        # Test file structure
        test_structure = TestFileStructureGeneration()
        test_structure.setup_method()
        try:
            test_structure.test_expected_file_structure_components()
            test_structure.test_agent_configuration_files()
            test_structure.test_skill_specific_configurations()
        finally:
            test_structure.teardown_method()
        
        # Test configuration settings
        test_config = TestConfigurationSettings()
        test_config.test_hierarchy_levels_configuration()
        test_config.test_verification_settings_configuration()
        test_config.test_llm_provider_configurations()
        test_config.test_training_configuration_settings()
        test_config.test_robot_configuration_settings()
        
        # Test environment variables
        test_env = TestEnvironmentVariables()
        test_env.test_api_key_environment_variable()
        test_env.test_object_config_path_environment_variable()
        
        # Test task config structure
        test_task_config = TestTaskConfigStructure()
        test_task_config.setup_method()
        try:
            test_task_config.test_task_config_interface()
            test_task_config.test_task_config_file_content_structure()
        finally:
            test_task_config.teardown_method()
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration tests failed: {e}")
        traceback.print_exc()
        return False

def validate_readme_claims():
    """Validate specific claims made in the README."""
    print("🔍 Validating README claims...")
    
    claims_validated = 0
    total_claims = 0
    
    # Claim 1: Core components can be imported
    total_claims += 1
    try:
        import importlib
        generation_module = importlib.import_module('genhrl.generation')
        
        core_components = ['TaskManager', 'SkillLibrary']
        for component in core_components:
            assert hasattr(generation_module, component), f"Missing component: {component}"
        
        claims_validated += 1
        print("✅ Claim: Core components can be imported")
        
    except Exception as e:
        print(f"❌ Claim failed: Core components import - {e}")
    
    # Claim 2: Workflow functions exist
    total_claims += 1
    try:
        generation_module = importlib.import_module('genhrl.generation')
        workflow_functions = ['create_task_with_workflow', 'remove_all_previous_skills']
        
        available_functions = []
        for func in workflow_functions:
            if hasattr(generation_module, func):
                available_functions.append(func)
        
        if available_functions:
            claims_validated += 1
            print(f"✅ Claim: Workflow functions exist - {available_functions}")
        else:
            print("❌ Claim failed: No workflow functions found")
        
    except Exception as e:
        print(f"❌ Claim failed: Workflow functions - {e}")
    
    # Claim 3: Training components exist
    total_claims += 1
    try:
        training_module = importlib.import_module('genhrl.training')
        training_components = ['TrainingOrchestrator']
        
        available_components = []
        for component in training_components:
            if hasattr(training_module, component):
                available_components.append(component)
        
        if available_components:
            claims_validated += 1
            print(f"✅ Claim: Training components exist - {available_components}")
        else:
            print("❌ Claim failed: No training components found")
        
    except Exception as e:
        print(f"❌ Claim failed: Training components - {e}")
    
    # Claim 4: Utility functions work
    total_claims += 1
    try:
        generation_module = importlib.import_module('genhrl.generation')
        
        if hasattr(generation_module, 'strip_markdown_formatting'):
            strip_func = getattr(generation_module, 'strip_markdown_formatting')
            test_input = "```python\nprint('test')\n```"
            result = strip_func(test_input)
            assert "print('test')" in result
            assert "```" not in result
            
            claims_validated += 1
            print("✅ Claim: Utility functions work")
        else:
            print("❌ Claim failed: strip_markdown_formatting not found")
        
    except Exception as e:
        print(f"❌ Claim failed: Utility functions - {e}")
    
    # Claim 5: Robot configuration exists
    total_claims += 1
    try:
        generation_module = importlib.import_module('genhrl.generation')
        
        if hasattr(generation_module, 'get_available_robots'):
            robots_func = getattr(generation_module, 'get_available_robots')
            robots = robots_func()
            assert isinstance(robots, list)
            assert len(robots) > 0
            
            claims_validated += 1
            print(f"✅ Claim: Robot configuration exists - {robots}")
        else:
            print("❌ Claim failed: get_available_robots not found")
        
    except Exception as e:
        print(f"❌ Claim failed: Robot configuration - {e}")
    
    print(f"\n📊 README Claims Validation: {claims_validated}/{total_claims} claims validated")
    return claims_validated == total_claims

def run_complete_validation():
    """Run complete validation of README functionality."""
    print("🚀 Starting complete GenHRL README validation...")
    print("=" * 60)
    
    results = {
        'installation': False,
        'workflows': False,
        'configuration': False,
        'readme_claims': False
    }
    
    # Run all test suites
    results['installation'] = run_installation_tests()
    print()
    
    results['workflows'] = run_workflow_tests()
    print()
    
    results['configuration'] = run_configuration_tests()
    print()
    
    results['readme_claims'] = validate_readme_claims()
    print()
    
    # Generate summary report
    print("=" * 60)
    print("📋 VALIDATION SUMMARY REPORT")
    print("=" * 60)
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper().replace('_', ' ')}: {status}")
    
    print(f"\nOVERALL RESULT: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("🎉 ALL README FUNCTIONALITY VALIDATED SUCCESSFULLY!")
        print("\n✅ Installation instructions work")
        print("✅ Programmatic API examples work")
        print("✅ Configuration options work")
        print("✅ File structure is correct")
        print("✅ All README claims are accurate")
        return True
    else:
        print("⚠️ SOME TESTS FAILED - README may need updates")
        print("\n📝 What this means:")
        print("- Some functionality described in README may not work as documented")
        print("- Users may encounter issues following README instructions")
        print("- Consider updating README or fixing implementation")
        return False

def quick_validation():
    """Run a quick validation focusing on core functionality."""
    print("⚡ Running quick validation...")
    
    try:
        # Test core imports
        from genhrl.generation import TaskManager, SkillLibrary
        print("✅ Core imports work")
        
        # Test basic initialization
        import tempfile
        test_dir = tempfile.mkdtemp()
        task_manager = TaskManager(test_dir, "test_key", "G1")
        print("✅ TaskManager initialization works")
        
        skill_library = SkillLibrary(f"{test_dir}/test_library.json")
        print("✅ SkillLibrary initialization works")
        
        # Test method existence
        assert hasattr(task_manager, 'create_task_from_description')
        assert hasattr(skill_library, 'list_primitive_skills')
        print("✅ Expected methods exist")
        
        print("\n🎉 Quick validation PASSED!")
        print("👍 Basic README functionality is working")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick validation FAILED: {e}")
        print("🚨 Basic README functionality has issues")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """Run tests based on command line arguments."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        success = quick_validation()
    else:
        success = run_complete_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)