# GenHRL Test Suite

This directory contains comprehensive tests that validate all functionality described in the README.

## Test Overview

The test suite ensures that:

1. **Installation instructions work correctly**
2. **All code examples in README are functional**  
3. **API signatures match documentation**
4. **File structures are generated as described**
5. **Configuration options work as documented**

## Test Files

### Core Test Modules

- **`test_installation.py`** - Validates installation verification steps from README
- **`test_workflows.py`** - Tests workflow functions and README examples
- **`test_config_and_structure.py`** - Tests configuration and file structure generation
- **`test_runner.py`** - Comprehensive test runner with reporting

### Configuration

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`__init__.py`** - Test package initialization

## Running Tests

### Quick Validation

Test basic functionality quickly:

```bash
# Quick validation of core functionality
python -m tests.test_runner quick

# Or run with pytest
pytest tests/test_installation.py -v
```

### Full Test Suite

Run comprehensive validation:

```bash
# Complete validation (recommended)
python -m tests.test_runner

# Or with pytest
pytest tests/ -v

# Run specific test categories
pytest tests/ -m readme  # README validation tests only
pytest tests/ -m unit     # Unit tests only
pytest tests/ -m integration  # Integration tests only
```

### Installation Verification

Verify your installation matches README instructions:

```bash
# Test installation as described in README
python tests/test_installation.py

# Test that README examples work
python tests/test_workflows.py
```

## Test Categories

### 📦 Installation Tests (`test_installation.py`)

Validates that:
- ✅ All imports work as shown in README
- ✅ Package structure is correct
- ✅ Environment variables are handled properly
- ✅ Version compatibility requirements are met
- ✅ Installation verification function works

### 🔄 Workflow Tests (`test_workflows.py`)

Validates that:
- ✅ `create_task_with_workflow` function exists and works
- ✅ `TaskManager` API matches README examples
- ✅ `HierarchicalCodeGenerator` configurations work
- ✅ `SkillLibrary` management functions exist
- ✅ Robot configuration functions work
- ✅ Utility functions work as documented

### ⚙️ Configuration Tests (`test_config_and_structure.py`)

Validates that:
- ✅ File structures match README documentation
- ✅ Agent configuration files are handled correctly
- ✅ Hierarchy levels work as described
- ✅ LLM provider configurations are valid
- ✅ Training configuration options work
- ✅ Environment variables are processed correctly

## Test Results Interpretation

### ✅ All Tests Pass
- README instructions are accurate
- All documented functionality works
- Users can follow README successfully

### ⚠️ Some Tests Fail
- Some README claims may be inaccurate
- Users might encounter issues
- Implementation or documentation needs updates

### ❌ Many Tests Fail
- Significant issues with README accuracy
- Major functionality gaps
- Installation or setup problems

## Example Test Output

```
🚀 Starting complete GenHRL README validation...
============================================================

🔍 Running installation verification tests...
✅ Core components imported successfully
✅ Training components imported successfully
✅ Additional components checked successfully
✅ API key found in environment
✅ GenHRL installation verified!
✅ Installation tests passed

🔍 Running workflow functionality tests...
✅ create_task_with_workflow signature verified
✅ TaskManager initialization from README works
✅ Workflow tests passed

🔍 Running configuration and structure tests...
✅ Expected file structure from README verified
✅ Agent configuration files verified
✅ Configuration tests passed

🔍 Validating README claims...
✅ Claim: Core components can be imported
✅ Claim: Workflow functions exist - ['create_task_with_workflow', 'remove_all_previous_skills']
✅ Claim: Training components exist - ['TrainingOrchestrator']
✅ Claim: Utility functions work
✅ Claim: Robot configuration exists - ['G1', 'H1']

📊 README Claims Validation: 5/5 claims validated

============================================================
📋 VALIDATION SUMMARY REPORT
============================================================
INSTALLATION: ✅ PASS
WORKFLOWS: ✅ PASS
CONFIGURATION: ✅ PASS
README CLAIMS: ✅ PASS

OVERALL RESULT: 4/4 test suites passed
🎉 ALL README FUNCTIONALITY VALIDATED SUCCESSFULLY!

✅ Installation instructions work
✅ Programmatic API examples work
✅ Configuration options work
✅ File structure is correct
✅ All README claims are accurate
```

## Test Coverage

The tests cover all major sections of the README:

### ✅ Installation Section
- Package imports
- Installation verification
- Environment setup

### ✅ Quick Start Section  
- Programmatic API examples
- TaskManager usage
- Main workflow functions

### ✅ Programmatic API Reference
- All API classes and functions
- Method signatures
- Configuration options

### ✅ Configuration Examples
- LLM provider configurations
- Verification settings
- Training configurations

### ✅ File Structure Documentation
- Generated directory structure
- Expected file types
- Configuration file formats

### ✅ Troubleshooting Section
- Installation verification
- Environment variable handling
- Error scenarios

## Adding New Tests

When updating the README, add corresponding tests:

1. **Add functionality test** in appropriate test file
2. **Update README claims validation** in `test_runner.py`
3. **Add example validation** in `test_workflows.py`
4. **Update expected file structure** in `test_config_and_structure.py`

## Continuous Integration

These tests should be run:
- ✅ Before merging README changes
- ✅ After installation procedure updates  
- ✅ When adding new API functionality
- ✅ Before releases

## Troubleshooting Tests

### Import Errors
- Check that GenHRL is installed: `pip install -e .`
- Verify Python path includes project root
- Ensure all dependencies are installed

### Path Issues
- Tests use temporary directories automatically
- No need to set up real IsaacLab installation for tests
- Mock objects are used to avoid external dependencies

### API Call Errors
- Tests avoid real API calls by default
- Use mocks and fixtures for testing interfaces
- Set `GENHRL_API_KEY=test_key` if needed

---

**Goal**: Ensure every claim in the README is tested and verified to work correctly after installation.