#!/bin/bash

# Script to test episode reward logging with and without success state loading

TASK_NAME=${1:-"navigate_test"}
ROBOT=${2:-"G1"}

echo "Testing episode reward logging for task: $TASK_NAME"
echo "Robot: $ROBOT"
echo ""

# Test 1: With success state loading disabled (should work)
echo "=== TEST 1: Success state loading DISABLED ==="
echo "This should show proper episode rewards in wandb"
echo "Running: GENHRL_DISABLE_SUCCESS_LOADING=1 genhrl train $TASK_NAME --robot $ROBOT --simple --steps 1000"
echo ""
read -p "Press Enter to start Test 1..."

GENHRL_DISABLE_SUCCESS_LOADING=1 genhrl train $TASK_NAME --robot $ROBOT --simple --steps 1000

echo ""
echo "Test 1 completed. Check wandb for episode reward logging."
echo ""
read -p "Press Enter to continue to Test 2..."

# Test 2: With success state loading enabled and selective position loading
echo "=== TEST 2: Success state loading ENABLED with selective position loading ==="
echo "This uses the new position-only loading that should preserve episode reward tracking"
echo "Running: genhrl train $TASK_NAME --robot $ROBOT --simple --steps 1000"
echo ""
read -p "Press Enter to start Test 2..."

genhrl train $TASK_NAME --robot $ROBOT --simple --steps 1000

echo ""
echo "Test 2 completed. Check wandb and compare episode reward logging."
echo ""

echo ""
echo "Analysis:"
echo "- Test 1 (no success loading): Should show proper episode rewards"
echo "- Test 2 (with success loading): Should ALSO show proper episode rewards"  
echo "- If both tests work, the fix is successful!"