# GenHRL UI Training Functionality

## Overview

The GenHRL UI now includes comprehensive training functionality that mirrors and enhances the command-line interface capabilities. Users can configure, start, monitor, and manage training sessions directly through the web interface.

## Features

### 1. Training Configuration Modal
- **Visual Configuration**: Intuitive form for setting training parameters
- **Real-time Status**: Shows current skill training status (completed/pending)
- **Parameter Validation**: Input validation and helpful tooltips
- **CLI Parity**: All CLI training options available in the UI

### 2. Real-time Training Progress
- **Live Monitoring**: Real-time progress updates via Server-Sent Events (SSE)
- **Skill Sequence**: Visual display of training order and current progress
- **Time Tracking**: Shows elapsed time for overall training and individual skills
- **Status Updates**: Real-time skill status and training stage information

### 3. Training Management
- **Start/Stop Control**: Start training with custom configuration or cancel active sessions
- **Session Management**: Track multiple training sessions across different tasks
- **Error Handling**: Clear error messages and recovery options
- **State Persistence**: Training state preserved across UI refreshes

## API Endpoints

### Backend Training API

#### `GET /api/tasks/{task_name}/training/status`
- **Purpose**: Get current training status for a task
- **Parameters**: `robot` (query parameter)
- **Returns**: Training status, skill progress, and session information

#### `POST /api/tasks/{task_name}/training/start`
- **Purpose**: Start training for a task
- **Parameters**: Training configuration in request body
- **Returns**: Session ID for progress tracking

#### `POST /api/tasks/{task_name}/training/cancel`
- **Purpose**: Cancel active training session
- **Parameters**: `robot` (query parameter)
- **Returns**: Cancellation confirmation

#### `GET /api/tasks/{task_name}/training/progress/{session_id}`
- **Purpose**: Server-Sent Events stream for real-time progress
- **Returns**: Continuous progress updates

## Training Configuration Options

The UI exposes all CLI training parameters:

### Core Parameters
- **Max Time per Skill** (minutes): Maximum training time per skill (default: 180)
- **Min Success States**: Required success states to complete a skill (default: 50)
- **Parallel Environments**: Number of simulation environments (1024-8192, default: 4096)
- **Random Seed**: Seed for reproducible training (default: 42)

### Training Options
- **New Run**: Clean all previous training data before starting
- **Skip Complete**: Skip skills that already have sufficient success states
- **Headless Mode**: Run without visual interface for faster training
- **Record Videos**: Enable video recording during training (with interval/length settings)

### Visual Helpers
- **Status Overview**: Shows total/completed/pending skills before training
- **Time Estimation**: Estimates total training time based on configuration
- **Help Documentation**: Contextual help explaining the training process

## Real-time Progress Monitoring

### Progress Stages
1. **Initializing**: Setting up training orchestrator
2. **Planning**: Determining skill training order
3. **Training**: Active skill training in progress
4. **Complete**: Training finished (success or failure)

### Progress Information
- **Overall Progress**: Percentage complete across all skills
- **Current Skill**: Which skill is currently being trained
- **Training Statistics**: Skills completed, remaining, and current status
- **Training Sequence**: Visual display of all skills with status indicators
- **Time Tracking**: Total elapsed time and per-skill timing

### Visual Indicators
- **Progress Bar**: Visual representation of overall completion
- **Status Icons**: Different icons for each training stage
- **Skill Status**: Color-coded skill cards showing completion status
- **Real-time Updates**: Live updates as training progresses

## Integration Points

### TaskDetail Page
- **Start Training Button**: Replaces placeholder with functional training
- **Training Modal**: Configure and start training for the specific task
- **Progress Monitoring**: Real-time progress tracking during training

### TaskList Page
- **Quick Training**: Start training directly from task cards
- **Bulk Operations**: Train multiple tasks (future enhancement)
- **Status Indicators**: Show training status on task cards

### Dashboard
- **Training Statistics**: Shows active training sessions count
- **Recent Activity**: Displays recent training completions
- **Quick Actions**: Fast access to training functionality

## Backend Implementation

### Training Orchestrator Integration
- **Direct Integration**: Uses GenHRL's `TrainingOrchestrator` class
- **Configuration Mapping**: Maps UI config to `TrainingConfig` objects
- **Process Management**: Manages training processes with proper cleanup
- **Session Tracking**: Tracks active sessions and prevents conflicts

### Progress Streaming
- **Server-Sent Events**: Real-time progress streaming to UI
- **Non-blocking Execution**: Training runs in background threads
- **Resource Management**: Proper cleanup of connections and processes
- **Error Propagation**: Clear error messages from training process

### Session Management
- **Unique Sessions**: Each training session gets a unique ID
- **State Tracking**: Track session state across requests
- **Cleanup**: Automatic cleanup of completed or cancelled sessions
- **Concurrency**: Handle multiple simultaneous training sessions

## CLI Equivalence

The UI training functionality provides complete equivalence to CLI commands:

### CLI Command: `genhrl train task_name`
**UI Equivalent**: Click "Start Training" button with default configuration

### CLI Command: `genhrl train task_name --max-time 120 --min-success-states 75 --new-run`
**UI Equivalent**: Configure training modal with:
- Max Time: 120 minutes
- Min Success States: 75
- New Run: enabled

### CLI Command: `genhrl train task_name --num-envs 8192 --seed 123 --video`
**UI Equivalent**: Configure training modal with:
- Parallel Environments: 8,192
- Random Seed: 123
- Record Videos: enabled

### CLI Command: `genhrl status task_name`
**UI Equivalent**: View task details page or check training status in modals

## Error Handling

### Training Errors
- **Setup Errors**: Clear messages for configuration or setup issues
- **Runtime Errors**: Real-time error reporting during training
- **Network Errors**: Handle connection issues gracefully
- **Resource Errors**: Handle GPU memory or system resource issues

### UI Error States
- **Loading States**: Show loading indicators during operations
- **Error Messages**: Clear, actionable error messages
- **Recovery Options**: Retry buttons and alternative actions
- **Validation Errors**: Input validation with helpful error messages

## Future Enhancements

### Planned Features
- **Training History**: View past training sessions and results
- **Performance Metrics**: Detailed training metrics and analytics
- **Batch Training**: Train multiple tasks simultaneously
- **Training Presets**: Save and reuse training configurations
- **Advanced Monitoring**: GPU usage, memory consumption, training curves

### Integration Opportunities
- **Tensorboard Integration**: Embed training visualizations
- **Model Comparison**: Compare different training runs
- **Hyperparameter Tuning**: Automated parameter optimization
- **Cloud Training**: Support for cloud-based training
- **Distributed Training**: Multi-GPU and multi-node training support

## Usage Examples

### Basic Training Workflow
1. Navigate to a task in TaskDetail or TaskList
2. Click "Start Training" button
3. Configure training parameters in the modal
4. Click "Start Training" to begin
5. Monitor progress in real-time
6. Training completes automatically or can be cancelled

### Advanced Configuration
1. Open training configuration modal
2. Adjust parameters based on your requirements:
   - Increase environments for faster training
   - Enable video recording for analysis
   - Set custom success state thresholds
   - Configure time limits per skill
3. Start training with custom configuration
4. Monitor detailed progress with skill-by-skill breakdown

### Error Recovery
1. If training fails, check error message in progress modal
2. Adjust configuration based on error (e.g., reduce environments for memory issues)
3. Retry training with updated configuration
4. Use "New Run" option to clean state if needed

## Technical Architecture

### Frontend Components
- **TrainingModal.js**: Configuration interface for training parameters
- **TrainingProgressModal.js**: Real-time progress monitoring interface
- **Integration**: Seamless integration into existing TaskDetail and TaskList pages

### Backend Services
- **Training API**: RESTful endpoints for training management
- **Progress Streaming**: Server-Sent Events for real-time updates
- **Session Management**: Track and manage training sessions
- **Process Orchestration**: Manage background training processes

### Communication Flow
1. **Configuration**: UI sends training config to backend API
2. **Session Creation**: Backend creates training session and returns session ID
3. **Process Start**: Backend starts training process in background
4. **Progress Streaming**: Real-time progress via SSE to frontend
5. **Completion**: Training completes and final status sent to UI

This training functionality brings the full power of GenHRL's training orchestration to the web interface, making it accessible to users who prefer visual interfaces while maintaining all the flexibility and control of the command-line interface.