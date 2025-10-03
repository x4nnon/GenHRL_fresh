# Terminal Integration for Training

This document explains the new terminal integration feature that allows you to see real-time training output directly in the UI.

## Features

### Real-time Terminal Output
- **Live Streaming**: See actual training logs as they happen
- **Color Coding**: Different colors for errors (red), warnings (yellow), success (green), and info (blue)
- **Auto-scroll**: Terminal automatically scrolls to show latest output
- **Timestamp**: Each line shows when it was logged

### Enhanced Training Modal
The training progress modal now has two tabs:

1. **Progress Tab**: High-level training overview with status cards and progress information
2. **Terminal Output Tab**: Real-time terminal logs with full training details

### Error Detection
- **Auto-switch**: When errors occur, the UI automatically switches to the Terminal Output tab
- **Error Badge**: The Terminal tab shows an "Errors" badge when errors are detected
- **Failure Handling**: When training fails, users are directed to check terminal output for details

### Terminal Controls
- **Copy**: Copy all terminal output to clipboard
- **Download**: Save terminal logs as a text file
- **Expand/Collapse**: Toggle between normal and expanded terminal view
- **Clear**: Clear the terminal output (available during progress)

## How to Use

### Starting Training
1. Go to any task detail page
2. Click "Start Training" button
3. Configure training parameters in the modal
4. Click "Start Training" to begin

### Monitoring Training
1. The Training Progress modal opens automatically
2. **Progress Tab**: Shows overall status, elapsed time, and skill progress
3. **Terminal Output Tab**: Shows real-time training logs
4. Switch between tabs as needed

### When Training Fails
1. The UI automatically switches to the Terminal Output tab
2. Look for red error messages or yellow warnings
3. Scroll through the logs to find the specific failure point
4. Use Copy or Download to save logs for debugging

### Terminal Output Colors
- **Red**: Errors, exceptions, failures
- **Yellow**: Warnings, failed attempts
- **Green**: Success messages, completions
- **Blue**: Info messages, stage transitions
- **White**: General output

## Technical Details

### Backend Changes
- Modified `start_training_async()` to capture real stdout/stderr
- Added `send_training_terminal_output()` function
- Streams terminal output via Server-Sent Events (SSE)
- Enhanced error handling and reporting

### Frontend Changes
- Updated `TrainingProgressModal` with tabbed interface
- Added terminal output handling and display
- Implemented auto-scrolling and color coding
- Added copy/download functionality

### Data Flow
1. Training process writes to stdout/stderr
2. Backend captures output line by line
3. Each line is sent to UI via SSE with timestamp
4. Frontend displays in terminal with appropriate styling
5. Error detection triggers tab switching

## Example Terminal Output

```
[14:23:45] === TRAINING INITIALIZATION ===
[14:23:45] Task: Pick_Up_Ball
[14:23:45] Robot: G1
[14:23:45] IsaacLab Path: /path/to/IsaacLab
[14:23:46] === STARTING TRAINING SEQUENCE ===
[14:23:46] Starting sequential training...
[14:23:47] --- Processing PRIMITIVE (1/3): GraspBall ---
[14:23:47] Running command: ./isaaclab.sh -p scripts/...
[14:24:15] Training completed for GraspBall at step 45000
[14:24:15] --- Completed GraspBall ---
```

## Troubleshooting

### No Terminal Output
- Check if training actually started
- Verify backend connection (look for SSE connection messages)
- Refresh the browser if connection is lost

### Terminal Not Auto-scrolling
- Click in the terminal area to focus it
- Check if manually scrolled up (auto-scroll only works at bottom)

### Missing Error Details
- Look for lines containing "ERROR", "Exception", "Traceback"
- Check both red and yellow colored lines
- Use Ctrl+F to search for specific error terms

### Performance
- Terminal keeps last 1000 lines to prevent memory issues
- Use Download button to save full logs if needed
- Expand terminal for better viewing of long error messages

## Future Enhancements

Potential improvements for future versions:
- Search functionality within terminal output
- Filtering by log level (errors only, etc.)
- Real-time training metrics visualization
- Integration with IsaacLab's native logging
- Persistent log storage and history 