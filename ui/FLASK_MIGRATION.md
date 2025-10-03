# Flask Migration Guide

This document outlines the migration from Node.js/Express backend to Flask backend for the GenHRL UI.

## Changes Made

### Backend Migration

**Old**: Node.js/Express server (`server/index.js`)
**New**: Flask server (`server/app.py`)

#### Key Changes:
1. **Server Framework**: Migrated from Express.js to Flask
2. **Language**: Backend now runs entirely in Python instead of Node.js
3. **Dependencies**: 
   - Removed: express, cors, fs-extra, child_process, uuid
   - Added: Flask, Flask-CORS (via requirements.txt)
4. **SSE Implementation**: Improved Server-Sent Events using Flask Response streaming with queues
5. **File Operations**: Now uses Python's os, shutil, and pathlib instead of Node.js fs-extra

### Project Structure

```
ui/
├── server/
│   └── app.py              # Flask backend (NEW)
├── client/                 # React frontend (unchanged)
├── requirements.txt        # Python dependencies (NEW)
├── package.json           # Updated for Flask
├── start-ui.sh            # Updated startup script
├── start-ui.bat           # Updated startup script
└── README.md              # Updated documentation
```

### Dependencies

#### Before (Node.js):
```json
{
  "dependencies": {
    "child_process": "^1.0.2",
    "concurrently": "^7.6.0",
    "cors": "^2.8.5",
    "express": "^4.18.2",
    "fs-extra": "^11.1.1",
    "multer": "^1.4.5-lts.1",
    "uuid": "^9.0.1"
  }
}
```

#### After (Flask):
```json
{
  "dependencies": {
    "concurrently": "^7.6.0"
  }
}
```

Plus new `requirements.txt`:
```
Flask==2.3.3
Flask-CORS==4.0.0
```

### API Endpoints (Unchanged)

All API endpoints remain the same:
- `GET /api/tasks` - List all tasks
- `GET /api/tasks/:taskName` - Get task details
- `POST /api/tasks` - Create new task
- `DELETE /api/tasks/:taskName` - Delete task
- `GET /api/robots` - List available robots
- `GET /api/progress/:sessionId` - SSE progress tracking
- `POST /api/test-progress` - Test progress tracking

### Frontend (No Changes)

The React frontend remains completely unchanged. It continues to:
- Use the same API endpoints
- Connect to the same port (5000) via proxy
- Handle the same response formats
- Use the same SSE connection for progress tracking

## Installation & Usage

### Prerequisites

**Before**: Node.js 16+, npm, Python 3.8+, GenHRL, IsaacLab
**After**: Python 3.8+, Node.js 16+ (for frontend only), npm, GenHRL, IsaacLab

### Installation

1. Navigate to the UI directory:
```bash
cd ui
```

2. Install all dependencies:
```bash
npm run install-all
```

This now runs:
- `pip install -r requirements.txt` (Flask backend)
- `cd client && npm install` (React frontend)

3. Start the servers:
```bash
npm start
```

This starts:
- Flask backend on port 5000
- React frontend on port 3000

### Development

#### Frontend Only:
```bash
cd client
npm start
```

#### Backend Only:
```bash
cd server
python3 app.py
```

#### Both Together:
```bash
npm start
```

## Benefits of Flask Migration

1. **Simplified Architecture**: All backend code now runs in Python, eliminating Node.js dependency for the backend
2. **Better Integration**: Direct access to Python GenHRL libraries without subprocess overhead
3. **Reduced Dependencies**: Fewer npm packages to maintain
4. **Improved SSE**: Better Server-Sent Events implementation with proper queuing
5. **Native Python**: All file operations and data processing in native Python
6. **Easier Debugging**: Single-language backend debugging

## Backwards Compatibility

- **API**: All endpoints maintain the same interface
- **Frontend**: React app unchanged, works identically
- **Data Formats**: All request/response formats preserved
- **File Structure**: Generated task files remain in the same locations

## Testing

The migration maintains 100% API compatibility. Test by:

1. Creating a new task
2. Viewing task details
3. Deleting a task
4. Monitoring progress via SSE

All functionality should work identically to the Node.js version.

## Troubleshooting

### Common Issues

1. **Flask not found**: Install with `pip install Flask Flask-CORS`
2. **Python path issues**: Ensure GenHRL is accessible via `python3 -c "import genhrl"`
3. **Port conflicts**: Ensure ports 5000 and 3000 are available
4. **Permission errors**: Check file system permissions for IsaacLab directory

### Logs

Flask development server provides detailed logging. Check console output for:
- Request/response details
- SSE connection status
- Task creation progress
- Error tracebacks

The Flask server runs with `debug=True` in development for detailed error reporting.