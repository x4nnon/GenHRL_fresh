# GenHRL UI

A web-based user interface for generating and managing hierarchical reinforcement learning tasks using the GenHRL framework.

## Features

- **Task Creation**: Generate new hierarchical RL tasks from natural language descriptions
- **Task Management**: Browse, search, and filter existing tasks
- **Hierarchy Visualization**: View skill hierarchies and their relationships
- **Skill Inspection**: Examine individual skills, rewards, and success criteria
- **Object Configuration**: View and manage scene object configurations
- **Training Integration**: Start and monitor training processes (coming soon)

## Setup

### Prerequisites

- Python 3.8+ with GenHRL installed
- Node.js 16+ and npm (for React frontend)
- IsaacLab installation
- API key from Google Gemini or Anthropic Claude

### Installation

1. Navigate to the UI directory:
```bash
cd ui
```

2. Install dependencies:
```bash
npm run install-all
```

This will install:
- Python backend dependencies (Flask, Flask-CORS)
- React frontend dependencies

3. Start the development server:
```bash
npm start
```

This will start:
- Flask backend API server on port 5000
- React frontend on port 3000

### Configuration

The UI expects:
- GenHRL to be installed in the parent directory (`../`)
- IsaacLab to be installed at `../IsaacLab`

If your setup is different, update the paths in `server/app.py`:
```python
ISAACLAB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../IsaacLab'))
GENHRL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
```

## Usage

### Creating Tasks

1. Navigate to "Create Task" in the sidebar
2. Fill in the task details:
   - **Task Name**: Unique identifier (use underscores, no spaces)
   - **Description**: Detailed natural language description
   - **Robot**: Select from available robots (G1, H1, etc.)
   - **Hierarchy Levels**: Choose complexity (1-3 levels)
   - **API Key**: Your LLM API key
3. Click "Create Task" to generate the hierarchical task

### Viewing Tasks

1. Go to "All Tasks" to see all generated tasks
2. Use search and filters to find specific tasks
3. Click on a task to view detailed information including:
   - Task overview and statistics
   - Skill hierarchy visualization
   - Individual skill details
   - Object configurations
   - Skill library data

### Managing Tasks

- **Delete**: Remove tasks and their associated files
- **Export**: Download task configurations (coming soon)
- **Train**: Start training processes (coming soon)

## API Endpoints

The backend provides these API endpoints:

- `GET /api/tasks` - List all tasks
- `GET /api/tasks/:taskName` - Get task details
- `POST /api/tasks` - Create new task
- `DELETE /api/tasks/:taskName` - Delete task
- `GET /api/robots` - List available robots

## Development

### Frontend (React)

```bash
cd client
npm start
```

### Backend (Flask)

```bash
cd server
python3 app.py
```

Alternatively, run both together:
```bash
npm start
```

### Building for Production

```bash
npm run build
```

## Architecture

- **Frontend**: React with Tailwind CSS
- **Backend**: Flask API server (Python)
- **Integration**: Direct Python imports and subprocess calls to GenHRL
- **Visualization**: JSON tree components and code syntax highlighting

## Troubleshooting

### Common Issues

1. **Task creation fails**: Check API key and network connectivity
2. **No tasks showing**: Ensure IsaacLab path is correct
3. **Permission errors**: Check file system permissions for task directories

### Logs

Backend logs are printed to the console when running in development mode.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the GenHRL framework and follows the same licensing terms.