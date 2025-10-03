#!/bin/bash

# GenHRL UI Startup Script
echo "🚀 Starting GenHRL UI..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip and try again."
    exit 1
fi

# Check if Node.js is installed (for frontend)
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check if npm is installed (for frontend)
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python backend dependencies..."
pip install -r requirements.txt

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "client/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd client && npm install && cd ..
fi

# Check if Python and GenHRL are available
echo "🔍 Checking GenHRL installation..."
if ! python3 -c "import genhrl.generation" 2>/dev/null; then
    echo "⚠️  Warning: GenHRL not found. Make sure it's installed and accessible."
    echo "   The UI will still start but task creation will not work."
fi

# Check if Flask is available
echo "🔍 Checking Flask installation..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found. Installing Flask..."
    pip install Flask Flask-CORS
fi

# Check if IsaacLab exists
ISAACLAB_PATH="../IsaacLab"
if [ ! -d "$ISAACLAB_PATH" ]; then
    echo "⚠️  Warning: IsaacLab not found at $ISAACLAB_PATH"
    echo "   You may need to update the path in server/app.py"
fi

echo "✅ Starting GenHRL UI..."
echo "   - Backend API (Flask): http://localhost:5000"
echo "   - Frontend UI (React): http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the servers"

# Set environment variables to fix webpack dev server issues
export DANGEROUSLY_DISABLE_HOST_CHECK=true

# Start both servers
npm start