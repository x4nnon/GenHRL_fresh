@echo off
echo 🚀 Starting GenHRL UI...

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where python3 >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Python 3 is not installed. Please install Python 3.8+ and try again.
        pause
        exit /b 1
    )
)

REM Check if pip is installed
where pip >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ pip is not installed. Please install pip and try again.
    pause
    exit /b 1
)

REM Check if Node.js is installed (for frontend)
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Node.js is not installed. Please install Node.js 16+ and try again.
    pause
    exit /b 1
)

REM Check if npm is installed (for frontend)
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ npm is not installed. Please install npm and try again.
    pause
    exit /b 1
)

REM Install Python dependencies
echo 📦 Installing Python backend dependencies...
pip install -r requirements.txt

REM Install frontend dependencies if node_modules doesn't exist
if not exist "client\node_modules" (
    echo 📦 Installing frontend dependencies...
    cd client
    npm install
    cd ..
)

REM Check if Flask is available
echo 🔍 Checking Flask installation...
python -c "import flask" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Flask not found. Installing Flask...
    pip install Flask Flask-CORS
)

REM Check if IsaacLab exists
if not exist "..\IsaacLab" (
    echo ⚠️  Warning: IsaacLab not found at ..\IsaacLab
    echo    You may need to update the path in server\app.py
)

echo ✅ Starting GenHRL UI...
echo    - Backend API (Flask): http://localhost:5000
echo    - Frontend UI (React): http://localhost:3000
echo.
echo Press Ctrl+C to stop the servers

REM Set environment variables to fix webpack dev server issues
set DANGEROUSLY_DISABLE_HOST_CHECK=true

REM Start both servers
npm start