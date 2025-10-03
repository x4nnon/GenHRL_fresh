# 🚀 Automated Port Cleanup System

This system automatically prevents the common "port already in use" issue when starting the development servers.

## ✅ **How It Works**

When you run `npm start`, the system automatically:
1. **Checks for hanging processes** on ports 3000 (React) and 5000 (Flask)
2. **Preserves recently started processes** (< 10 seconds old)
3. **Kills only old zombie processes** cleanly
4. **Starts the dev servers** normally

## 🎯 **Available Commands**

### **From `ui/` directory (Main UI):**
```bash
npm start              # Auto-cleanup + start both servers
npm run dev            # Explicit cleanup + start
npm run clean-start    # Alternative cleanup + start
npm run clean-ports    # Just run cleanup (no start)
```

### **From `ui/client/` directory (Frontend only):**
```bash
npm start              # Start React only (no cleanup - use for full-stack)
npm run start-standalone  # Cleanup + start React only (use for frontend-only dev)
npm run dev            # Explicit cleanup + start
npm run clean-start    # Alternative cleanup + start
npm run clean-ports    # Just run cleanup (no start)
```

### **Manual cleanup (from project root):**
```bash
./kill-dev-ports.sh    # Run cleanup script directly
```

## 🎯 **Use Cases**

### **Full-Stack Development (Recommended):**
```bash
cd ui && npm start     # Cleans ports + starts both Flask backend & React frontend
```

### **Frontend-Only Development:**
```bash
cd ui/client && npm run start-standalone  # Cleans ports + starts React only
```

### **Manual Control:**
```bash
./kill-dev-ports.sh   # Clean ports manually
cd ui && npm start     # Then start normally
```

## 🔧 **Technical Details**

- **Smart cleanup**: Only kills processes older than 10 seconds
- **`prestart` script**: Automatically runs before main UI `npm start`
- **Port cleanup**: Handles ports 3000, 5000 intelligently
- **Process preservation**: Keeps recently started legitimate processes
- **Cross-platform**: Works on Linux/macOS (Windows users can use Git Bash)

## 🐛 **If Issues Persist**

1. **Manual port check:**
   ```bash
   lsof -i:3000    # Check what's using port 3000
   lsof -i:5000    # Check what's using port 5000
   ```

2. **Manual process kill:**
   ```bash
   lsof -ti:3000 | xargs kill -9  # Force kill port 3000
   lsof -ti:5000 | xargs kill -9  # Force kill port 5000
   ```

3. **Clear React cache:**
   ```bash
   cd ui/client
   rm -rf node_modules/.cache
   npm start
   ```

## 💡 **Benefits**

- ✅ **No more hanging npm start**
- ✅ **Consistent development workflow** 
- ✅ **Automatic cleanup on every start**
- ✅ **Works for both frontend and full-stack development**
- ✅ **Multiple command options for different preferences**

**Happy coding! 🎉** 