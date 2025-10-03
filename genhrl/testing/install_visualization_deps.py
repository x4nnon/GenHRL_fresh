#!/usr/bin/env python3
"""
Installation helper for visualization dependencies.
This script helps users install the required dependencies for image generation.
"""

import subprocess
import sys
import platform
from pathlib import Path

def run_command(command, description):
    """Run a shell command and report results."""
    try:
        print(f"🔄 {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_graphviz_installed():
    """Check if graphviz is available."""
    try:
        import graphviz
        print("✅ Python graphviz package is already installed")
        return True
    except ImportError:
        print("❌ Python graphviz package not found")
        return False

def check_system_graphviz():
    """Check if system Graphviz is installed."""
    try:
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ System Graphviz is installed: {result.stderr.strip()}")
            return True
        else:
            print("❌ System Graphviz not found")
            return False
    except FileNotFoundError:
        print("❌ System Graphviz not found")
        return False

def install_python_graphviz():
    """Install the Python graphviz package."""
    return run_command(
        f"{sys.executable} -m pip install graphviz",
        "Installing Python graphviz package"
    )

def install_system_graphviz():
    """Install system Graphviz based on the operating system."""
    system = platform.system().lower()
    
    if system == "linux":
        # Try to detect the distribution
        try:
            with open('/etc/os-release', 'r') as f:
                os_info = f.read().lower()
            
            if 'ubuntu' in os_info or 'debian' in os_info:
                return run_command(
                    "sudo apt-get update && sudo apt-get install -y graphviz",
                    "Installing system Graphviz (Ubuntu/Debian)"
                )
            elif 'fedora' in os_info or 'rhel' in os_info or 'centos' in os_info:
                return run_command(
                    "sudo dnf install -y graphviz || sudo yum install -y graphviz",
                    "Installing system Graphviz (Fedora/RHEL/CentOS)"
                )
            else:
                print("❓ Linux distribution not automatically supported")
                print("   Please install graphviz using your package manager:")
                print("   - Ubuntu/Debian: sudo apt-get install graphviz")
                print("   - Fedora/RHEL: sudo dnf install graphviz")
                print("   - Arch: sudo pacman -S graphviz")
                return False
        except Exception:
            print("❓ Could not detect Linux distribution")
            print("   Please install graphviz using your package manager")
            return False
            
    elif system == "darwin":  # macOS
        # Check if Homebrew is available
        try:
            subprocess.run(['brew', '--version'], capture_output=True, check=True)
            return run_command(
                "brew install graphviz",
                "Installing system Graphviz (macOS via Homebrew)"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❓ Homebrew not found on macOS")
            print("   Please install Homebrew first: https://brew.sh/")
            print("   Then run: brew install graphviz")
            return False
            
    elif system == "windows":
        print("❓ Windows automatic installation not supported")
        print("   Please download and install Graphviz from:")
        print("   https://graphviz.org/download/")
        print("   Make sure to add it to your PATH")
        return False
        
    else:
        print(f"❓ Unsupported operating system: {system}")
        return False

def test_installation():
    """Test if the installation was successful."""
    print("\n🧪 Testing installation...")
    
    if not check_graphviz_installed():
        return False
        
    if not check_system_graphviz():
        return False
    
    # Try to create a simple diagram
    try:
        import graphviz
        dot = graphviz.Digraph()
        dot.node('test', 'Test Node')
        dot.render('/tmp/test_graphviz', format='png', cleanup=True)
        print("✅ Image generation test successful")
        return True
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        return False

def main():
    """Main installation function."""
    print("🔧 GenHRL Visualization Dependencies Installer")
    print("=" * 50)
    
    # Check current status
    print("\n📋 Checking current installation status...")
    python_ok = check_graphviz_installed()
    system_ok = check_system_graphviz()
    
    if python_ok and system_ok:
        print("\n🎉 All dependencies are already installed!")
        if test_installation():
            print("✅ Installation verified - you're ready to generate hierarchy images!")
            return
    
    # Install missing components
    print("\n🔧 Installing missing dependencies...")
    
    if not python_ok:
        if not install_python_graphviz():
            print("❌ Failed to install Python graphviz package")
            print("   Try manually: pip install graphviz")
            return
    
    if not system_ok:
        if not install_system_graphviz():
            print("❌ Failed to install system Graphviz")
            print("   Please install manually for your operating system")
            return
    
    # Final test
    print("\n🧪 Running final verification...")
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        print("✅ You can now generate hierarchy images with:")
        print("   python genhrl/testing/visualize_skill_hierarchies.py")
    else:
        print("\n❌ Installation verification failed")
        print("   Please check the error messages above and try manual installation")

if __name__ == "__main__":
    main()