#!/usr/bin/env python3
"""
Complete advanced deployment script for Transformer Inventory Predictor
Handles TensorFlow installation and advanced model deployment
"""

import os
import sys
import subprocess
import webbrowser
import time
import platform
from pathlib import Path

def get_system_info():
    """Get system information for optimal TensorFlow installation"""
    system = platform.system()
    machine = platform.machine()
    python_version = sys.version_info
    
    print(f"🖥️  System: {system} {machine}")
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if GPU is available (simple check)
    gpu_available = False
    try:
        if system == "Windows":
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
            gpu_available = result.returncode == 0
        elif system in ["Linux", "Darwin"]:
            result = subprocess.run(['which', 'nvidia-smi'], capture_output=True)
            gpu_available = result.returncode == 0
    except:
        pass
    
    print(f"🎮 GPU Available: {'Yes' if gpu_available else 'No (CPU-only recommended)'}")
    return system, gpu_available, python_version

def create_advanced_structure():
    """Create directory structure for advanced model"""
    print("🏗️  Creating advanced project structure...")
    
    directories = [
        'templates',
        'uploads', 
        'models',
        'exported_models',
        'static/css',
        'static/js',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def install_tensorflow_optimized(gpu_available, python_version):
    """Install optimized TensorFlow based on system"""
    print("\n🚀 Installing TensorFlow (Advanced AI Models)...")
    
    # Determine best TensorFlow version
    if python_version >= (3, 12):
        tf_version = "tensorflow>=2.15.0"
        print("⚡ Using TensorFlow 2.15+ for Python 3.12+")
    elif python_version >= (3, 9):
        tf_version = "tensorflow==2.13.0"
        print("⚡ Using TensorFlow 2.13 for Python 3.9+")
    else:
        tf_version = "tensorflow==2.10.0"
        print("⚡ Using TensorFlow 2.10 for older Python")
    
    try:
        # Uninstall existing TensorFlow versions
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "tensorflow", "tensorflow-cpu", "tensorflow-gpu", "-y"], 
                      capture_output=True)
        
        # Install appropriate version
        if gpu_available:
            print("🎮 Installing GPU-enabled TensorFlow...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", tf_version])
        else:
            print("💻 Installing CPU-optimized TensorFlow...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.13.0"])
        
        print("✅ TensorFlow installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ TensorFlow installation failed: {e}")
        print("💡 Trying alternative installation...")
        
        # Alternative installation
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.13.0", "--no-cache-dir"])
            print("✅ Alternative TensorFlow installation successful!")
            return True
        except:
            return False

def install_advanced_dependencies():
    """Install all dependencies for advanced model"""
    print("\n📦 Installing advanced AI dependencies...")
    
    # Core dependencies
    core_deps = [
        "Flask==2.3.3",
        "pandas==2.1.1",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "Werkzeug==2.3.7",
        "gunicorn==21.2.0",
        "h5py==3.9.0"
    ]
    
    for dep in core_deps:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {dep}: {e}")
    
    print("✅ Core dependencies installed!")
    return True

def test_advanced_setup():
    """Test the advanced model setup"""
    print("\n🧪 Testing advanced AI setup...")
    
    tests = []
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} loaded successfully")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        else:
            print("💻 Running on CPU (this is fine for most use cases)")
        
        # Test basic model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        print("✅ Neural network creation test passed")
        tests.append(True)
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        tests.append(False)
    
    # Test other dependencies
    deps_to_test = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('sklearn', 'sklearn'),
        ('flask', 'flask')
    ]
    
    for dep_name, import_name in deps_to_test:
        try:
            __import__(import_name)
            print(f"✅ {dep_name} imported successfully")
            tests.append(True)
        except ImportError as e:
            print(f"❌ {dep_name} import failed: {e}")
            tests.append(False)
    
    # Test Flask app
    try:
        if os.path.exists('app.py'):
            print("✅ Flask app file found")
            tests.append(True)
        else:
            print("⚠️  Flask app file not found")
            tests.append(False)
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        tests.append(False)
    
    success_rate = sum(tests) / len(tests)
    print(f"\n📊 Test Results: {sum(tests)}/{len(tests)} passed ({success_rate*100:.1f}%)")
    
    return success_rate > 0.8

def check_required_files():
    """Check if all required files exist"""
    print("\n📋 Checking required files...")
    
    required_files = {
        'app.py': 'Main Flask application with advanced models',
        'templates/advanced_index.html': 'Advanced web interface template',
        'Requirements.txt': 'Python dependencies'  # Note the capital R
    }
    
    missing_files = []
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ Found: {filename}")
        else:
            print(f"❌ Missing: {filename} ({description})")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files!")
        return False
    
    print("✅ All required files found!")
    return True

def run_advanced_server():
    """Run the advanced Flask server"""
    print("\n🚀 Starting Advanced AI Inventory Predictor...")
    print("🌐 Your advanced ML web app will be available at:")
    print("   • Local: http://localhost:5000")
    print("   • Network: http://[your-ip]:5000")
    print("\n🧠 Features available:")
    print("   ✅ Transformer Models with Multi-Head Attention")
    print("   ✅ LSTM with Attention Mechanism")  
    print("   ✅ CNN-LSTM Hybrid Architecture")
    print("   ✅ Advanced Fine-tuning & Early Stopping")
    print("   ✅ Real-time Training Monitoring")
    print("   ✅ Confidence Intervals & Model Analysis")
    print("   ✅ Feature Importance Visualization")
    print("   ✅ Model Export & Import")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    # Open browser after delay
    def open_browser():
        time.sleep(4)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Set environment variables
        os.environ['FLASK_ENV'] = 'development'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Import and run advanced app
        sys.path.insert(0, '.')
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
        
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        print("💡 Make sure you have created the app.py file with the advanced model code")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def create_sample_data():
    """Create sample CSV data for testing"""
    print("\n📊 Creating sample data for testing...")
    
    sample_data = """Name,Inventory_Cycle,Physical Storage Quantity,Demanded,Delivered,satisfaction_rate (%),Stock_Class,Max_Stock,Family,GroupName
BAGUETTE,1,150.0,120,118,98.3,A,200,BAKERY,Food
BAGUETTE,2,145.0,125,125,100.0,A,200,BAKERY,Food
BAGUETTE,3,140.0,130,128,98.5,A,200,BAKERY,Food
BAGUETTE,4,155.0,115,115,100.0,A,200,BAKERY,Food
BAGUETTE,5,160.0,140,138,98.6,A,200,BAKERY,Food
CROISSANT,1,80.0,70,68,97.1,B,120,BAKERY,Food
CROISSANT,2,85.0,75,75,100.0,B,120,BAKERY,Food
CROISSANT,3,82.0,78,76,97.4,B,120,BAKERY,Food
CROISSANT,4,88.0,65,65,100.0,B,120,BAKERY,Food
CROISSANT,5,90.0,80,79,98.8,B,120,BAKERY,Food
LAPTOP_DELL,1,25.0,20,19,95.0,A,50,ELECTRONICS,Tech
LAPTOP_DELL,2,28.0,22,22,100.0,A,50,ELECTRONICS,Tech
LAPTOP_DELL,3,26.0,24,23,95.8,A,50,ELECTRONICS,Tech
LAPTOP_DELL,4,30.0,18,18,100.0,A,50,ELECTRONICS,Tech
LAPTOP_DELL,5,32.0,26,25,96.2,A,50,ELECTRONICS,Tech
PHONE_SAMSUNG,1,45.0,40,38,95.0,A,80,ELECTRONICS,Tech
PHONE_SAMSUNG,2,48.0,42,42,100.0,A,80,ELECTRONICS,Tech
PHONE_SAMSUNG,3,46.0,44,42,95.5,A,80,ELECTRONICS,Tech
PHONE_SAMSUNG,4,50.0,35,35,100.0,A,80,ELECTRONICS,Tech
PHONE_SAMSUNG,5,52.0,46,45,97.8,A,80,ELECTRONICS,Tech
DETERGENT,1,200.0,180,175,97.2,C,300,CLEANING,Consommable
DETERGENT,2,195.0,185,185,100.0,C,300,CLEANING,Consommable
DETERGENT,3,190.0,190,188,98.9,C,300,CLEANING,Consommable
DETERGENT,4,205.0,170,170,100.0,C,300,CLEANING,Consommable
DETERGENT,5,210.0,200,198,99.0,C,300,CLEANING,Consommable"""

    sample_file = 'sample_inventory_data.csv'
    with open(sample_file, 'w') as f:
        f.write(sample_data)
    
    print(f"✅ Created: {sample_file}")
    print(f"📊 Sample contains: 25 records, 5 products, 3 families")
    print("💡 Use this file to test your advanced AI models!")
    
    return sample_file

def create_deployment_configs():
    """Create deployment configuration files"""
    print("\n📄 Creating deployment configurations...")
    
    # Enhanced Procfile for Heroku
    with open('Procfile', 'w') as f:
        f.write('web: gunicorn app:app --timeout 300 --workers 2\n')
    print("✅ Created: Procfile")
    
    # Runtime for TensorFlow compatibility
    with open('runtime.txt', 'w') as f:
        f.write('python-3.10.12\n')
    print("✅ Created: runtime.txt")
    
    # Enhanced .gitignore
    gitignore_content = """# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# TensorFlow/ML models
models/*.h5
models/*.pb
exported_models/
*.ckpt
*.tflite

# Data files
uploads/*.csv
uploads/*.xlsx
*.db
*.sqlite

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ Created: .gitignore")

def show_deployment_guide():
    """Show comprehensive deployment guide"""
    print("\n" + "="*60)
    print("🌐 ADVANCED AI MODEL DEPLOYMENT GUIDE")
    print("="*60)
    
    print("\n🔥 RECOMMENDED: Heroku (Best for ML models)")
    print("-" * 40)
    print("1. Create account: https://heroku.com")
    print("2. Install Heroku CLI")
    print("3. Commands:")
    print("   heroku login")
    print("   heroku create your-ai-predictor")
    print("   heroku config:set TF_CPP_MIN_LOG_LEVEL=2")
    print("   git init")
    print("   git add .")
    print('   git commit -m "Deploy AI model"')
    print("   git push heroku main")
    
    print("\n⚡ OPTION 2: Railway (Simple)")
    print("-" * 30)
    print("1. Account: https://railway.app")
    print("2. Connect GitHub repository")
    print("3. Auto-deploy!")
    
    print("\n🌊 OPTION 3: Render")
    print("-" * 20)
    print("1. Account: https://render.com")
    print("2. Create Web Service")
    print("3. Build: pip install -r requirements.txt")
    print("4. Start: gunicorn app:app")

def main():
    """Main deployment function"""
    print("🧠 Advanced AI Inventory Predictor Deployment")
    print("=" * 60)
    
    # Get system info
    system, gpu_available, python_version = get_system_info()
    
    print("\nWhat would you like to do?")
    print("1. 🏠 Complete local setup (recommended)")
    print("2. 🌐 Prepare for cloud deployment")
    print("3. 📖 Show deployment guide")
    print("4. 🧪 Test current setup")
    print("5. 📊 Create sample data")
    print("6. 🚀 Start server directly")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        print("\n🏠 COMPLETE LOCAL SETUP")
        print("=" * 30)
        
        # Create structure
        create_advanced_structure()
        
        # Check files
        if not check_required_files():
            print("\n💡 Please ensure all required files exist")
            return
        
        # Install TensorFlow
        if install_tensorflow_optimized(gpu_available, python_version):
            if install_advanced_dependencies():
                if test_advanced_setup():
                    print("\n✅ Advanced AI setup complete!")
                    
                    # Offer to create sample data
                    create_sample = input("\n📊 Create sample data for testing? (y/n): ").lower()
                    if create_sample in ['y', 'yes']:
                        create_sample_data()
                    
                    # Offer to start server
                    start_server = input("\n🚀 Start the server now? (y/n): ").lower()
                    if start_server in ['y', 'yes']:
                        run_advanced_server()
                else:
                    print("\n❌ Setup test failed.")
            else:
                print("\n❌ Dependency installation failed.")
        else:
            print("\n❌ TensorFlow installation failed.")
    
    elif choice == "2":
        print("\n🌐 CLOUD DEPLOYMENT PREPARATION")
        print("=" * 40)
        
        create_advanced_structure()
        create_deployment_configs()
        
        if check_required_files():
            print("\n✅ Cloud deployment ready!")
            show_deployment_guide()
        else:
            print("\n❌ Missing required files")
    
    elif choice == "3":
        show_deployment_guide()
    
    elif choice == "4":
        print("\n🧪 TESTING SETUP")
        print("=" * 20)
        
        if check_required_files():
            if test_advanced_setup():
                print("\n✅ Setup looks good!")
            else:
                print("\n❌ Setup has issues")
        else:
            print("\n❌ Missing files")
    
    elif choice == "5":
        print("\n📊 CREATING SAMPLE DATA")
        print("=" * 25)
        create_sample_data()
    
    elif choice == "6":
        print("\n🚀 STARTING SERVER")
        print("=" * 20)
        if check_required_files():
            run_advanced_server()
        else:
            print("❌ Cannot start - missing files")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()