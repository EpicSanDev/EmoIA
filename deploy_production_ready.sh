#!/bin/bash

# EmoIA v3.0 Production Ready Deployment Script
# Comprehensive setup for RTX 2070 Super optimized AI assistant

set -e  # Exit on any error

echo "🚀 EmoIA v3.0 - Production Ready Deployment"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}=== $1 ===${NC}"
}

# Check system requirements
check_system_requirements() {
    print_header "SYSTEM REQUIREMENTS CHECK"
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        if echo "$GPU_INFO" | grep -q "2070"; then
            print_success "RTX 2070 Super detected: $GPU_INFO"
        else
            print_warning "RTX 2070 Super not detected. Current GPU: $GPU_INFO"
            print_warning "Performance optimizations may not be optimal"
        fi
    else
        print_error "NVIDIA GPU not detected or nvidia-smi not available"
        print_warning "Falling back to CPU mode"
    fi
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -ge 60 ]; then
        print_success "Sufficient RAM detected: ${TOTAL_RAM}GB"
    else
        print_warning "Recommended 64GB RAM. Current: ${TOTAL_RAM}GB"
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python detected: $PYTHON_VERSION"
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js detected: $NODE_VERSION"
    else
        print_error "Node.js not found. Please install Node.js 18+"
        exit 1
    fi
    
    echo ""
}

# Setup Python environment
setup_python_environment() {
    print_header "PYTHON ENVIRONMENT SETUP"
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    
    # Core dependencies
    pip install fastapi uvicorn[standard] websockets
    pip install pydantic sqlalchemy psycopg2-binary
    pip install python-multipart aiofiles
    
    # AI/ML dependencies
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers sentence-transformers
    pip install numpy pandas scikit-learn
    
    # GPU monitoring (install if possible, skip if not available)
    pip install GPUtil psutil || print_warning "GPUtil not available - GPU monitoring disabled"
    
    # Additional dependencies
    pip install pyyaml python-jose[cryptography] passlib[bcrypt]
    pip install pytest pytest-asyncio httpx
    
    print_success "Python dependencies installed"
    echo ""
}

# Setup Node.js environment
setup_node_environment() {
    print_header "FRONTEND ENVIRONMENT SETUP"
    
    cd frontend
    
    # Check if package.json exists
    if [ ! -f "package.json" ]; then
        print_error "package.json not found in frontend directory"
        print_status "Creating React TypeScript project..."
        npx create-react-app . --template typescript --use-npm
    fi
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    # Install additional dependencies for EmoIA
    npm install react-i18next i18next
    npm install chart.js react-chartjs-2
    npm install @types/react @types/react-dom
    
    print_success "Frontend dependencies installed"
    cd ..
    echo ""
}

# Setup configuration
setup_configuration() {
    print_header "CONFIGURATION SETUP"
    
    # Create logs directory
    mkdir -p logs
    
    # Create cache directory
    mkdir -p cache
    
    # Create models directory
    mkdir -p models
    
    # Set up environment variables
    if [ ! -f ".env" ]; then
        print_status "Creating environment configuration..."
        cat > .env << EOL
# EmoIA v3.0 Environment Configuration
ENVIRONMENT=production
DEBUG=false

# Database
DATABASE_URL=sqlite:///./emoia.db

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET=your-jwt-secret-change-in-production

# GPU Optimization
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_LIMIT=7.5

# Cache
REDIS_URL=redis://localhost:6379/0

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
EOL
        print_success "Environment configuration created"
        print_warning "Please update .env file with secure keys for production"
    else
        print_status "Environment configuration already exists"
    fi
    
    echo ""
}

# Setup database
setup_database() {
    print_header "DATABASE SETUP"
    
    print_status "Setting up SQLite database..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Create database tables (this would typically be done with migrations)
    python3 -c "
import sqlite3
import os

db_path = 'emoia.db'
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create basic tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            preferences TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            message TEXT,
            response TEXT,
            emotion_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            description TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print('Database initialized successfully')
else:
    print('Database already exists')
"
    
    print_success "Database setup completed"
    echo ""
}

# Build frontend
build_frontend() {
    print_header "FRONTEND BUILD"
    
    cd frontend
    
    print_status "Building React frontend for production..."
    npm run build
    
    print_success "Frontend build completed"
    cd ..
    echo ""
}

# Setup monitoring
setup_monitoring() {
    print_header "MONITORING SETUP"
    
    # Create monitoring directory
    mkdir -p monitoring
    
    # Create Prometheus config
    cat > monitoring/prometheus.yml << EOL
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'emoia-api'
    static_configs:
      - targets: ['localhost:8000']
        
  - job_name: 'system-metrics'
    static_configs:
      - targets: ['localhost:9100']
EOL
    
    print_success "Monitoring configuration created"
    echo ""
}

# Create startup scripts
create_startup_scripts() {
    print_header "STARTUP SCRIPTS CREATION"
    
    # Create backend startup script
    cat > start_backend.sh << 'EOL'
#!/bin/bash
echo "🚀 Starting EmoIA Backend..."
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
uvicorn src.core.api:app --host 0.0.0.0 --port 8000 --workers 4
EOL
    chmod +x start_backend.sh
    
    # Create frontend startup script
    cat > start_frontend.sh << 'EOL'
#!/bin/bash
echo "🚀 Starting EmoIA Frontend..."
cd frontend
npm start
EOL
    chmod +x start_frontend.sh
    
    # Create complete startup script
    cat > start_emoia.sh << 'EOL'
#!/bin/bash
echo "🚀 Starting Complete EmoIA System..."

# Start backend in background
./start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
./start_frontend.sh &
FRONTEND_PID=$!

echo "✅ EmoIA System Started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap 'kill $BACKEND_PID $FRONTEND_PID; exit' INT
wait
EOL
    chmod +x start_emoia.sh
    
    print_success "Startup scripts created"
    echo ""
}

# GPU optimization setup
setup_gpu_optimization() {
    print_header "GPU OPTIMIZATION SETUP"
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "Configuring RTX 2070 Super optimizations..."
        
        # Set GPU performance mode
        sudo nvidia-smi -pm 1 2>/dev/null || print_warning "Could not enable persistence mode (requires sudo)"
        
        # Set maximum performance
        sudo nvidia-smi -ac 3500,1770 2>/dev/null || print_warning "Could not set memory/graphics clocks (requires sudo)"
        
        print_success "GPU optimization configured"
    else
        print_warning "NVIDIA GPU not available - skipping GPU optimization"
    fi
    
    echo ""
}

# Final system check
final_system_check() {
    print_header "FINAL SYSTEM CHECK"
    
    # Check if all key files exist
    local files_to_check=(
        "config.yaml"
        "frontend/src/themes.css"
        "frontend/src/components/TaskManager.tsx"
        "frontend/src/components/SmartCalendar.tsx"
        "src/core/advanced_api.py"
        "src/gpu_optimization/rtx_optimizer.py"
        "EMOIA_PRODUCTION_ROADMAP.md"
        "PRODUCTION_COMPLETION_SUMMARY.md"
    )
    
    local missing_files=()
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            print_success "✓ $file"
        else
            print_error "✗ $file (missing)"
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        print_success "All core files present"
    else
        print_warning "Some files are missing - system may not function optimally"
    fi
    
    echo ""
}

# Main deployment function
main() {
    echo "🎯 Starting EmoIA v3.0 Production Deployment..."
    echo ""
    
    check_system_requirements
    setup_python_environment
    setup_node_environment
    setup_configuration
    setup_database
    build_frontend
    setup_monitoring
    create_startup_scripts
    setup_gpu_optimization
    final_system_check
    
    print_header "DEPLOYMENT COMPLETE"
    print_success "🎉 EmoIA v3.0 is ready for production!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Review and update .env file with production keys"
    echo "2. Run: ./start_emoia.sh to start the complete system"
    echo "3. Open http://localhost:3000 to access the application"
    echo "4. Check http://localhost:8000/docs for API documentation"
    echo ""
    echo "🔧 Advanced:"
    echo "- Monitor GPU performance: nvidia-smi"
    echo "- View logs: tail -f logs/*.log"
    echo "- GPU optimization report: Check logs for RTX optimizer output"
    echo ""
    print_success "🚀 EmoIA v3.0 - Production Ready!"
}

# Run main function
main "$@"