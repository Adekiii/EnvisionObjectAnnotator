#!/bin/bash
# EnvisionObjectAnnotator Setup Script (Mac/Linux)
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo ""
echo "========================================"
echo "  EnvisionObjectAnnotator Setup"
echo "========================================"
echo ""

# Check prerequisites
echo "[1/5] Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install Python 3.10+"
    exit 1
fi
echo "  Found: $(python3 --version)"

if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Install Node.js 18+"
    exit 1
fi
echo "  Found: Node.js $(node --version)"

if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: ffmpeg not found. Video processing may fail."
    echo "  Install: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)"
else
    echo "  Found: ffmpeg"
fi

if ! command -v git &> /dev/null; then
    echo "ERROR: Git not found."
    exit 1
fi
echo "  Found: git"

# Create backend venv and install dependencies
echo ""
echo "[2/5] Setting up backend (Python venv + dependencies)..."
cd backend

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created virtual environment"
else
    echo "  Virtual environment already exists"
fi

PIP="./.venv/bin/pip"
PYTHON="./.venv/bin/python"

echo "  Installing backend requirements..."
$PIP install --upgrade pip -q
$PIP install -r requirements.txt -q

# Install PyTorch
echo "  Installing PyTorch..."
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS - default build (includes Apple GPU / MPS support)
    $PIP install torch torchvision torchaudio -q
else
    # Linux - try CUDA first, fall back to CPU-only build
    $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
    $PIP install torch torchvision torchaudio -q
fi

# Verify GPU availability
CUDA_CHECK=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>&1)
if [[ "$CUDA_CHECK" == "True" ]]; then
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1)
    echo "  GPU detected: $GPU_NAME"
elif [[ "$(uname)" == "Darwin" ]]; then
    MPS_CHECK=$($PYTHON -c "import torch; print(torch.backends.mps.is_available())" 2>&1)
    if [[ "$MPS_CHECK" == "True" ]]; then
        echo "  Apple GPU (MPS) detected"
    else
        echo "  WARNING: No GPU detected! Processing will be very slow."
    fi
else
    echo ""
    echo "  WARNING: No GPU detected! Processing will be very slow."
    echo "  To enable GPU acceleration:"
    echo "    1. Install NVIDIA drivers: https://www.nvidia.com/drivers"
    echo "    2. Install CUDA 12.1+: https://developer.nvidia.com/cuda-downloads"
    echo "    3. Re-run this setup script"
    echo ""
fi

echo "  Backend Python setup complete"

# Vendor + install MobileSAM and Cutie
echo ""
echo "[3/5] Installing MobileSAM (click -> mask) and Cutie (mask -> track)..."
mkdir -p vendor

if [ ! -d "vendor/MobileSAM" ]; then
    echo "  Cloning MobileSAM..."
    git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM.git vendor/MobileSAM -q
else
    echo "  vendor/MobileSAM already exists"
fi
$PIP install -e vendor/MobileSAM --no-deps -q

if [ ! -d "vendor/Cutie" ]; then
    echo "  Cloning Cutie..."
    git clone --depth 1 https://github.com/hkchengrex/Cutie.git vendor/Cutie -q
else
    echo "  vendor/Cutie already exists"
fi
$PIP install -e vendor/Cutie --no-deps -q

echo "  MobileSAM and Cutie installed"

# Checkpoints
echo ""
echo "[4/5] Setting up model checkpoints..."
mkdir -p checkpoints

if [ ! -f "checkpoints/mobile_sam.pt" ]; then
    if [ -f "vendor/MobileSAM/weights/mobile_sam.pt" ]; then
        cp vendor/MobileSAM/weights/mobile_sam.pt checkpoints/mobile_sam.pt
        echo "  mobile_sam.pt copied from the vendored MobileSAM repo"
    else
        echo "  WARNING: vendor/MobileSAM/weights/mobile_sam.pt not found -- MobileSAM checkpoint is missing."
    fi
else
    echo "  mobile_sam.pt already exists"
fi
echo "  (Cutie's own checkpoint, ~140MB, downloads automatically the first time you process a video)"

cd ..

# Install frontend
echo ""
echo "[5/5] Setting up frontend (Node.js)..."
cd frontend
npm install --silent 2>/dev/null
cd ..
echo "  Frontend setup complete"

# Done
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  Terminal 1: cd backend && ./.venv/bin/uvicorn app.main:app --reload"
echo "  Terminal 2: cd frontend && npm run dev"
echo ""
echo "Then open: http://localhost:5173"
echo ""
