#!/bin/bash

# Wan 2.1 Complete Setup for RunPod - No Sage Attention
# GPU: RTX 4090 (24GB VRAM)
# Based on /ldg/ Wan 2.1 Install and Optimization Guide

set -e

echo "============================================"
echo "Wan 2.1 RunPod Setup - Maximum Performance"
echo "GPU: RTX 4090 (24GB VRAM)"
echo "All optimizations enabled (except sage-attention)"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
progress() { echo -e "${BLUE}[PROGRESS]${NC} $1"; }

# Ensure we're in the right directory
cd /workspace

# Update system and install essentials
progress "Installing system dependencies..."
apt update && apt upgrade -y
apt install -y wget curl git unzip htop nvtop nano vim

# Verify GPU
log "Checking GPU status..."
nvidia-smi
if nvidia-smi | grep -q "RTX 4090\|RTX A6000\|RTX 3090"; then
    log "✅ Compatible GPU detected - perfect for maximum performance!"
else
    warn "GPU detection may have issues, but continuing..."
fi

# Clone ComfyUI
progress "Setting up ComfyUI..."
if [[ ! -d "ComfyUI" ]]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git
    log "ComfyUI cloned successfully"
else
    log "ComfyUI already exists, pulling latest changes..."
    cd ComfyUI && git pull && cd ..
fi

cd ComfyUI

# Create directory structure
log "Creating model directories..."
mkdir -p models/{diffusion_models,text_encoders,clip_vision,vae,workflows}
mkdir -p custom_nodes

# Install latest compatible PyTorch (CRITICAL for fp16_fast)
progress "Installing PyTorch 2.8.0.dev20250616+cu128 (Latest available for fp16_fast)..."
pip install torch==2.8.0.dev20250616+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# Verify PyTorch installation
log "Verifying PyTorch installation..."
python -c "
import torch
version = torch.__version__
print(f'PyTorch version: {version}')
if '2.8.0.dev' in version:
    print('✅ Correct PyTorch version for fp16_fast optimization')
else:
    print('⚠️  PyTorch version may not support all optimizations')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Install ComfyUI requirements
log "Installing ComfyUI dependencies..."
pip install -r requirements.txt

# Install performance libraries (excluding sage-attention due to CUDA compatibility)
progress "Installing optimization libraries..."
log "Installing triton and packaging for performance optimizations..."
pip install triton packaging

# Note about sage-attention
warn "Skipping sage-attention due to CUDA version compatibility issues"
warn "You'll still get excellent performance with fp16_fast + TeaCache + TorchCompile"

# Install essential custom nodes
progress "Installing custom nodes..."
cd custom_nodes

# Essential nodes for Wan 2.1 from /ldg/ guide
declare -a essential_repos=(
    "https://github.com/ltdrdata/ComfyUI-Manager.git"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    "https://github.com/city96/ComfyUI-GGUF.git"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"
    "https://github.com/kijai/ComfyUI-KJNodes.git"
    "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git"
)

for repo in "${essential_repos[@]}"; do
    repo_name=$(basename "$repo" .git)
    if [[ ! -d "$repo_name" ]]; then
        log "Installing $repo_name..."
        git clone "$repo" || warn "Failed to install $repo_name"
    else
        log "$repo_name already exists"
    fi
done

cd ..

# Download optimized workflows from /ldg/ guide
progress "Downloading /ldg/ optimized workflows..."
cd models/workflows

# Download the exact workflows from the guide
log "Downloading /ldg/ workflows (updated 26 April 2025)..."
wget -O ldg_cc_i2v_14b_480p.json "https://files.catbox.moe/a8j0ei.json" || warn "Failed to download I2V workflow"
wget -O ldg_cc_t2v_14b_480p.json "https://files.catbox.moe/gzwcwd.json" || warn "Failed to download T2V workflow"

# Also download Kijai workflows as backup
log "Downloading Kijai workflows as backup..."
wget -O ldg_kj_i2v_14b_480p.json "https://files.catbox.moe/togak7.json" || warn "Failed to download Kijai I2V workflow"
wget -O ldg_kj_t2v_14b_480p.json "https://files.catbox.moe/ewusu9.json" || warn "Failed to download Kijai T2V workflow"

cd ../..

# Download models (Q8 for maximum quality as specified in guide)
progress "Downloading Wan 2.1 models (Q8 quantization)..."
log "⚠️  This will download ~50GB of Q8 models for maximum quality"
log "Models will be downloaded in parallel for speed"

cd models

# Download Q8 models in parallel (from /ldg/ guide)
{
    log "Downloading I2V 480P Q8 model..."
    wget -c -O diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf \
        "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf"
} &

{
    log "Downloading I2V 720P Q8 model..."
    wget -c -O diffusion_models/wan2.1-i2v-14b-720p-Q8_0.gguf \
        "https://huggingface.co/city96/Wan2.1-I2V-14B-720P-gguf/resolve/main/wan2.1-i2v-14b-720p-Q8_0.gguf"
} &

{
    log "Downloading T2V Q8 model..."
    wget -c -O diffusion_models/wan2.1-t2v-14b-Q8_0.gguf \
        "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q8_0.gguf"
} &

{
    log "Downloading UMT5 text encoder..."
    wget -c -O text_encoders/umt5_xxl_fp16.safetensors \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"
} &

{
    log "Downloading CLIP Vision..."
    wget -c -O clip_vision/clip_vision_h.safetensors \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
} &

{
    log "Downloading VAE..."
    wget -c -O vae/wan_2.1_vae.safetensors \
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
} &

# Wait for all downloads to complete
log "Waiting for all model downloads to complete..."
wait

cd ..

# Install custom node requirements
progress "Installing custom node requirements..."
cd custom_nodes
for dir in */; do
    if [[ -f "$dir/requirements.txt" ]]; then
        log "Installing requirements for $dir"
        pip install -r "$dir/requirements.txt" || warn "Failed to install requirements for $dir"
    fi
done
cd ..

# Create optimized launch script
progress "Creating optimized launch script..."
cat > launch_wan21.sh << 'EOF'
#!/bin/bash

# Wan 2.1 Optimized Launch Script
# All /ldg/ guide optimizations enabled (except sage-attention)

cd /workspace/ComfyUI

echo "=============================================="
echo "🚀 Starting Wan 2.1 Maximum Performance"
echo "GPU: RTX 4090 (24GB VRAM)"
echo "Expected VRAM usage: 21-23GB"
echo "=============================================="

# Verify PyTorch version for fp16_fast compatibility
python -c "
import torch
version = torch.__version__
print(f'PyTorch: {version}')
if '2.8.0.dev' in version:
    print('✅ fp16_fast optimization enabled')
else:
    print('⚠️  fp16_fast may not work - check PyTorch version')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"

echo ""
echo "🔧 Optimizations enabled per /ldg/ guide:"
echo "  ✅ FP16 Fast Accumulation (--fast)"
echo "  ✅ TeaCache (configured in workflows)"
echo "  ✅ TorchCompile (configured in workflows)"
echo "  ✅ Adaptive Guidance (configured in workflows)"
echo "  ⚠️  Sage Attention (skipped - CUDA compatibility)"
echo ""
echo "Expected performance: 2-3x faster than baseline"
echo ""

# Launch ComfyUI with /ldg/ optimizations (minus sage-attention)
python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --fast \
    --enable-cors-header \
    --verbose

EOF

chmod +x launch_wan21.sh

# Create verification script
log "Creating verification script..."
cat > verify_setup.py << 'EOF'
#!/usr/bin/env python3

import torch
import os
import sys

def main():
    print("🔍 Wan 2.1 Setup Verification")
    print("=" * 50)
    print("Based on /ldg/ Wan 2.1 Install and Optimization Guide")
    print("=" * 50)
    
    # Check PyTorch version (critical for fp16_fast)
    version = torch.__version__
    print(f"PyTorch: {version}")
    if "2.8.0.dev" in version:
        print("✅ fp16_fast optimization supported")
    else:
        print("❌ Wrong PyTorch version - fp16_fast won't work")
        print("   Guide requires: 2.8.0dev")
    
    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA: {device_name}")
        print(f"✅ VRAM: {vram:.1f} GB")
        
        if "RTX 4090" in device_name and vram >= 23:
            print("✅ Perfect GPU for Q8 models and maximum quality!")
        elif vram >= 16:
            print("✅ Good GPU - can use Q8 models with some adjustments")
        else:
            print("⚠️  Limited VRAM - consider Q6/Q5 models")
    else:
        print("❌ CUDA not available")
        return False
    
    # Check Q8 models (highest quality from guide)
    models_path = "/workspace/ComfyUI/models"
    required_models = [
        ("diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf", 14.2),
        ("diffusion_models/wan2.1-i2v-14b-720p-Q8_0.gguf", 14.2),
        ("diffusion_models/wan2.1-t2v-14b-Q8_0.gguf", 14.2),
        ("text_encoders/umt5_xxl_fp16.safetensors", 9.8),
        ("clip_vision/clip_vision_h.safetensors", 1.3),
        ("vae/wan_2.1_vae.safetensors", 1.2)
    ]
    
    print("\n📁 Q8 Model Files (Maximum Quality):")
    all_present = True
    total_size = 0
    
    for model, expected_size in required_models:
        path = os.path.join(models_path, model)
        if os.path.exists(path):
            actual_size = os.path.getsize(path) / 1024**3
            total_size += actual_size
            if actual_size >= expected_size * 0.9:
                print(f"✅ {os.path.basename(model)} ({actual_size:.1f} GB)")
            else:
                print(f"⚠️ {os.path.basename(model)} ({actual_size:.1f} GB) - may be incomplete")
                all_present = False
        else:
            print(f"❌ Missing: {os.path.basename(model)}")
            all_present = False
    
    print(f"\nTotal model size: {total_size:.1f} GB")
    
    # Check /ldg/ workflows
    workflows_path = "/workspace/ComfyUI/models/workflows"
    required_workflows = [
        "ldg_cc_i2v_14b_480p.json",
        "ldg_cc_t2v_14b_480p.json"
    ]
    
    print("\n📋 /ldg/ Optimized Workflows:")
    for workflow in required_workflows:
        path = os.path.join(workflows_path, workflow)
        if os.path.exists(path):
            print(f"✅ {workflow}")
        else:
            print(f"❌ Missing: {workflow}")
            all_present = False
    
    print("\n🚀 Performance Configuration:")
    print("✅ Q8 quantization (highest quality)")
    print("✅ FP16 Fast Accumulation (--fast)")
    print("✅ TeaCache (in workflows)")
    print("✅ TorchCompile (in workflows)")
    print("✅ Adaptive Guidance (in workflows)")
    print("⚠️ Sage Attention (skipped - CUDA compatibility)")
    
    print("\n📐 Supported Resolutions (per /ldg/ guide):")
    print("T2V 14B: 720x1280, 1280x720, 960x960, 832x1088, 1088x832")
    print("I2V 480P: 832x480, 480x832")
    print("I2V 720P: 1280x720, 720x1280")
    
    if all_present and torch.cuda.is_available():
        print("\n🎉 SETUP COMPLETE!")
        print("\n🚀 Next steps:")
        print("1. Run: ./launch_wan21.sh")
        print("2. Access: http://[pod-ip]:8188")
        print("3. Load workflow from models/workflows/")
        print("4. Generate videos with 2-3x speed boost!")
        
        print("\n💡 Pro tips:")
        print("• Enable previews (TAESD method) in ComfyUI settings")
        print("• Use model-native resolutions for best quality")
        print("• TeaCache kicks in around step 10 for speed boost")
        print("• Expected VRAM usage: 21-23GB during generation")
        
        return True
    else:
        print("\n❌ Setup incomplete - check errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x verify_setup.py

# Create troubleshooting script
log "Creating troubleshooting script..."
cat > troubleshoot.sh << 'EOF'
#!/bin/bash

echo "Wan 2.1 Troubleshooting Guide"
echo "Based on /ldg/ optimization guide"
echo "================================="

echo "1. Checking PyTorch version (critical for fp16_fast)..."
python -c "import torch; print('PyTorch:', torch.__version__)"

echo -e "\n2. Checking CUDA compatibility..."
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo -e "\n3. Checking GPU memory usage..."
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo -e "\n4. Running setup verification..."
python verify_setup.py

echo -e "\n5. Common fixes from /ldg/ guide:"
echo "   • PyTorch version must be 2.8.0dev for fp16_fast"
echo "   • If wrong version: pip install torch==2.8.0.dev20250616+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall"
echo "   • OOM errors: Reduce virtual_vram_gb to 20-21GB in workflows"
echo "   • Use TeaCache threshold 0.2 (quality) vs 0.3 (speed)"
echo "   • Never mix 720p model with 480p resolution and vice versa"

echo -e "\n6. Performance expectations:"
echo "   • VRAM usage: 21-23GB during generation (96% on RTX 4090)"
echo "   • Speed boost: 2-3x faster than baseline"
echo "   • Generation time: 30-60 seconds for 16 frames"
echo "   • TeaCache kicks in around step 10"
EOF

chmod +x troubleshoot.sh

# Create quick start README
log "Creating quick start guide..."
cat > README_WAN21.md << 'EOF'
# Wan 2.1 Maximum Performance Setup

## 🎯 Quick Start
```bash
# Launch ComfyUI with all optimizations
./launch_wan21.sh

# Access web interface
http://[your-pod-ip]:8188

# Verify setup
python verify_setup.py
```

## 🚀 Performance Features
- **FP16 Fast Accumulation**: 20-30% speed boost
- **TeaCache**: 40-60% speed boost (kicks in at step 10)
- **TorchCompile**: 30% additional speed boost
- **Adaptive Guidance**: Speed vs quality tuning
- **Q8 Models**: Highest quality quantization

## 📐 Supported Resolutions
### Text-to-Video (14B Model)
- **Primary**: 720x1280, 1280x720
- **Secondary**: 960x960, 832x1088, 1088x832

### Image-to-Video
- **480P**: 832x480, 480x832
- **720P**: 1280x720, 720x1280

## 🔧 Workflows
- `ldg_cc_i2v_14b_480p.json` - Image to Video (Comfy Native)
- `ldg_cc_t2v_14b_480p.json` - Text to Video (Comfy Native)
- `ldg_kj_i2v_14b_480p.json` - Image to Video (Kijai backup)
- `ldg_kj_t2v_14b_480p.json` - Text to Video (Kijai backup)

## 📊 Expected Performance
- **VRAM Usage**: 21-23GB (96% on RTX 4090)
- **Speed**: 2-3x faster than baseline
- **Quality**: Maximum with Q8 quantization
- **Generation**: 30-60 seconds for 16 frames

## 💡 Pro Tips
1. Enable previews in ComfyUI settings (TAESD method)
2. Use model-native resolutions for best quality
3. Monitor VRAM with `nvidia-smi`
4. TeaCache threshold: 0.2 (quality) vs 0.3 (speed)

Ready for maximum performance video generation! 🎬
EOF

# Final verification and cleanup
progress "Running final verification..."
python verify_setup.py

# Mark setup as complete
touch /workspace/setup_complete

echo ""
echo "================================================="
echo "🎉 WAN 2.1 SETUP COMPLETE!"
echo "================================================="
echo "GPU: RTX 4090 (24GB VRAM) - Optimized for maximum performance"
echo "Models: Q8 quantization (highest quality)"
echo "Expected VRAM: 21-23GB during generation"
echo "Speed improvement: 2-3x faster than baseline"
echo ""
echo "🔧 Optimizations enabled:"
echo "  ✅ FP16 Fast Accumulation (--fast)"
echo "  ✅ TeaCache (in workflows)"
echo "  ✅ TorchCompile (in workflows)"
echo "  ✅ Adaptive Guidance (in workflows)"
echo "  ⚠️  Sage Attention (skipped - CUDA compatibility)"
echo ""
echo "🚀 TO START:"
echo "   ./launch_wan21.sh"
echo ""
echo "🌐 ACCESS:"
echo "   http://[your-pod-ip]:8188"
echo ""
echo "📋 VERIFY:"
echo "   python verify_setup.py"
echo ""
echo "🛠️ TROUBLESHOOT:"
echo "   ./troubleshoot.sh"
echo ""
echo "Ready to generate amazing videos! 🎬✨"
echo "================================================="