{
  "name": "Wan 2.1 Maximum Performance",
  "description": "Complete Wan 2.1 video generation setup with all optimizations enabled for 24GB VRAM",
  "readme": "# Wan 2.1 Maximum Performance Template\n\nThis template provides a fully optimized Wan 2.1 setup with:\n- Q8 quantization models (highest quality)\n- All performance optimizations enabled\n- 21-23GB VRAM utilization\n- 2-3x speed improvement over baseline\n\n## Quick Start\n1. Start the pod\n2. Run: `cd /workspace && ./launch_wan21.sh`\n3. Open ComfyUI at port 8188\n4. Load workflows from `/workspace/ComfyUI/workflows/`\n\n## Models Included\n- Wan 2.1 I2V 14B 480P (Q8)\n- Wan 2.1 I2V 14B 720P (Q8)\n- Wan 2.1 T2V 14B (Q8)\n- UMT5 XXL Text Encoder\n- CLIP Vision H\n- Wan 2.1 VAE\n\n## Performance Features\n- FP16 Fast Accumulation\n- Sage Attention\n- TeaCache Optimization\n- TorchCompile (30XX/40XX/50XX series)\n- Adaptive Guidance\n\n## Expected Performance\n- Generation Speed: 30-60 seconds for 16 frames\n- Interpolation: Additional 10-15 seconds for 32 frames\n- VRAM Usage: 21-23GB during inference",
  "dockerImage": "runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
  "containerDiskInGb": 150,
  "volumeMountPath": "/workspace",
  "env": [
    {
      "key": "CUDA_HOME",
      "value": "/usr/local/cuda"
    },
    {
      "key": "PATH",
      "value": "/usr/local/cuda/bin:$PATH"
    },
    {
      "key": "LD_LIBRARY_PATH", 
      "value": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    },
    {
      "key": "PYTORCH_CUDA_ALLOC_CONF",
      "value": "max_split_size_mb:512"
    }
  ],
  "ports": [
    {
      "containerPort": 8188,
      "type": "http"
    },
    {
      "containerPort": 22,
      "type": "tcp"
    }
  ],
  "startScript": "#!/bin/bash\n\n# Wan 2.1 RunPod Template Start Script\nset -e\n\necho \"Starting Wan 2.1 Setup...\"\n\n# Update system\napt update && apt upgrade -y\napt install -y git wget curl unzip python3-pip build-essential\n\n# Set environment variables\nexport CUDA_HOME=/usr/local/cuda\nexport PATH=$CUDA_HOME/bin:$PATH\nexport LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH\n\ncd /workspace\n\n# Download and run setup script if not already done\nif [[ ! -f \"/workspace/setup_complete\" ]]; then\n    echo \"Running initial setup...\"\n    \n    # Download the setup script\n    cat > setup_wan21.sh << 'SETUP_EOF'\n#!/bin/bash\n\n# Wan 2.1 RunPod Automated Setup Script\n# Based on /ldg/ Wan 2.1 Install and Optimization Guide\n# Optimized for 24GB VRAM with all performance optimizations\n\nset -e\n\necho \"==========================================\"\necho \"Wan 2.1 RunPod Setup - Maximum Performance\"\necho \"==========================================\"\n\n# Color codes for output\nRED='\\033[0;31m'\nGREEN='\\033[0;32m'\nYELLOW='\\033[1;33m'\nNC='\\033[0m' # No Color\n\nlog() {\n    echo -e \"${GREEN}[INFO]${NC} $1\"\n}\n\nwarn() {\n    echo -e \"${YELLOW}[WARN]${NC} $1\"\n}\n\nerror() {\n    echo -e \"${RED}[ERROR]${NC} $1\"\n}\n\ncd /workspace\n\n# Clone ComfyUI\nlog \"Cloning ComfyUI...\"\nif [[ ! -d \"ComfyUI\" ]]; then\n    git clone https://github.com/comfyanonymous/ComfyUI.git\nfi\ncd ComfyUI\n\n# Create directory structure\nlog \"Creating model directories...\"\nmkdir -p models/diffusion_models\nmkdir -p models/text_encoders  \nmkdir -p models/clip_vision\nmkdir -p models/vae\nmkdir -p workflows\nmkdir -p custom_nodes\n\n# Install specific PyTorch version (CRITICAL - must be exact version)\nlog \"Installing PyTorch 2.8.0.dev20250317+cu128...\"\npip install torch==2.8.0.dev20250317+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall\n\n# Install other requirements\nlog \"Installing ComfyUI requirements...\"\npip install -r requirements.txt\n\n# Install additional performance libraries\nlog \"Installing performance libraries...\"\npip install sage-attention\npip install triton\npip install packaging\n\n# Install required custom nodes\nlog \"Installing custom nodes...\"\ncd custom_nodes\n\n# ComfyUI Manager\nif [[ ! -d \"ComfyUI-Manager\" ]]; then\n    git clone https://github.com/ltdrdata/ComfyUI-Manager.git\nfi\n\n# Video Helper Suite\nif [[ ! -d \"ComfyUI-VideoHelperSuite\" ]]; then\n    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git\nfi\n\n# GGUF support (for Comfy Native)\nif [[ ! -d \"ComfyUI-GGUF\" ]]; then\n    git clone https://github.com/city96/ComfyUI-GGUF.git\nfi\n\n# KJ Nodes\nif [[ ! -d \"ComfyUI-KJNodes\" ]]; then\n    git clone https://github.com/kijai/ComfyUI-KJNodes.git\nfi\n\n# Advanced ControlNet\nif [[ ! -d \"ComfyUI-Advanced-ControlNet\" ]]; then\n    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git\nfi\n\n# Frame Interpolation\nif [[ ! -d \"ComfyUI-Frame-Interpolation\" ]]; then\n    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git\nfi\n\ncd ..\n\n# Download workflows\nlog \"Downloading optimized workflows...\"\ncd workflows\n\n# Comfy Native workflows (recommended)\nwget -O ldg_cc_i2v_14b_480p.json https://files.catbox.moe/a8j0ei.json\nwget -O ldg_cc_t2v_14b_480p.json https://files.catbox.moe/gzwcwd.json\n\ncd ..\n\n# Download models - Comfy Native (Q8 - highest quality for 24GB VRAM)\nlog \"Downloading Wan 2.1 models (Q8 quantization for maximum quality)...\"\n\n# I2V Models\nlog \"Downloading I2V models...\"\ncd models/diffusion_models\nwget -c https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf &\nwget -c https://huggingface.co/city96/Wan2.1-I2V-14B-720P-gguf/resolve/main/wan2.1-i2v-14b-720p-Q8_0.gguf &\n\n# T2V Model  \nlog \"Downloading T2V model...\"\nwget -c https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q8_0.gguf &\n\n# Text Encoders\nlog \"Downloading text encoders...\"\ncd ../text_encoders\nwget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors &\n\n# CLIP Vision\nlog \"Downloading CLIP Vision...\"\ncd ../clip_vision\nwget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors &\n\n# VAE\nlog \"Downloading VAE...\"\ncd ../vae\nwget -c https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors &\n\n# Wait for all downloads to complete\nwait\n\ncd ../..\n\n# Create optimized launch script\nlog \"Creating optimized launch script...\"\ncat > launch_wan21.sh << 'LAUNCH_EOF'\n#!/bin/bash\n\n# Wan 2.1 Optimized Launch Script\n# All performance optimizations enabled\n\nexport CUDA_HOME=/usr/local/cuda\nexport PATH=$CUDA_HOME/bin:$PATH\nexport LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH\n\ncd /workspace/ComfyUI\n\necho \"==========================================\"\necho \"Starting Wan 2.1 with all optimizations\"\necho \"Expected VRAM usage: 21-23GB\"\necho \"==========================================\"\n\n# Launch with all optimizations\npython main.py \\\n    --listen 0.0.0.0 \\\n    --port 8188 \\\n    --use-sage-attention \\\n    --fast \\\n    --enable-cors-header \\\n    --verbose\n\nLAUNCH_EOF\n\nchmod +x launch_wan21.sh\n\n# Install all custom node requirements\nlog \"Installing custom node requirements...\"\ncd custom_nodes\nfor dir in */; do\n    if [[ -f \"$dir/requirements.txt\" ]]; then\n        log \"Installing requirements for $dir\"\n        pip install -r \"$dir/requirements.txt\" || warn \"Failed to install requirements for $dir\"\n    fi\ndone\ncd ..\n\nlog \"==========================================\"\nlog \"Setup complete! 🎉\"\nlog \"==========================================\"\nlog \"To start: ./launch_wan21.sh\"\nlog \"Expected VRAM usage: 21-23GB\"\nlog \"==========================================\"\n\n# Mark setup as complete\ntouch /workspace/setup_complete\n\nSETUP_EOF\n\n    chmod +x setup_wan21.sh\n    ./setup_wan21.sh\nelse\n    echo \"Setup already complete, starting ComfyUI...\"\nfi\n\n# Start ComfyUI\ncd /workspace/ComfyUI\n./launch_wan21.sh",
  "imageName": "wan21-max-performance",
  "isServerless": false,
  "volumeInGb": 150,
  "category": "AI/ML",
  "isRunpodOfficial": false,
  "minVcpuCount": 8,
  "minMemoryInGb": 32,
  "minGpuCount": 1,
  "gpuTypes": [
    "RTX 4090",
    "RTX A6000", 
    "RTX 3090",
    "A100 40GB",
    "A100 80GB"
  ],
  "supportedCuda": ["11.8", "12.0", "12.1"],
  "runtimeInMin": 60,
  "tags": [
    "video-generation",
    "ai",
    "comfyui",
    "wan-2.1",
    "stable-diffusion",
    "machine-learning"
  ]
}