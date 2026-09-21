# Environment Setup

The commands below target Linux, Python 3.12, and CUDA 13. Run all inference commands from the `OmniFysics-Nano-V2/` repository root.

## Conda Setup

Step 1: Create the Python environment.

```bash
conda create -n OmniFysics-Nano-V2 python=3.12.11 -y
conda activate OmniFysics-Nano-V2
```

Step 2: Install the CUDA 13 PyTorch packages.

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/test/cu130
```

Step 3: Install multimodal inference dependencies.

```bash
pip install \
  tqdm==4.70.0 psutil==7.2.2 wget==3.2 gdown==6.1.0 inflect==7.5.0 \
  scipy==1.18.0 matplotlib==3.11.1 transformers==5.7.0 torchdata==0.11.0 \
  modelscope==1.39.1 modelscope-hub==0.2.0 diffusers==0.31.0 safetensors \
  Pillow soundfile librosa==0.11.0 av==18.1.0 numpy==2.5.3
```

Step 4: Install the CosyVoice runtime dependencies. 

```bash
pip install \
  setuptools==81.0.0 torchcodec==0.10.0 openai-whisper==20250625 \
  onnxruntime-gpu==1.29.0 hyperpyyaml==1.2.3 wetext==0.1.4 \
  omegaconf==2.3.1 conformer==0.3.2 hydra-core==1.3.5 lightning==2.6.5 \
  pyworld==0.3.5 x-transformers==2.25.5 pyarrow==25.0.0
```

Step 5: Install the Flash-Attention components.

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flash-linear-attention==0.4.1 liger-kernel==0.8.1
```

## Verify the Environment

```bash
python -c "import torch, torchaudio, transformers; print(torch.__version__); print(torch.cuda.is_available()); print(torchaudio.__version__); print(transformers.__version__)"
```
