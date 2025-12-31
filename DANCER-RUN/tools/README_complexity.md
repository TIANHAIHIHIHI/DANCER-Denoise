# Complexity Analysis Tool

This folder contains `complexity.py` to compute model Params (M), FLOPs (M) and
inference time (ms) for the models in `models/`.

Prerequisites
- Python with PyTorch installed (use the environment you normally run experiments with)
- Install extra packages:

```bash
pip install -r requirements.txt
# or just:
pip install ptflops torchinfo
```

Quick run

```bash
python tools/complexity.py --models UNet ACDAE DACNN DANCER --input 1 1 256 --device cuda --warmup 50 --iters 200
```

Outputs
- `results/complexity.csv`: CSV with columns `model,params,flops,time_ms`.

Notes
- If a model constructor accepts `in_channels`, the script will pass the requested
  channel count (default single-channel EEG). If not, the script inserts a 1x1 Conv adapter to map the requested
  input channels to the model's expected channels so all models can be profiled
  with the same input shape.
- FLOPs are computed with `ptflops` (MACs × 2 -> FLOPs). Timing is measured on
  the selected device with CUDA syncs when appropriate.

If you want me to run the script here and collect results, confirm GPU availability and permission to execute benchmark runs.
