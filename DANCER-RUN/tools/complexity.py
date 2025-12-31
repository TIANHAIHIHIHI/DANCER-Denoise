#!/usr/bin/env python3
"""
Compute Params, FLOPs and inference time for models in this repo.

Usage example:
  python tools/complexity.py --models UNet ACDAE DACNN DANCER --input 1 2 256 --device cuda --warmup 50 --iters 200

The script will try to instantiate models with the requested input channels when
the constructor accepts `in_channels`. Otherwise it will add a 1x1 Conv adapter
to map the requested input channels to the model's expected input channels.
"""
import argparse
import csv
import inspect
import time
from typing import Any

import torch
import torch.nn as nn

try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None


def import_models_module():
    # import model classes from models/ by name
    from models import UNet, ACDAE, DACNN, DANCER

    return {"UNet": UNet, "ACDAE": ACDAE, "DACNN": DACNN, "DANCER": DANCER}


def try_construct(ModelClass: Any, in_channels: int):
    sig = inspect.signature(ModelClass.__init__)
    kwargs = {}
    if "in_channels" in sig.parameters:
        kwargs["in_channels"] = in_channels
    try:
        model = ModelClass(**kwargs)
    except Exception:
        model = ModelClass()
    return model


def first_conv_in_channels(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            try:
                return m.in_channels
            except Exception:
                continue
    return None


class AdapterModel(nn.Module):
    def __init__(self, model: nn.Module, in_channels: int, model_in: int):
        super().__init__()
        self.adapter = None
        if model_in is not None and model_in != in_channels:
            self.adapter = nn.Conv1d(in_channels, model_in, kernel_size=1)
        self.model = model

    def forward(self, x):
        if self.adapter is not None:
            x = self.adapter(x)
        return self.model(x)


def compute_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: nn.Module, input_shape: tuple):
    if get_model_complexity_info is None:
        raise RuntimeError("ptflops not available. Install with `pip install ptflops`")
    macs, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
    flops = macs * 2
    return flops, params


def measure_time(model: nn.Module, device: torch.device, input_tensor: torch.Tensor, warmup: int, iters: int):
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    avg_ms = (t1 - t0) / iters * 1000.0
    return avg_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["UNet", "ACDAE", "DACNN", "DANCER"]) 
    parser.add_argument("--input", nargs=3, type=int, default=[1, 1, 256], help="batch channels length (batch, channels, length)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", default="results/complexity.csv")
    args = parser.parse_args()

    models_pkg = import_models_module()

    batch, channels, length = args.input
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results = []
    for name in args.models:
        if name not in models_pkg:
            print(f"Model {name} not found in models package. Skipping.")
            continue
        ModelClass = models_pkg[name]
        model = try_construct(ModelClass, in_channels=channels)

        # determine model expected input channels
        model_in = first_conv_in_channels(model)

        wrapped = AdapterModel(model, in_channels=channels, model_in=model_in)

        # compute params
        params = compute_params(wrapped)

        # compute flops (on CPU)
        try:
            flops, _ = compute_flops(wrapped.cpu(), (channels, length))
        except Exception as e:
            print(f"FLOPs computation failed for {name}: {e}")
            flops = None

        # measure time on device
        input_tensor = torch.randn(batch, channels, length)
        try:
            wrapped = wrapped.to(device)
            time_ms = measure_time(wrapped, device, input_tensor, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            print(f"Timing failed for {name}: {e}")
            time_ms = None

        results.append({
            "model": name,
            "params": params / 1e6,
            "flops": (flops / 1e6) if flops is not None else None,
            "time_ms": time_ms,
        })

        print(f"{name}: params={params/1e6:.2f}M, flops={(flops/1e6) if flops else 'N/A'}M, time_ms={time_ms}")

    # write CSV
    out_path = args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "params", "flops", "time_ms"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
