# TFLite → Core ML Conversion Status

## Goal

Convert 2 INT8 TFLite models to Core ML `.mlpackage` for GPU/Neural Engine acceleration on iOS.

- `SonoEdge/Models/heart_quality_int8full.tflite` (SQA gate, 144KB)
- `SonoEdge/Models/heart_model_int8full.tflite` (Diagnosis, 144KB)

## Approach: TF Keras Bridge

Direct TFLite→CoreML is not supported by coremltools (tested v7.1, v8.2, v9.0). Instead:

1. Rebuild model architecture in TF Keras (matching TFLite op graph)
2. Extract dequantized weights from TFLite `pseudo_qconst` tensors
3. Set weights into Keras layers
4. Convert Keras → Core ML via `coremltools.convert(source="tensorflow")`

Script: `convert_models.py`

## Model Architecture (from TFLite tensor names & flatbuffer analysis)

The model was originally **PyTorch** (`torch.nn.modules.*`), exported to TFLite. Architecture:

```
Input: [1, 1, 64, 64] NCHW INT8 → reshape → [1, 64, 64, 1] NHWC

conv1:     DW Conv(1→32ch, mult=32, 3×3) + BatchNorm(fused) + ReLU
layer2:    DW Conv(3×3) → PW Conv(32→64) → CoordAtt(64ch, r=8) → MaxPool(2×2)
layer3:    DW Conv(3×3) → PW Conv(64→128) → CoordAtt(128ch, r=8) → MaxPool(2×2)
layer4:    DW Conv(3×3) → PW Conv(128→256) → CoordAtt(256ch, r=16) → MaxPool(2×2)
global:    ReduceSum over H,W
fc:        Dense(256→2)
```

### Coordinate Attention Block (CoordAtt)

From TFLite tensor analysis (layer2 example, NHWC format):

1. pool_h = AvgPool over W axis → [B, H, 1, C]
2. pool_w = AvgPool over H axis → [B, 1, W, C]
3. Transpose pool_w [B, 1, W, C] → [B, W, 1, C]
4. Concat pool_h + pool_w_T along H → [B, H+W, 1, C]
5. Conv1×1 reduce C → C//r → ReLU
6. Split into h_branch [B, H, 1, C//r] and w_branch [B, W, 1, C//r]
7. h_branch: Conv1×1 project C//r → C → Sigmoid → h_attn [B, H, 1, C]
8. w_branch: Transpose → Conv1×1 project → Sigmoid → w_attn [B, 1, W, C]
9. Output = x × h_attn × w_attn (broadcast multiply)

Our keras CoordAtt reproduces the same math using Lambda layers, Concatenate, Conv2D, Multiply.

## Weight Mapping (verified correct)

34 INT8 weight tensors (`tfl.pseudo_qconst[0-33]`) map to 17 Keras layers:

| TFLite qconst | Shape | Keras Layer |
|---|---|---|
| qconst33/32 | (1,3,3,32)/(32,) | conv1_dw (DW, 1→32ch) |
| qconst31/30 | (1,3,3,32)/(32,) | layer2_dw (DW, 32→32ch) |
| qconst29/28 | (64,1,1,32)/(64,) | layer2_pw (1×1, 32→64) |
| qconst27/26 | (8,1,1,64)/(8,) | layer2_ca_reduce |
| qconst25/24 | (64,1,1,8)/(64,) | layer2_ca_h_proj |
| qconst23/22 | (64,1,1,8)/(64,) | layer2_ca_w_proj |
| qconst21/20 | (1,3,3,64)/(64,) | layer3_dw |
| qconst19/18 | (128,1,1,64)/(128,) | layer3_pw |
| qconst17/16 | (8,1,1,128)/(8,) | layer3_ca_reduce |
| qconst15/14 | (128,1,1,8)/(128,) | layer3_ca_h_proj |
| qconst13/12 | (128,1,1,8)/(128,) | layer3_ca_w_proj |
| qconst11/10 | (1,3,3,128)/(128,) | layer4_dw |
| qconst9/8 | (256,1,1,128)/(256,) | layer4_pw |
| qconst7/6 | (16,1,1,256)/(16,) | layer4_ca_reduce |
| qconst5/4 | (256,1,1,16)/(256,) | layer4_ca_h_proj |
| qconst3/2 | (256,1,1,16)/(256,) | layer4_ca_w_proj |
| qconst1/0 | (2,256)/(2,) | fc (Dense) |

Weight format conversions handle TFLite→Keras differences:
- DW Conv: TFLite `[1,H,W,InCh×Mult]` → Keras `[H,W,InCh,Mult]`
- PW Conv: TFLite `[OutCh,1,1,InCh]` → Keras `[1,1,InCh,OutCh]`
- FC: TFLite `[OutCh,InCh]` → Keras `[InCh,OutCh]`

All 17 layer weight shapes verified correct on last run.

## What Works

1. **Weight extraction**: 34 pseudo_qconst tensors extracted and dequantized to float32
2. **Weight mapping**: All 17 layers receive weights with correct shapes
3. **Model build**: Keras model builds and runs forward pass without errors
4. **SavedModel export**: `model.export()` works (Keras 3 format)

## Current Blockers

### Blocker 1: Keras output diverges massively from TFLite

```
TFLite output: [[-2.5456028   2.503176  ]]
Keras output:  [[-470.5264     368.25998 ]]
Diff: 467.98  (should be near 0)
```

For diagnosis model: diff ~272. This suggests the Keras model architecture doesn't match the TFLite graph. Given the 100-200× scale difference, likely causes:

- **BatchNorm**: Tensor names show `BatchNorm2d` is present (e.g., `Conv2d_0;BatchNorm2d_1;ReLU_2`). In many TFLite exports, BN is **fused** into Conv weights (weights already incorporate BN params), but if BN ops exist separately in the TFLite graph, the Keras model would need explicit BN layers or we'd need to extract BN params separately and fuse them.

- **CoordAtt detail mismatch**: The original PyTorch CoordAtt may use `AdaptiveAvgPool2d` with specific output sizes that differ from simple ReduceMean. Need to verify the exact PyTorch source code.

- **Quantization/dequantization**: Input scale=0.3247 (zp=57), output scale=0.0424 (zp=-2). The test input is random normal ~N(0,1), which should be within INT8 range after quantization.

- **Activation fusion**: The TFLite model may have fused ReLU into preceding ops, changing the effective computation.

### Blocker 2: Core ML conversion fails

```
NotImplementedError: Only a single concrete function is supported.
```

Root cause: TF 2.16 + Keras 3 exports SavedModel with multiple concrete functions. coremltools (tested up to v9.0) only supports single-function SavedModels.

Possible fixes:
- Downgrade to TF 2.12 (last tested version per coremltools docs)
- Use `tf.function` with `input_signature` to get a single concrete function, then convert that
- Save in H5 format (`.keras` or `.h5`) and try loading differently
- Use `ct.convert(model, source="tensorflow")` directly on the Keras model object instead of SavedModel path

### Blocker 3: TF/coremltools version incompatibility

TF 2.16.2 is much newer than what coremltools tests against (TF 2.12.0). This may cause additional issues even after fixing the SavedModel export.

## Quantization Parameters

| Model | Input scale/zp | Output scale/zp |
|---|---|---|
| heart_quality_int8full | 0.3246915 / 57 | 0.04242671 / -2 |
| heart_model_int8full | 0.3246915 / 57 (assumed same) | ~0.04 / ~-2 |

Input format: `[1, 1, 64, 64]` INT8 (NCHW), reshaped internally to `[1, 64, 64, 1]` NHWC.
Output format: `[1, 2]` INT8.

## Environment

**Python packages are in a virtualenv at `.venv/`** (project root). System Python is clean.

```
Python: 3.9 (venv: /Users/raexu/Documents/SonoEdge/.venv)
TensorFlow: 2.16.2 (Keras 3)
coremltools: 9.0
macOS: Darwin 25.4.0 (ARM64)
```

### How to run

```bash
# Activate venv first:
source .venv/bin/activate
python convert_models.py

# Or directly:
.venv/bin/python convert_models.py
```

### To clean up everything

```bash
rm -rf .venv
```

That's it — one command. Everything (~1.3GB) is contained in `.venv/`, nothing in system Python.

## Key Files

- `convert_models.py` — Main conversion script
- `SonoEdge/Models/heart_quality_int8full.tflite` — SQA model (144KB)
- `SonoEdge/Models/heart_model_int8full.tflite` — Diagnosis model (144KB)

## What to Try Next

1. **Debug output mismatch**: Extract intermediate layer outputs from both TFLite and Keras, compare at each stage to find where divergence begins (conv1, layer2_pw, layer2_ca, etc.)
2. **Verify BatchNorm fusion**: Parse TFLite flatbuffer ops to check if BN is fused or separate. If separate, extract BN params and add to Keras model or manually fuse.
3. **Fix Core ML conversion**: Try `ct.convert(keras_model, source="tensorflow")` directly, or use TF 2.12 in a virtualenv, or manually extract concrete function.
4. **Alternative approach**: If bridging proves unreliable, could try ONNX as intermediate: PyTorch → ONNX → coremltools (coremltools has ONNX source support).
