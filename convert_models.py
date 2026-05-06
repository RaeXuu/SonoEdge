#!/Users/raexu/Documents/SonoEdge/.venv/bin/python3
"""
Convert INT8 TFLite LightweightCNN models to Core ML (.mlpackage).

Strategy: Reconstruct model in TF Keras (matching TFLite graph exactly),
extract weights from TFLite, set in Keras, convert Keras -> Core ML via coremltools.

Architecture (from TFLite op graph analysis):
  Input: [1, 64, 64, 1] NHWC (after reshape from NCHW [1,1,64,64])

  conv1: DW Conv(mult=32, 3x3) + BN + ReLU  -> 1ch -> 32ch
  layer2: DW(3x3) -> PW(32->64) -> CoordAtt(64,r=8) -> MaxPool(2x2)
  layer3: DW(3x3) -> PW(64->128) -> CoordAtt(128,r=8) -> MaxPool(2x2)
  layer4: DW(3x3) -> PW(128->256) -> CoordAtt(256,r=16) -> MaxPool(2x2)
  gap: ReduceSum over H,W axes
  fc: Dense(256->2)
"""

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import coremltools as ct

MODEL_DIR = os.path.join(os.path.dirname(__file__), "SonoEdge", "Models")


# ---------------------------------------------------------------------------
# Coordinate Attention block (reconstructed from TFLite ops)
# ---------------------------------------------------------------------------
class CoordAtt(layers.Layer):
    """Coordinate Attention module matching TFLite graph exactly."""

    def __init__(self, channels, reduction, name_prefix="ca"):
        super().__init__(name=name_prefix)
        self.channels = channels
        self.reduction = reduction
        self.prefix = name_prefix

    def build(self, input_shape):
        # Weights will be set externally via set_weights
        pass

    def call(self, x):
        # Input: [B, H, W, C] NHWC
        # Step 1: Transpose to make channels first for pooling ops
        # The TFLite graph does TRANSPOSE [0,3,1,2]: NHWC -> NCHW
        x_t = tf.transpose(x, [0, 3, 1, 2])  # [B, C, H, W]

        # Step 2: Global pooling along W direction (H embedding)
        pool_h = tf.reduce_mean(x_t, axis=3, keepdims=True)  # [B, C, H, 1]

        # Step 3: Global pooling along H direction (W embedding)
        pool_w = tf.reduce_mean(x_t, axis=2, keepdims=True)  # [B, C, 1, W]

        # Step 4: Transpose pool_w [B, C, 1, W] -> [B, C, W, 1] for concat
        pool_w_t = tf.transpose(pool_w, [0, 1, 3, 2])  # [B, C, W, 1]

        # Step 5: Concat along H dimension [B, C, H+W, 1]
        cat_hw = tf.concat([pool_h, pool_w_t], axis=2)  # [B, C, 2*H, 1]
        # But pool_h is [B, C, H, 1] and pool_w_t is [B, C, W, 1]
        # For H=W=64: concat gives [B, C, 128, 1]

        # Step 6: Transpose back to NHWC for conv
        cat_nhwc = tf.transpose(cat_hw, [0, 2, 3, 1])  # [B, 128, 1, C]

        # Step 7: 1x1 Conv to reduce channels: C -> C//r
        cat_reduced = self.conv_reduce(cat_nhwc)  # [B, 128, 1, C//r]

        # Step 8: Split along H dimension (first C//r of 2*C//r)
        h_branch, w_branch = tf.split(
            cat_reduced, [self.channels, self.channels], axis=1
        )  # each [B, C//r, 1, C//r] wait this is wrong

        # Actually, the TFLite graph uses SLICE to split cat_reduced
        # cat_reduced = [B, H+W, 1, C//r]
        # h_branch = first H rows, w_branch = last W rows
        # For H=W: split in half along dim 1

        # Step 9: H branch -> 1x1 Conv to project back to C channels
        h_attn = self.conv_h(h_branch)  # [B, H, 1, C]
        h_attn = tf.sigmoid(h_attn)

        # Step 10: W branch -> transpose to [B, 1, W, C//r] -> 1x1 Conv -> [B, 1, W, C]
        w_branch_t = tf.transpose(w_branch, [0, 2, 1, 3])  # [B, 1, W, C//r]
        w_attn = self.conv_w(w_branch_t)  # [B, 1, W, C]
        w_attn = tf.sigmoid(w_attn)

        # Step 11: Apply attention
        # h_attn: [B, H, 1, C], w_attn: [B, 1, W, C]
        # Need to broadcast over the full [B, H, W, C]
        x_out = tf.multiply(x, h_attn)
        x_out = tf.multiply(x_out, w_attn)

        return x_out


# ---------------------------------------------------------------------------
# Build the exact Keras model
# ---------------------------------------------------------------------------
def build_keras_model():
    """Reconstruct the LightweightCNN in Keras matching TFLite graph."""
    inp = layers.Input(shape=(64, 64, 1), name="input")

    # ---- conv1: DW Conv (mult=32) + BN + ReLU ----
    x = layers.DepthwiseConv2D(
        kernel_size=3, padding="same", depth_multiplier=32,
        use_bias=True, name="conv1_dw"
    )(inp)
    x = layers.ReLU(name="conv1_relu")(x)

    # ---- layer2: DW -> PW -> CoordAtt -> MaxPool ----
    x = layers.DepthwiseConv2D(
        kernel_size=3, padding="same", depth_multiplier=1,
        use_bias=True, name="layer2_dw"
    )(x)
    x = layers.ReLU(name="layer2_dw_relu")(x)
    x = layers.Conv2D(64, 1, padding="valid", use_bias=True, name="layer2_pw")(x)
    x = layers.ReLU(name="layer2_pw_relu")(x)
    x = _coord_att_block(x, 64, 8, "layer2")
    x = layers.MaxPool2D(2, name="layer2_maxpool")(x)

    # ---- layer3: DW -> PW -> CoordAtt -> MaxPool ----
    x = layers.DepthwiseConv2D(
        kernel_size=3, padding="same", depth_multiplier=1,
        use_bias=True, name="layer3_dw"
    )(x)
    x = layers.ReLU(name="layer3_dw_relu")(x)
    x = layers.Conv2D(128, 1, padding="valid", use_bias=True, name="layer3_pw")(x)
    x = layers.ReLU(name="layer3_pw_relu")(x)
    x = _coord_att_block(x, 128, 8, "layer3")
    x = layers.MaxPool2D(2, name="layer3_maxpool")(x)

    # ---- layer4: DW -> PW -> CoordAtt -> MaxPool ----
    x = layers.DepthwiseConv2D(
        kernel_size=3, padding="same", depth_multiplier=1,
        use_bias=True, name="layer4_dw"
    )(x)
    x = layers.ReLU(name="layer4_dw_relu")(x)
    x = layers.Conv2D(256, 1, padding="valid", use_bias=True, name="layer4_pw")(x)
    x = layers.ReLU(name="layer4_pw_relu")(x)
    x = _coord_att_block(x, 256, 16, "layer4")
    x = layers.MaxPool2D(2, name="layer4_maxpool")(x)

    # ---- Global Sum Pool ----
    x = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=[1, 2], keepdims=False),
        name="global_sumpool"
    )(x)  # [B, 256]

    # ---- FC ----
    x = layers.Dense(2, use_bias=True, name="fc")(x)

    return keras.Model(inputs=inp, outputs=x, name="LightweightCNN")


def _coord_att_block(x, channels, reduction, name):
    """Coordinate Attention as a functional block of Keras layers.

    Matches the TFLite graph:
    1. Transpose NHWC->NCHW, pool H/W, transpose+concat
    2. Conv1x1 reduce channels
    3. Split into H/W branches
    4. Each branch: Conv1x1 project + Sigmoid
    5. Multiply with original features
    """
    prefix = f"{name}_ca"
    h = int(x.shape[1])  # spatial height
    w = int(x.shape[2])  # spatial width

    # Pool H (avg over W): [B, H, W, C] -> [B, H, 1, C]
    pool_h = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2, keepdims=True),
        name=f"{prefix}_pool_h"
    )(x)

    # Pool W (avg over H): [B, H, W, C] -> [B, 1, W, C]
    pool_w = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        name=f"{prefix}_pool_w"
    )(x)

    # Transpose pool_w to [B, W, 1, C] for concat along H axis
    pool_w_t = layers.Lambda(
        lambda t: tf.transpose(t, [0, 2, 1, 3]),
        name=f"{prefix}_pool_w_T"
    )(pool_w)

    # Concat along H axis: [B, H+W, 1, C]
    cat_hw = layers.Concatenate(axis=1, name=f"{prefix}_cat")([pool_h, pool_w_t])

    # 1x1 Conv to reduce channels: C -> C//r
    reduced = layers.Conv2D(
        reduction, 1, padding="valid", use_bias=True,
        name=f"{prefix}_reduce"
    )(cat_hw)
    reduced = layers.ReLU(name=f"{prefix}_reduce_relu")(reduced)

    # Split: first H rows -> h_branch, last W rows -> w_branch
    h_branch = layers.Lambda(
        lambda t: t[:, :h, :, :], name=f"{prefix}_h_branch"
    )(reduced)
    w_branch = layers.Lambda(
        lambda t: t[:, h:, :, :], name=f"{prefix}_w_branch"
    )(reduced)

    # H branch: 1x1 Conv -> channels, Sigmoid
    h_attn = layers.Conv2D(
        channels, 1, padding="valid", use_bias=True,
        name=f"{prefix}_h_proj"
    )(h_branch)
    h_attn = layers.Activation("sigmoid", name=f"{prefix}_h_sigmoid")(h_attn)

    # W branch: transpose to [B, 1, W, reduction] then Conv -> channels, Sigmoid
    w_branch_t = layers.Lambda(
        lambda t: tf.transpose(t, [0, 2, 1, 3]),
        name=f"{prefix}_w_branch_T"
    )(w_branch)
    w_attn = layers.Conv2D(
        channels, 1, padding="valid", use_bias=True,
        name=f"{prefix}_w_proj"
    )(w_branch_t)
    w_attn = layers.Activation("sigmoid", name=f"{prefix}_w_sigmoid")(w_attn)

    # Apply attention (broadcast)
    out = layers.Multiply(name=f"{prefix}_mul_h")([x, h_attn])
    out = layers.Multiply(name=f"{prefix}_mul_w")([out, w_attn])

    return out


# ---------------------------------------------------------------------------
# Weight extraction & mapping
# ---------------------------------------------------------------------------
def extract_tflite_weights(tflite_path):
    """Extract all pseudo_qconst weights from a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    tensors = interpreter.get_tensor_details()

    weights = {}
    for t in tensors:
        name = t["name"]
        if "pseudo_qconst" not in name:
            continue
        try:
            data = interpreter.get_tensor(t["index"])
            qp = t.get("quantization_parameters", {})
            s = qp.get("scales", None)
            z = qp.get("zero_points", None)
            weights[name] = {
                "data": data.copy(),
                "scales": np.array(s, dtype=np.float32) if s is not None else None,
                "zp": z,
            }
        except ValueError:
            pass
    return weights


def qw(tflite_w):
    """Dequantize a TFLite weight tensor to float32."""
    data = tflite_w["data"]
    scales = tflite_w["scales"]
    out = data.astype(np.float32)
    if scales is not None:
        # Broadcast scales along the first axis
        shape = [1] * out.ndim
        shape[0] = -1
        out = out * scales.reshape(shape)
    return out


def qb(tflite_w):
    """Dequantize bias tensor (INT32)."""
    data = tflite_w["data"]
    scales = tflite_w["scales"]
    out = data.astype(np.float32)
    if scales is not None:
        out = out * scales
    return out


def dw_weight_to_keras(tflite_data, in_channels, depth_multiplier):
    """
    Convert TFLite DW conv weight to Keras format.

    TFLite: [1, H, W, InCh * Mult] with per-channel scales on last axis
    Keras:  [H, W, InCh, Mult]
    """
    # Dequantize with channel scaling on last axis (axis 3 for DW conv)
    data = tflite_data["data"].astype(np.float32)
    scales = tflite_data["scales"]
    if scales is not None:
        scales = scales.reshape(1, 1, 1, -1)
        data = data * scales
    h, w = data.shape[1], data.shape[2]  # 3, 3
    reshaped = data.reshape(1, h, w, in_channels, depth_multiplier)
    return np.transpose(reshaped, (1, 2, 3, 4, 0)).squeeze(-1)
    # [H, W, InCh, Mult]


def pw_weight_to_keras(tflite_data):
    """
    Convert TFLite pointwise conv weight to Keras format.

    TFLite: [OutCh, 1, 1, InCh]
    Keras:  [1, 1, InCh, OutCh]
    """
    f = qw(tflite_data)
    return np.transpose(f, (1, 2, 3, 0))


def map_weights_to_keras(model, tflite_weights):
    """
    Map TFLite pseudo_qconst weights to Keras layer weights.
    See the op graph analysis for the index-to-layer mapping.
    """
    def w(idx):
        name = f"tfl.pseudo_qconst{idx if idx > 0 else ''}"
        if idx == 0:
            name = "tfl.pseudo_qconst"
        return tflite_weights[name]

    weight_map = {}

    # conv1_dw: TFLite idx 33,33 weight/bias, mult=32, in_ch=1
    weight_map["conv1_dw"] = [
        dw_weight_to_keras(w(33), in_channels=1, depth_multiplier=32),
        qb(w(32)),
    ]

    # layer2_dw: idx 31,30, in_ch=32, mult=1
    weight_map["layer2_dw"] = [
        dw_weight_to_keras(w(31), in_channels=32, depth_multiplier=1),
        qb(w(30)),
    ]
    # layer2_pw: idx 29,28
    weight_map["layer2_pw"] = [pw_weight_to_keras(w(29)), qb(w(28))]
    # layer2_ca_reduce: idx 27,26
    weight_map["layer2_ca_reduce"] = [pw_weight_to_keras(w(27)), qb(w(26))]
    # layer2_ca_h_proj: idx 25,24
    weight_map["layer2_ca_h_proj"] = [pw_weight_to_keras(w(25)), qb(w(24))]
    # layer2_ca_w_proj: idx 23,22
    weight_map["layer2_ca_w_proj"] = [pw_weight_to_keras(w(23)), qb(w(22))]

    # layer3_dw: idx 21,20, in_ch=64, mult=1
    weight_map["layer3_dw"] = [
        dw_weight_to_keras(w(21), in_channels=64, depth_multiplier=1),
        qb(w(20)),
    ]
    # layer3_pw: idx 19,18
    weight_map["layer3_pw"] = [pw_weight_to_keras(w(19)), qb(w(18))]
    # layer3_ca_reduce: idx 17,16
    weight_map["layer3_ca_reduce"] = [pw_weight_to_keras(w(17)), qb(w(16))]
    # layer3_ca_h_proj: idx 15,14
    weight_map["layer3_ca_h_proj"] = [pw_weight_to_keras(w(15)), qb(w(14))]
    # layer3_ca_w_proj: idx 13,12
    weight_map["layer3_ca_w_proj"] = [pw_weight_to_keras(w(13)), qb(w(12))]

    # layer4_dw: idx 11,10, in_ch=128, mult=1
    weight_map["layer4_dw"] = [
        dw_weight_to_keras(w(11), in_channels=128, depth_multiplier=1),
        qb(w(10)),
    ]
    # layer4_pw: idx 9,8
    weight_map["layer4_pw"] = [pw_weight_to_keras(w(9)), qb(w(8))]
    # layer4_ca_reduce: idx 7,6
    weight_map["layer4_ca_reduce"] = [pw_weight_to_keras(w(7)), qb(w(6))]
    # layer4_ca_h_proj: idx 5,4
    weight_map["layer4_ca_h_proj"] = [pw_weight_to_keras(w(5)), qb(w(4))]
    # layer4_ca_w_proj: idx 3,2
    weight_map["layer4_ca_w_proj"] = [pw_weight_to_keras(w(3)), qb(w(2))]

    # FC: idx 1,0
    fc_w = qw(w(1))  # [2, 256]
    weight_map["fc"] = [fc_w.T.astype(np.float32), qb(w(0))]

    return weight_map


def set_keras_weights(model, weight_map):
    """Set layer weights in the Keras model."""
    for layer in model.layers:
        if layer.name in weight_map:
            w_list = weight_map[layer.name]
            print(f"  Setting weights for {layer.name}: {[w.shape for w in w_list]}")
            # Check expected shapes
            expected = layer.get_weights()
            if len(expected) == 0:
                # Lambda layers have no weights
                continue
            try:
                layer.set_weights(w_list)
            except Exception as e:
                print(f"  WARNING: {layer.name}: {e}")
                print(f"    Expected shapes: {[w.shape for w in expected]}")
                print(f"    Got shapes:      {[w.shape for w in w_list]}")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert_tflite_to_coreml(tflite_path, output_name, label):
    print(f"\n{'='*60}")
    print(f"Converting: {label} -> {output_name}.mlpackage")
    print(f"{'='*60}")

    # 1. Build Keras model
    print("\n[1/4] Building Keras model...")
    model = build_keras_model()
    model.summary()

    # 2. Extract TFLite weights
    print("\n[2/4] Extracting TFLite weights...")
    tflite_w = extract_tflite_weights(tflite_path)
    print(f"  Extracted {len(tflite_w)} weight tensors")

    # 3. Map and set weights
    print("\n[3/4] Mapping weights to Keras layers...")
    weight_map = map_weights_to_keras(model, tflite_w)
    set_keras_weights(model, weight_map)

    # 4. Build to initialize, then test forward pass
    test_input = np.random.randn(1, 64, 64, 1).astype(np.float32)
    _ = model(test_input)

    # 5. Verify against TFLite output
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    in_s = inp_d["quantization_parameters"]["scales"][0]
    in_z = inp_d["quantization_parameters"]["zero_points"][0]
    out_s = out_d["quantization_parameters"]["scales"][0]
    out_z = out_d["quantization_parameters"]["zero_points"][0]

    # Reshape input: Keras takes [B,64,64,1], TFLite takes [B,1,64,64]
    tflite_input = np.transpose(test_input, (0, 3, 1, 2))  # [B,1,64,64]
    tflite_input_q = np.clip(np.round(tflite_input / in_s + in_z), -128, 127).astype(np.int8)
    interpreter.set_tensor(inp_d["index"], tflite_input_q)
    interpreter.invoke()
    tflite_out_q = interpreter.get_tensor(out_d["index"])
    tflite_out = (tflite_out_q.astype(np.float32) - out_z) * out_s

    keras_out = model(test_input).numpy()

    print(f"\n  TFLite output: {tflite_out}")
    print(f"  Keras output:  {keras_out}")
    print(f"  Diff: {np.abs(tflite_out - keras_out).max():.6f}")

    # 6. Convert to Core ML
    print("\n[4/4] Converting to Core ML...")
    try:
        # Save as SavedModel first
        saved_model_dir = os.path.join(MODEL_DIR, output_name + "_savedmodel")
        model.export(saved_model_dir)
        print(f"  SavedModel exported to {saved_model_dir}")

        # Convert
        mlmodel = ct.convert(
            saved_model_dir,
            source="tensorflow",
            inputs=[ct.TensorType(shape=(1, 64, 64, 1))],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS16,
        )

        mlmodel.author = "SonoEdge"
        mlmodel.short_description = (
            "Signal quality assessment gate" if "quality" in output_name
            else "Heart sound diagnosis (Normal/Abnormal)"
        )

        out_path = os.path.join(MODEL_DIR, output_name + ".mlpackage")
        mlmodel.save(out_path)
        print(f"\n  Core ML model saved: {out_path}")
        print(f"  Size: {os.path.getsize(out_path + '/weights/weight.bin')} bytes")

        # Cleanup savedmodel
        import shutil
        shutil.rmtree(saved_model_dir)

        return True

    except Exception as e:
        print(f"  ERROR during Core ML conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Starting conversion...")

    models = [
        ("heart_quality_int8full.tflite", "heart_quality_int8full", "SQA"),
        ("heart_model_int8full.tflite", "heart_model_int8full", "Diagnosis"),
    ]

    for tflite_name, output_name, label in models:
        tflite_path = os.path.join(MODEL_DIR, tflite_name)
        if not os.path.exists(tflite_path):
            print(f"SKIP: {tflite_path} not found")
            continue
        try:
            convert_tflite_to_coreml(tflite_path, output_name, label)
        except Exception as e:
            print(f"FAILED: {label} - {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
