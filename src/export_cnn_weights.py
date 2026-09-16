"""学習済み小型CNN(cnn_embedded.keras)の重みをC言語のヘッダ(cnn_weights.h)に書き出す。
正規化パラメータ(mu/sd)も同梱。さらにC実装の検証用に、ランダムな入力と
Kerasの予測確率(cnn_ref.txt)を生成する。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from pathlib import Path
import numpy as np
from tensorflow import keras

import config as C

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "c_impl" / "cnn_weights.h"
CNN_IN = Path(r"C:\melc_tmp") / "cnn_in.txt"     # ASCIIパス(日本語回避)
CNN_REF = ROOT / "cnn_ref.txt"


def carr(name, arr):
    """numpy配列をC言語のflat float配列定義の文字列に"""
    flat = arr.flatten()
    body = ", ".join(f"{v:.8e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{{body}}};\n"


def main():
    # M1(RMS正規化+実機録音)で再学習したモデルを既定にする。
    # 旧: cnn_embedded.keras / embedded_norm.npz (固定ゲイン前提・較正が必要だった)
    if (C.MODELS / "cnn_device.keras").exists():
        model = keras.models.load_model(C.MODELS / "cnn_device.keras")
        nz = np.load(C.MODELS / "device_norm.npz")
        print("モデル: cnn_device.keras (RMS正規化・実機録音込み)")
    else:
        model = keras.models.load_model(C.MODELS / "cnn_embedded.keras")
        nz = np.load(C.MODELS / "embedded_norm.npz")
        print("モデル: cnn_embedded.keras (旧・固定ゲイン前提)")
    mu, sd = nz["mu"], nz["sd"]

    # Sequential: 0 Conv 1 Pool 2 Conv 3 Pool 4 Conv 5 GAP 6 Dense 7 Dropout 8 Dense
    c1w, c1b = model.layers[0].get_weights()
    c2w, c2b = model.layers[2].get_weights()
    c3w, c3b = model.layers[4].get_weights()
    d1w, d1b = model.layers[6].get_weights()
    d2w, d2b = model.layers[8].get_weights()

    with open(HEADER, "w") as f:
        f.write("/* 自動生成: 小型CNNの重み (export_cnn_weights.py) */\n")
        f.write("#ifndef CNN_WEIGHTS_H\n#define CNN_WEIGHTS_H\n")
        f.write(f"#define N_FEAT {mu.size}\n\n")
        f.write(carr("MU", mu)); f.write(carr("SD", sd))
        f.write(carr("C1_W", c1w)); f.write(carr("C1_B", c1b))   # (3,3,1,8)
        f.write(carr("C2_W", c2w)); f.write(carr("C2_B", c2b))   # (3,3,8,16)
        f.write(carr("C3_W", c3w)); f.write(carr("C3_B", c3b))   # (3,3,16,32)
        f.write(carr("D1_W", d1w)); f.write(carr("D1_B", d1b))   # (32,32)
        f.write(carr("D2_W", d2w)); f.write(carr("D2_B", d2b))   # (32,3)
        f.write("#endif\n")
    print(f"ヘッダ生成: {HEADER}")

    # 検証用: ランダムなraw特徴量(mu,sd相当のスケール) → Keras予測
    rng = np.random.default_rng(0)
    K = 16
    raw = (mu + sd * rng.standard_normal((K, mu.size))).astype(np.float32)
    xn = ((raw - mu) / sd).reshape(K, 32, 30, 1)
    probs = model.predict(xn, verbose=0)
    CNN_IN.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(CNN_IN, raw.flatten(), fmt="%.8e")
    np.savetxt(CNN_REF, probs.flatten(), fmt="%.8e")
    print(f"検証入力 {CNN_IN} ({raw.size}値, {K}サンプル×{mu.size}) と Keras予測 {CNN_REF} ({probs.size}値) 生成")


if __name__ == "__main__":
    main()
