"""エンドツーエンド検証: samples.txt(音声) を
  Python全体(特徴量→正規化→Keras) と C(predict.exe) で処理し、確率を比較。
使い方: python e2e_check.py gen   /   python e2e_check.py compare
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import features_ref as F
import config as C

mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
if mode == "gen":
    from tensorflow import keras
    sig = np.loadtxt(ROOT / "samples.txt")
    raw = F.melspec(sig).flatten()                       # 960 (mel優先)
    nz = np.load(C.MODELS / "embedded_norm.npz")
    xn = ((raw - nz["mu"]) / nz["sd"]).reshape(1, 32, 30, 1)
    model = keras.models.load_model(C.MODELS / "cnn_embedded.keras")
    probs = model.predict(xn, verbose=0).flatten()
    np.savetxt(ROOT / "e2e_ref.txt", probs, fmt="%.8e")
    print(f"Keras全体パイプライン予測: siren={probs[0]:.3f} car_horn={probs[1]:.3f} other={probs[2]:.3f}")
elif mode == "compare":
    ref = np.loadtxt(ROOT / "e2e_ref.txt")
    cout = np.loadtxt(ROOT / "e2e_c.txt")
    diff = np.abs(ref - cout)
    print(f"確率の最大絶対誤差 : {diff.max():.3e}")
    print(f"予測クラス: Python={ref.argmax()} / C={cout.argmax()}")
    print("判定: " + ("一致(OK) ✅" if diff.max() < 2e-3 and ref.argmax() == cout.argmax() else "不一致(NG) ❌"))
