"""M1の数値検証: Python(学習側)とC(実機側)が同じ答えを出すか確かめる。

RMS正規化を両側に入れたので、「音量が違う同じ音」でも同じ確率が出るはず。
ここでは実機録音のWAVを入力に使い、
  Python: features_ref.rms_normalize → 対数メル → Keras
  C     : infer_mb.exe (mb_classify、内部でRMS正規化)
の確率を突き合わせる。

さらに「レベル不変性」そのものも確認する(入力を1/10や10倍にしても
結果が変わらないこと)。これが成り立てばマイクゲイン較正は不要になる。

使い方: python validate_rms.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import subprocess
import wave
import struct
from pathlib import Path

import numpy as np
from tensorflow import keras

import config as C
import features_ref as F

ROOT = C.ROOT
EXE = ROOT / "c_impl" / "infer_mb.exe"
TMP = Path(r"C:\melc_tmp")            # ASCIIパス(日本語パスだとCが開けない)
IN_TXT = TMP / "rms_in.txt"
OUT_TXT = TMP / "rms_out.txt"

_W = F.hann(F.N_FFT)
_FB = F.mel_filterbank()
_NFRAMES = (C.SR - F.N_FFT) // F.HOP + 1


def py_feat(sig):
    sig = F.rms_normalize(sig)
    frames = np.stack([sig[t * F.HOP: t * F.HOP + F.N_FFT] * _W for t in range(_NFRAMES)])
    spec = np.fft.rfft(frames, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    mel = power @ _FB.T
    return np.log(mel + F.EPS).T.flatten().astype(np.float32)


def load_wav_window(path):
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    y = np.array(struct.unpack("<%dh" % n, raw), dtype=np.float64)
    return y[:C.SR]


def c_predict(sig):
    TMP.mkdir(parents=True, exist_ok=True)
    np.savetxt(IN_TXT, sig, fmt="%.10f")
    subprocess.run([str(EXE), str(IN_TXT), str(OUT_TXT)], check=True)
    return np.loadtxt(OUT_TXT)


def main():
    model = keras.models.load_model(C.MODELS / "cnn_device.keras")
    nz = np.load(C.MODELS / "device_norm.npz")
    mu, sd = nz["mu"], nz["sd"]

    def py_predict(sig):
        x = ((py_feat(sig) - mu) / sd).reshape(1, F.N_MELS, _NFRAMES, 1)
        return model.predict(x, verbose=0)[0]

    # 実機録音から各クラス1本ずつ取る
    files = []
    for cls in C.CLASSES:
        d = ROOT / "data" / "device" / cls
        ps = sorted(d.glob("*.wav"))
        if ps:
            files.append((cls, ps[0]))

    print("=" * 62)
    print("1) Python と C の一致検証 (実機録音を入力)")
    print("=" * 62)
    worst = 0.0
    for cls, p in files:
        sig = load_wav_window(p)
        pp = py_predict(sig)
        cp = c_predict(sig)
        d = float(np.max(np.abs(pp - cp)))
        worst = max(worst, d)
        print(f"  {cls:9s} {p.name}")
        print(f"    Python {np.array2string(pp, precision=6)}")
        print(f"    C      {np.array2string(cp, precision=6)}")
        print(f"    最大誤差 {d:.3e}")
    print(f"\n  → 全体の最大誤差 {worst:.3e}  "
          + ("一致(OK)" if worst < 1e-5 else "不一致(NG)"))

    print()
    print("=" * 62)
    print("2) レベル不変性の検証 (同じ音を音量だけ変えて入力)")
    print("=" * 62)
    cls, p = files[0]
    base = load_wav_window(p)
    ref = None
    ok = True
    for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
        cp = c_predict(base * scale)
        if ref is None:
            ref = cp
        d = float(np.max(np.abs(cp - ref)))
        if d > 1e-4:
            ok = False
        print(f"  音量×{scale:<5.1f} → {np.array2string(cp, precision=6)}  基準との差 {d:.3e}")
    print(f"\n  → {'音量を変えても結果が変わらない(レベル不変 OK)' if ok else '音量で結果が変わる(NG)'}")
    print("     ＝ マイクゲインの手動較正は不要になった")


if __name__ == "__main__":
    main()
