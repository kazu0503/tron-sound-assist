#!/usr/bin/env python3
"""
check_recording.py : 実機録音が「学習に使える品質か」を判定する。

振幅の絶対値は小さくてよい(モデルは対数メル+標準化を使うため)。
本当に重要なのは「音の周波数パターンがノイズより上に出ているか」なので、
  1) 振幅(pp/RMS)
  2) 対数メルスペクトログラムのASCII表示(模様が見えるか)
  3) SNR推定(有音フレーム vs 静かなフレームのエネルギー差)
を出す。

使い方: python tools/check_recording.py data/device/siren/xxx.wav [...]
"""
import sys
import wave
import struct
from pathlib import Path

import numpy as np

SR = 8000
N_FFT = 512
HOP = 256
N_MELS = 32


def hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_fb(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    """HTK式メルフィルタバンク(学習側 features_ref.py と同じ定義)。"""
    fmax = fmax or sr / 2
    n_bins = n_fft // 2 + 1
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        for k in range(l, min(c, n_bins)):
            if c > l:
                fb[m, k] = (k - l) / (c - l)
        for k in range(c, min(r, n_bins)):
            if r > c:
                fb[m, k] = (r - k) / (r - c)
    return fb


FB = mel_fb()
WIN = np.hanning(N_FFT + 1)[:N_FFT]   # periodic Hann


def logmel(x):
    frames = []
    for s in range(0, len(x) - N_FFT + 1, HOP):
        seg = x[s:s + N_FFT] * WIN
        spec = np.abs(np.fft.rfft(seg)) ** 2
        frames.append(FB @ spec)
    if not frames:
        return np.zeros((N_MELS, 0))
    M = np.array(frames).T                      # (n_mels, n_frames)
    return np.log(M + 1e-10)


def ascii_spec(L, rows=16, cols=60):
    """対数メルをASCIIの濃淡で表示。模様が見えれば音が入っている。"""
    chars = " .:-=+*#%@"
    n_m, n_t = L.shape
    if n_t == 0:
        return "(データなし)"
    # 表示サイズへ縮約
    mi = np.linspace(0, n_m - 1, rows).astype(int)
    ti = np.linspace(0, n_t - 1, cols).astype(int)
    S = L[np.ix_(mi, ti)]
    lo, hi = np.percentile(S, 5), np.percentile(S, 99)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    Q = np.clip((S - lo) / (hi - lo), 0, 1)
    out = []
    for r in range(rows - 1, -1, -1):          # 高域を上に
        line = "".join(chars[int(q * (len(chars) - 1))] for q in Q[r])
        out.append(f"  |{line}|")
    return "\n".join(out)


def analyze(path):
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        raw = w.readframes(n)
    y = np.array(struct.unpack("<%dh" % n, raw), dtype=np.float64)

    dc = y.mean()
    ac = y - dc
    pp = int(y.max() - y.min())
    rms = float(np.sqrt((ac ** 2).mean()))

    L = logmel(ac)
    # フレームごとの総エネルギー → 有音/静音の差 = SNR目安
    fe = L.mean(axis=0)
    if len(fe) >= 8:
        hi = np.percentile(fe, 90)             # 鳴っている時
        lo = np.percentile(fe, 10)             # 静かな時
        snr_db = float((hi - lo) * 10 / np.log(10))
    else:
        snr_db = 0.0

    print(f"\n=== {path.name} ===")
    print(f"  {sr}Hz {n}サンプル ({n/sr:.1f}秒)  DC={dc:.0f}  pp={pp}  RMS={rms:.1f}")
    print(f"  スペクトル変化(SNR目安) = {snr_db:.1f} dB")
    print(ascii_spec(L))

    if snr_db >= 10:
        v = "OK: 音のパターンがはっきり出ています。学習に使えます。"
    elif snr_db >= 5:
        v = "△ ぎりぎり: パターンは出ていますが弱め。もう少し近づけると良い。"
    else:
        v = "NG: ほぼノイズだけ。音がマイクに届いていません。"
    print(f"  判定 -> {v}")
    return snr_db


def main(argv):
    if len(argv) < 2:
        root = Path(__file__).resolve().parents[1] / "data" / "device"
        files = sorted(root.rglob("*.wav"))
    else:
        files = [Path(a) for a in argv[1:]]
    if not files:
        print("wavが見つかりません")
        return 1
    for f in files:
        analyze(f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
