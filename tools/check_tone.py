#!/usr/bin/env python3
"""
check_tone.py : 録音に「音のピーク(倍音構造)」がノイズの上に出ているかを判定する。

check_recording.py の SNR は「有音フレーム vs 静音フレーム」の差を見るため、
3秒間ずっと鳴り続けている録音では 0 に近く出てしまい役に立たない。
そこで本スクリプトは平均パワースペクトルを見て
  - 支配的な周波数(ピーク)がどこか
  - ピークがノイズ床(中央値)より何dB上か = トーナル・コントラスト
を出す。日本のサイレンは約770/960Hz付近、クラクションは約400-500Hzが基本波。

使い方: python tools/check_tone.py [wav ...]   (省略時は data/device 全部)
"""
import sys
import wave
import struct
from pathlib import Path

import numpy as np

N_FFT = 2048          # 周波数分解能を上げる(約4Hz)


def load(path):
    with wave.open(str(path), "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        raw = w.readframes(n)
    y = np.array(struct.unpack("<%dh" % n, raw), dtype=np.float64)
    return y - y.mean(), sr


def avg_spectrum(y, sr):
    win = np.hanning(N_FFT)
    hop = N_FFT // 2
    acc, cnt = None, 0
    for s in range(0, len(y) - N_FFT + 1, hop):
        spec = np.abs(np.fft.rfft(y[s:s + N_FFT] * win)) ** 2
        acc = spec if acc is None else acc + spec
        cnt += 1
    if cnt == 0:
        return None, None
    P = acc / cnt
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    return f, P


def analyze(path):
    y, sr = load(path)
    f, P = avg_spectrum(y, sr)
    if P is None:
        print(f"{path.name}: 短すぎます")
        return

    db = 10 * np.log10(P + 1e-12)
    # 50Hz未満(DCドリフト)は除外して評価
    m = f >= 50
    fm, dbm = f[m], db[m]
    noise = np.median(dbm)                       # ノイズ床
    contrast = float(dbm.max() - noise)          # ピークの突出量

    # 上位ピーク(近接ピークをまとめる)
    order = np.argsort(dbm)[::-1]
    peaks = []
    for i in order:
        if all(abs(fm[i] - pf) > 60 for pf, _ in peaks):
            peaks.append((float(fm[i]), float(dbm[i] - noise)))
        if len(peaks) >= 4:
            break

    print(f"\n=== {path.name} ===")
    print(f"  ピーク突出量(トーナル・コントラスト) = {contrast:.1f} dB")
    print("  主な周波数: " + ", ".join(f"{pf:.0f}Hz(+{pd:.0f}dB)" for pf, pd in peaks))

    if contrast >= 20:
        v = "OK: はっきりした音のピークがあります。学習に十分使えます。"
    elif contrast >= 12:
        v = "△ 使える: ピークは出ていますが弱め。可能ならもう少し近づけて録り直すと良い。"
    else:
        v = "NG: ノイズに埋もれています。音がマイクに届いていません。"
    print(f"  判定 -> {v}")


def main(argv):
    if len(argv) > 1:
        files = [Path(a) for a in argv[1:]]
    else:
        root = Path(__file__).resolve().parents[1] / "data" / "device"
        files = sorted(root.rglob("*.wav"))
    if not files:
        print("wavが見つかりません")
        return 1
    for p in files:
        analyze(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
