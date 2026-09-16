#!/usr/bin/env python3
"""
check_window.py : 「1秒窓ごと」に音のピークを調べる。

クラクションのような短い音は、3秒の平均スペクトルで見ると無音部に薄められて
ノイズ(電源ハム等)に負けてしまう。モデルが実際に判定するのは1秒窓なので、
1秒窓を少しずつずらして調べ「一番よく鳴っている窓」を報告する。
その窓が良ければ、そこを切り出して学習に使える。

さらに低周波(<200Hz)のハム・ドリフトは除外して評価する。
クラクションの基本波は約400〜500Hz、サイレンは約700〜1000Hz。

使い方: python tools/check_window.py <wav ...>
"""
import sys
import wave
import struct
from pathlib import Path

import numpy as np

N_FFT = 2048
WIN_SEC = 1.0
HOP_SEC = 0.25
FMIN = 200.0        # これ未満(ハム/ドリフト)は評価から外す


def load(path):
    with wave.open(str(path), "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        raw = w.readframes(n)
    y = np.array(struct.unpack("<%dh" % n, raw), dtype=np.float64)
    return y - y.mean(), sr


def spectrum(y, sr):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    win = np.hanning(N_FFT)
    hop = N_FFT // 2
    acc, cnt = None, 0
    for s in range(0, len(y) - N_FFT + 1, hop):
        sp = np.abs(np.fft.rfft(y[s:s + N_FFT] * win)) ** 2
        acc = sp if acc is None else acc + sp
        cnt += 1
    P = acc / max(cnt, 1)
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    return f, P


def score(y, sr):
    """トーナル・コントラストと上位ピークを返す(低周波除外)。"""
    f, P = spectrum(y, sr)
    db = 10 * np.log10(P + 1e-12)
    m = f >= FMIN
    fm, dbm = f[m], db[m]
    noise = np.median(dbm)
    contrast = float(dbm.max() - noise)
    order = np.argsort(dbm)[::-1]
    peaks = []
    for i in order:
        if all(abs(fm[i] - pf) > 60 for pf, _ in peaks):
            peaks.append((float(fm[i]), float(dbm[i] - noise)))
        if len(peaks) >= 3:
            break
    return contrast, peaks


def analyze(path):
    y, sr = load(path)
    wn, hn = int(WIN_SEC * sr), int(HOP_SEC * sr)
    best = (-1.0, 0.0, [])
    for s in range(0, max(1, len(y) - wn + 1), hn):
        c, pk = score(y[s:s + wn], sr)
        if c > best[0]:
            best = (c, s / sr, pk)
    c, t, pk = best
    ptxt = ", ".join(f"{pf:.0f}Hz(+{pd:.0f}dB)" for pf, pd in pk)
    if c >= 20:
        v = "OK"
    elif c >= 13:
        v = "△弱い"
    else:
        v = "NG"
    print(f"  {path.name}")
    print(f"    最良1秒窓 = {t:.2f}秒地点  突出量 {c:.1f}dB  [{ptxt}]  -> {v}")
    return c, t


def main(argv):
    files = [Path(a) for a in argv[1:]]
    if not files:
        print("使い方: python tools/check_window.py <wav ...>")
        return 1
    res = []
    for p in files:
        res.append((analyze(p)[0], p.name))
    ok = sum(1 for c, _ in res if c >= 20)
    weak = sum(1 for c, _ in res if 13 <= c < 20)
    ng = sum(1 for c, _ in res if c < 13)
    print(f"\n  合計 {len(res)}本 : OK {ok} / 弱い {weak} / NG {ng}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
