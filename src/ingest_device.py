"""実機(micro:bit内蔵マイク)で録音した音声を学習データとして読み込む。

data/device/<class>/*.wav (8kHz mono int16, SAADC生値) を 1秒窓に切って返す。

重要:
  - 実機の生値はDCオフセット(約870)が乗っており振幅も小さい。RMS正規化で
    吸収されるので、ここではスケール調整をしない(float化するだけ)。
  - fold は「ファイル単位」で割り当てる。同じファイルから切った窓が
    train と test に分かれるとデータリークになるため。
  - src=3 を付けて公開データ(0=ESC,1=日本,2=US8K)と区別し、評価時に
    「実機データでの正解率」を別途出せるようにする。
"""
import wave
import struct
from pathlib import Path

import numpy as np

import config as C

SRC_DEVICE = 3
DEVICE_DIR = C.ROOT / "data" / "device"


def _load_int16_wav(path):
    """8kHz mono int16 の WAV を float配列で読む(スケール変換しない)。"""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: 8kHz mono 16bit ではありません")
        if w.getframerate() != C.SR:
            raise ValueError(f"{path.name}: サンプルレートが {w.getframerate()} です")
        n = w.getnframes()
        raw = w.readframes(n)
    return np.array(struct.unpack("<%dh" % n, raw), dtype=np.float64)


def gather_device(verbose=True):
    """実機録音を1秒窓にして (waveform, class_id, fold, src) のリストで返す。"""
    items = []
    if not DEVICE_DIR.exists():
        if verbose:
            print("※ data/device が無いため実機録音はスキップします")
        return items

    win = int(C.SR * C.WINDOW_SEC)
    stats = {}
    for cls in C.CLASSES:
        d = DEVICE_DIR / cls
        if not d.exists():
            continue
        files = sorted(d.glob("*.wav"))
        nwin = 0
        for i, p in enumerate(files):
            try:
                sig = _load_int16_wav(p)
            except Exception as e:
                if verbose:
                    print(f"  [警告] 読込失敗 {p.name}: {e}")
                continue
            fold = (i % 5) + 1          # ファイル単位でfold割当(リーク防止)
            nfull = len(sig) // win
            if nfull == 0:
                continue
            for k in range(nfull):
                items.append((sig[k * win:(k + 1) * win],
                              C.CLASS_TO_ID[cls], fold, SRC_DEVICE))
                nwin += 1
        stats[cls] = (len(files), nwin)

    if verbose:
        print("実機録音(data/device):")
        for cls, (nf, nw) in stats.items():
            print(f"  {cls:10s} {nf:3d}ファイル → {nw:3d}窓")
    return items


if __name__ == "__main__":
    items = gather_device()
    print(f"\n合計 {len(items)} 窓")
    for f in range(1, 6):
        n = sum(1 for it in items if it[2] == f)
        print(f"  fold{f}: {n}窓")
