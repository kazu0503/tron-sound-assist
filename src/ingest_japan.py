"""data/japan/<class>/ に置いた日本の音源を特徴量化する。
ESC-50 と同じ前処理(8kHz・1秒窓・メル32×32)で、後で結合して学習できる。

リーク防止のため、クリップ単位で fold(1〜5) を割り当てる
(同じ音源の窓が train/test に混ざらないようにする)。
wav / mp3 どちらも可。
"""
import numpy as np
import librosa

import config as C
from preprocess import clip_to_windows, mel_feature


def main():
    X, y, folds = [], [], []
    clip_idx = 0
    for cls in ["siren", "car_horn"]:
        d = C.ROOT / "data" / "japan" / cls
        files = sorted(p for p in d.glob("*") if p.suffix.lower() in (".wav", ".mp3"))
        print(f"{cls:10s}: {len(files)} ファイル")
        for p in files:
            sig, _ = librosa.load(p, sr=C.SR, mono=True)
            fold = (clip_idx % 5) + 1          # クリップ単位で5分割
            for w in clip_to_windows(sig):
                X.append(mel_feature(w))
                y.append(C.CLASS_TO_ID[cls])
                folds.append(fold)
            clip_idx += 1

    if not X:
        print("\n日本データが見つかりません。")
        print("  data/japan/siren/ と data/japan/car_horn/ に wav/mp3 を入れてください。")
        return

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    folds = np.array(folds, dtype=np.int64)
    np.save(C.DATA_PROC / "X_japan.npy", X)
    np.save(C.DATA_PROC / "y_japan.npy", y)
    np.save(C.DATA_PROC / "folds_japan.npy", folds)

    print(f"\n日本データ特徴量 X: {X.shape}")
    for i, c in enumerate(C.CLASSES):
        n = int((y == i).sum())
        if n:
            print(f"  {c:10s}: {n} 窓")
    print(f"保存先: {C.DATA_PROC}")


if __name__ == "__main__":
    main()
