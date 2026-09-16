"""UrbanSound8K を取り込み、3クラス(siren/car_horn/other)の特徴量を作る。

UrbanSound8K(8732クリップ・10fold)から:
  - siren (929) と car_horn (429) は全採用 ← クラクションのデータ枯渇を解消
  - 残り8クラス(犬・ドリル・子供の声・空調 等の都市雑音)から一部を "other" に採用
    ← 実環境に近い負例が増え、誤報抑制にも効く

ESC-50 と同じ前処理(8kHz・1秒窓・メル32×32)。fold(1〜10)を保持してリーク防止。
"""
import csv
import numpy as np
import librosa

import config as C
from preprocess import clip_to_windows, mel_feature


def main():
    meta = C.US8K_DIR / "metadata" / "UrbanSound8K.csv"
    if not meta.exists():
        print(f"UrbanSound8K が見つかりません: {meta}")
        print("ダウンロード/展開が完了しているか確認してください。")
        return

    rows = []
    with open(meta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["slice_file_name"], int(r["fold"]), r["class"]))

    rng = np.random.default_rng(C.RANDOM_SEED)
    targets = [(fn, fold, cls) for fn, fold, cls in rows if cls in ("siren", "car_horn")]
    others = [(fn, fold) for fn, fold, cls in rows if cls not in ("siren", "car_horn")]
    idx = rng.permutation(len(others))[: C.US8K_OTHER_CLIPS]

    selected = [(fn, fold, cls) for fn, fold, cls in targets]
    selected += [(others[i][0], others[i][1], "other") for i in idx]
    print(f"採用クリップ: {len(selected)} (siren/car_horn 全件 + other {len(idx)})")

    X, y, folds = [], [], []
    skipped = 0
    for k, (fn, fold, cls) in enumerate(selected):
        path = C.US8K_DIR / "audio" / f"fold{fold}" / fn
        try:
            sig, _ = librosa.load(path, sr=C.SR, mono=True)
        except Exception:
            skipped += 1
            continue
        for w in clip_to_windows(sig):     # 1秒未満のクリップは窓0個=自動スキップ
            X.append(mel_feature(w))
            y.append(C.CLASS_TO_ID[cls])
            folds.append(fold)
        if (k + 1) % 300 == 0:
            print(f"  {k+1}/{len(selected)} 処理済み")

    X = np.stack(X); y = np.array(y, dtype=np.int64); folds = np.array(folds, dtype=np.int64)
    np.save(C.DATA_PROC / "X_us8k.npy", X)
    np.save(C.DATA_PROC / "y_us8k.npy", y)
    np.save(C.DATA_PROC / "folds_us8k.npy", folds)

    print(f"\nUrbanSound8K特徴量 X: {X.shape}  (読み込み失敗 {skipped}件)")
    for i, c in enumerate(C.CLASSES):
        print(f"  {c:10s}: {int((y == i).sum())} 窓")
    print(f"保存先: {C.DATA_PROC}")


if __name__ == "__main__":
    main()
