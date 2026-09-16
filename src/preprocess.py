"""ESC-50 を読み込み、3クラス(siren/car_horn/other)の特徴量データを作る。

処理の流れ:
  1. ESC-50 の meta/esc50.csv を読む
  2. 各クリップを siren / car_horn / other に振り分ける
     ("other" は多様性確保のため各カテゴリから少数ずつ採用)
  3. 8kHz で読み込み、1秒窓に分割
  4. 各窓をメルスペクトログラム(32×32)に変換し、平坦化して特徴量に
  5. X(特徴量) / y(ラベル) / folds(ESC-50のfold番号) を保存

fold を保存するのは、学習/評価で「同じ元クリップの窓が train と test に
混ざる(=データリーク)」のを防ぐため。評価時は fold で分割する。
"""
import csv
import json
import numpy as np
import librosa

import config as C


def load_metadata():
    """esc50.csv を読み、(filename, fold, category) のリストを返す"""
    meta_path = C.ESC50_DIR / "meta" / "esc50.csv"
    rows = []
    with open(meta_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["filename"], int(r["fold"]), r["category"]))
    return rows


def select_clips(rows):
    """各クリップに割り当てるクラスを決める。
    other は各カテゴリ OTHER_CLIPS_PER_CATEGORY 個までに絞る。"""
    rng = np.random.default_rng(C.RANDOM_SEED)
    selected = []          # (filename, fold, class_name)
    other_by_cat = {}
    for fn, fold, cat in rows:
        if cat in C.ESC50_TARGET_MAP:
            selected.append((fn, fold, C.ESC50_TARGET_MAP[cat]))
        else:
            other_by_cat.setdefault(cat, []).append((fn, fold))
    # other を各カテゴリから均等サンプリング
    for cat, items in other_by_cat.items():
        idx = rng.permutation(len(items))[: C.OTHER_CLIPS_PER_CATEGORY]
        for i in idx:
            fn, fold = items[i]
            selected.append((fn, fold, "other"))
    return selected


def clip_to_windows(y):
    """1次元音声を WINDOW_SEC 秒の窓に分割(端数は捨てる)"""
    win = int(C.SR * C.WINDOW_SEC)
    n = len(y) // win
    return [y[i * win:(i + 1) * win] for i in range(n)]


def mel_feature(y):
    """1窓 → メルスペクトログラム(dB) を平坦化した特徴ベクトル"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP, n_mels=C.N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=1.0)   # ref固定=実機と揃えやすい
    return mel_db.astype(np.float32).flatten()


def main():
    rows = load_metadata()
    selected = select_clips(rows)
    print(f"採用クリップ数: {len(selected)}")

    X, y, folds = [], [], []
    audio_dir = C.ESC50_DIR / "audio"
    for k, (fn, fold, cls) in enumerate(selected):
        sig, _ = librosa.load(audio_dir / fn, sr=C.SR, mono=True)
        for w in clip_to_windows(sig):
            X.append(mel_feature(w))
            y.append(C.CLASS_TO_ID[cls])
            folds.append(fold)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(selected)} クリップ処理済み")

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    folds = np.array(folds, dtype=np.int64)

    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    np.save(C.DATA_PROC / "X.npy", X)
    np.save(C.DATA_PROC / "y.npy", y)
    np.save(C.DATA_PROC / "folds.npy", folds)
    with open(C.DATA_PROC / "classes.json", "w", encoding="utf-8") as f:
        json.dump(C.CLASSES, f, ensure_ascii=False, indent=2)

    print(f"\n特徴量 X: {X.shape}  (窓数 × {C.N_MELS*32}次元)")
    for i, c in enumerate(C.CLASSES):
        print(f"  クラス {c:10s}: {int((y == i).sum())} 窓")
    print(f"保存先: {C.DATA_PROC}")


if __name__ == "__main__":
    main()
