"""小型CNNで3クラス分類(siren/car_horn/other)を学習・評価する。

RFの限界(recallと誤報抑制の両立不能)を、32×32メルを「画像」として
扱うCNNで打破するのが狙い。

データ: ESC-50 + 日本サイレン + UrbanSound8K を結合。
雑音データ拡張(ターゲット+街頭雑音をSNR可変で合成)を学習データに適用。
評価は fold5。クリーン版と雑音入り版の両方で測る。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import csv
import numpy as np
import librosa
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import config as C
from preprocess import load_metadata, select_clips, clip_to_windows, mel_feature
from train_augmented import mix_noise

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

rng = np.random.default_rng(C.RANDOM_SEED)
tf.random.set_seed(C.RANDOM_SEED)
K_AUG = 2
SNR_RANGE = (-5, 15)
TEST_SNR = 5.0
SIDE = 32   # メルは 32(帯域) × 32(フレーム)


def gather():
    """ESC-50 + 日本 + UrbanSound8K の全クリップを1秒窓で集める。
    各窓: (waveform, class_id, fold, src)  src: 0=ESC,1=日本,2=US8K"""
    items = []
    audio = C.ESC50_DIR / "audio"
    for fn, fold, cls in select_clips(load_metadata()):
        sig, _ = librosa.load(audio / fn, sr=C.SR, mono=True)
        for w in clip_to_windows(sig):
            items.append((w, C.CLASS_TO_ID[cls], fold, 0))

    ci = 0
    for clsname in ["siren", "car_horn"]:
        for p in sorted((C.ROOT / "data" / "japan" / clsname).glob("*")):
            if p.suffix.lower() not in (".wav", ".mp3"):
                continue
            sig, _ = librosa.load(p, sr=C.SR, mono=True)
            fold = (ci % 5) + 1
            for w in clip_to_windows(sig):
                items.append((w, C.CLASS_TO_ID[clsname], fold, 1))
            ci += 1

    meta = C.US8K_DIR / "metadata" / "UrbanSound8K.csv"
    if meta.exists():
        rows = []
        with open(meta, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["slice_file_name"], int(r["fold"]), r["class"]))
        targets = [(fn, fold, cls) for fn, fold, cls in rows if cls in ("siren", "car_horn")]
        others = [(fn, fold) for fn, fold, cls in rows if cls not in ("siren", "car_horn")]
        idx = rng.permutation(len(others))[: C.US8K_OTHER_CLIPS]
        sel = [(fn, fold, cls) for fn, fold, cls in targets]
        sel += [(others[i][0], others[i][1], "other") for i in idx]
        for fn, fold, cls in sel:
            try:
                sig, _ = librosa.load(C.US8K_DIR / "audio" / f"fold{fold}" / fn, sr=C.SR, mono=True)
            except Exception:
                continue
            for w in clip_to_windows(sig):
                items.append((w, C.CLASS_TO_ID[cls], fold, 2))
    else:
        print("※ UrbanSound8K未検出。ESC-50+日本のみで学習します。")
    return items


def build_model():
    m = keras.Sequential([
        keras.Input((SIDE, SIDE, 1)),
        layers.Conv2D(8, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(C.CLASSES), activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    items = gather()
    siren_id = C.CLASS_TO_ID["siren"]
    train = [it for it in items if it[2] != 5]
    test = [it for it in items if it[2] == 5]
    bg_tr = [w for (w, c, f, s) in train if c == C.CLASS_TO_ID["other"]]
    bg_te = [w for (w, c, f, s) in test if c == C.CLASS_TO_ID["other"]]
    print(f"窓数: 学習(拡張前) {len(train)} / テスト {len(test)}")

    # 学習: 原音 + 雑音入りK個
    Xtr, ytr = [], []
    for (w, c, f, s) in train:
        Xtr.append(mel_feature(w)); ytr.append(c)
        for _ in range(K_AUG):
            bg = bg_tr[int(rng.integers(len(bg_tr)))]
            Xtr.append(mel_feature(mix_noise(w, bg, float(rng.uniform(*SNR_RANGE))))); ytr.append(c)
    Xtr = np.stack(Xtr); ytr = np.array(ytr)
    print(f"学習窓(拡張後) {len(ytr)}")

    # 標準化(学習統計で) → 32×32×1
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_i = ((Xtr - mu) / sd).reshape(-1, SIDE, SIDE, 1)

    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    cwd = {int(k): float(v) for k, v in zip(np.unique(ytr), cw)}

    model = build_model()
    model.fit(Xtr_i, ytr, epochs=25, batch_size=64, class_weight=cwd, verbose=2)

    for label, noisy in [("クリーン", False), (f"雑音入り(SNR {TEST_SNR}dB)", True)]:
        Xte, yte, ste = [], [], []
        for (w, c, f, s) in test:
            sig = mix_noise(w, bg_te[int(rng.integers(len(bg_te)))], TEST_SNR) if noisy else w
            Xte.append(mel_feature(sig)); yte.append(c); ste.append(s)
        Xte_i = ((np.stack(Xte) - mu) / sd).reshape(-1, SIDE, SIDE, 1)
        yte = np.array(yte); ste = np.array(ste)
        pred = model.predict(Xte_i, verbose=0).argmax(1)
        print(f"\n========== CNN テスト: {label} ==========")
        print(classification_report(yte, pred, target_names=C.CLASSES, digits=3))
        cm = confusion_matrix(yte, pred)
        print("混同行列(行=正解,列=予測): " + " ".join(f"{c[:8]:>8s}" for c in C.CLASSES))
        for i, c in enumerate(C.CLASSES):
            print(f"{c[:8]:>8s}  " + " ".join(f"{v:8d}" for v in cm[i]))
        for name, sv in [("日本産", 1), ("ESC-50産", 0), ("US8K産", 2)]:
            m = (yte == siren_id) & (ste == sv)
            if m.sum():
                print(f"   サイレン {name}: recall {(pred[m]==siren_id).mean():.2f} ({int((pred[m]==siren_id).sum())}/{int(m.sum())})")

    C.MODELS.mkdir(parents=True, exist_ok=True)
    model.save(C.MODELS / "cnn_model.keras")
    np.savez(C.MODELS / "cnn_norm.npz", mu=mu, sd=sd)
    print(f"\nモデル保存: {C.MODELS / 'cnn_model.keras'}")


if __name__ == "__main__":
    main()
