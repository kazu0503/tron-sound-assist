"""M1: RMS正規化(レベル不変化)＋実機録音を加えた再学習。

狙い:
  実機でサイレンがCAR_HORNと誤判定される原因は、学習データ(公開音源)と
  実機入力(内蔵マイク+固定ゲイン)の分布ズレ。対策を2つ同時に入れる。

  (1) RMS正規化 … 窓ごとにDC除去+基準RMSへスケーリングしてから特徴量化する。
      入力の絶対音量が特徴量に影響しなくなり、マイクのゲイン較正が不要になる。
      Python(ここ)とC(infer_mb.c)で同一定義にすること。
  (2) 実機録音の投入 … data/device の実録音を学習に加える。公開データに比べ
      数が少ない(91窓)ため、雑音拡張を厚くして影響力を確保する。
      実機データの雑音混合には「実機で録った環境音」を使い、実機のノイズ床に
      合わせる。

評価: fold5をテストに使う(ファイル単位fold割当でリーク防止)。
      公開データ・実機データそれぞれの正解率を分けて表示する。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import config as C
import features_ref as F
from train_cnn import gather                  # 公開データ収集を再利用
from train_augmented import mix_noise
from ingest_device import gather_device, SRC_DEVICE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

rng = np.random.default_rng(C.RANDOM_SEED)
tf.random.set_seed(C.RANDOM_SEED)

K_AUG_PUB = 2        # 公開データ1窓あたりの雑音拡張数
K_AUG_DEV = 15       # 実機データは数が少ないので厚く拡張(影響力確保)
SNR_RANGE = (-5, 15)
TEST_SNR = 5.0

# 窓・フィルタバンクは1度だけ計算してベクトル化(高速化)
_W = F.hann(F.N_FFT)
_FB = F.mel_filterbank()                       # (32, 257)
_NFRAMES = (C.SR - F.N_FFT) // F.HOP + 1       # 30


def feat(sig):
    """1秒窓 → RMS正規化 → 対数メル(32×30=960次元)。C版と同一定義。"""
    sig = F.rms_normalize(sig)
    frames = np.stack([sig[t * F.HOP: t * F.HOP + F.N_FFT] * _W
                       for t in range(_NFRAMES)])
    spec = np.fft.rfft(frames, axis=1)
    power = spec.real ** 2 + spec.imag ** 2    # (30, 257)
    mel = power @ _FB.T                        # (30, 32)
    return np.log(mel + F.EPS).T.flatten().astype(np.float32)


def build_features(items, k_aug, bg):
    """原音 + 雑音拡張k個 を特徴量化する。"""
    X, y, src = [], [], []
    for (w, c, f, s) in items:
        X.append(feat(w)); y.append(c); src.append(s)
        for _ in range(k_aug):
            n = bg[int(rng.integers(len(bg)))]
            X.append(feat(mix_noise(w, n, float(rng.uniform(*SNR_RANGE)))))
            y.append(c); src.append(s)
    return X, y, src


def cnn():
    m = keras.Sequential([
        keras.Input((F.N_MELS, _NFRAMES, 1)),
        layers.Conv2D(8, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"), layers.Dropout(0.3),
        layers.Dense(len(C.CLASSES), activation="softmax"),
    ])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def report(name, y, pred, src=None):
    acc = (pred == y).mean()
    print(f"\n--- {name} (n={len(y)}) 正解率 {acc:.3f} ---")
    labels = list(range(len(C.CLASSES)))
    print(classification_report(y, pred, labels=labels,
                                target_names=C.CLASSES, digits=3, zero_division=0))
    cm = confusion_matrix(y, pred, labels=labels)
    print("混同行列(行=正解,列=予測): " + " ".join(f"{c[:8]:>8s}" for c in C.CLASSES))
    for i, c in enumerate(C.CLASSES):
        print(f"{c[:8]:>8s}  " + " ".join(f"{v:8d}" for v in cm[i]))
    return acc


def main():
    # ---- データ収集 ----
    pub = gather()
    dev = gather_device()
    print(f"\n公開データ {len(pub)}窓 / 実機データ {len(dev)}窓")

    pub_tr = [it for it in pub if it[2] != 5]
    pub_te = [it for it in pub if it[2] == 5]
    dev_tr = [it for it in dev if it[2] != 5]
    dev_te = [it for it in dev if it[2] == 5]

    other_id = C.CLASS_TO_ID["other"]
    bg_pub = [w for (w, c, f, s) in pub_tr if c == other_id]
    bg_dev = [w for (w, c, f, s) in dev_tr if c == other_id] or bg_pub
    bg_te = [w for (w, c, f, s) in pub_te if c == other_id]

    # ---- 特徴量(学習) ----
    print("特徴量を計算中...")
    Xp, yp, sp = build_features(pub_tr, K_AUG_PUB, bg_pub)
    Xd, yd, sd_ = build_features(dev_tr, K_AUG_DEV, bg_dev)
    Xtr = np.stack(Xp + Xd)
    ytr = np.array(yp + yd)
    print(f"学習窓(拡張後) 公開 {len(yp)} + 実機 {len(yd)} = {len(ytr)}")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = ((Xtr - mu) / sd).reshape(-1, F.N_MELS, _NFRAMES, 1)

    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    cwd = {int(k): float(v) for k, v in zip(np.unique(ytr), cw)}

    # ---- 学習 ----
    print("学習中...")
    model = cnn()
    model.fit(Xtr_n, ytr, epochs=30, batch_size=64, class_weight=cwd, verbose=2)

    # ---- 評価 ----
    def run(items, noisy):
        if not items:
            return None, None
        X, y = [], []
        for (w, c, f, s) in items:
            sig = mix_noise(w, bg_te[int(rng.integers(len(bg_te)))], TEST_SNR) if noisy else w
            X.append(feat(sig)); y.append(c)
        Xi = ((np.stack(X) - mu) / sd).reshape(-1, F.N_MELS, _NFRAMES, 1)
        return model.predict(Xi, verbose=0).argmax(1), np.array(y)

    print("\n" + "=" * 60)
    print("評価 (fold5)")
    print("=" * 60)
    for label, noisy in [("クリーン", False), (f"雑音入り(SNR{TEST_SNR}dB)", True)]:
        p, y = run(pub_te, noisy)
        if p is not None:
            report(f"公開データ・{label}", y, p)
    p, y = run(dev_te, False)
    if p is not None:
        report("★実機録音データ(最重要)", y, p)

    # ---- 保存 ----
    C.MODELS.mkdir(parents=True, exist_ok=True)
    model.save(C.MODELS / "cnn_device.keras")
    np.savez(C.MODELS / "device_norm.npz", mu=mu, sd=sd)
    print(f"\nパラメータ数 {model.count_params()}")
    print(f"保存: {C.MODELS / 'cnn_device.keras'} / device_norm.npz")


if __name__ == "__main__":
    main()
