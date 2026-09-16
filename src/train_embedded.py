"""組み込み版メル特徴量(features_ref / melc.c と一致)で再学習し、
小型MLPと小型CNNの精度を比較する。

目的: 推論のC実装方式を決める材料を得る。
  - 差が小さい → X案: 手書き小型MLP(依存なし・端から端までC検証可・固く完成)
  - 差が大きい → Y案: CNNをNNabla C Runtime等でC化(高精度)

特徴量は melc.c と同一定義(librosa非依存・32×30=960)。
データ: ESC-50+日本+UrbanSound8K、雑音拡張あり、fold5評価、クリーン/雑音両方。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import config as C
import features_ref as F
from train_cnn import gather          # 波形収集を再利用
from train_augmented import mix_noise

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

rng = np.random.default_rng(C.RANDOM_SEED)
tf.random.set_seed(C.RANDOM_SEED)
K_AUG = 2
SNR_RANGE = (-5, 15)
TEST_SNR = 5.0

# 特徴量(melc.cと同一)をベクトル化して高速化。窓・フィルタバンクは1度だけ計算。
_W = F.hann(F.N_FFT)
_FB = F.mel_filterbank()              # (32, 257)
_NFRAMES = (C.SR - F.N_FFT) // F.HOP + 1


def feat(sig):
    frames = np.stack([sig[t * F.HOP: t * F.HOP + F.N_FFT] * _W for t in range(_NFRAMES)])
    spec = np.fft.rfft(frames, axis=1)
    power = spec.real ** 2 + spec.imag ** 2          # (30, 257)
    mel = power @ _FB.T                                # (30, 32)
    return np.log(mel + F.EPS).T.flatten().astype(np.float32)   # (32×30=960,)


def build_features(items, augment, bg):
    X, y, src = [], [], []
    for (w, c, f, s) in items:
        X.append(feat(w)); y.append(c); src.append(s)
        if augment:
            for _ in range(K_AUG):
                n = bg[int(rng.integers(len(bg)))]
                X.append(feat(mix_noise(w, n, float(rng.uniform(*SNR_RANGE)))))
                y.append(c); src.append(s)
    return np.stack(X), np.array(y), np.array(src)


def mlp(din):
    m = keras.Sequential([
        keras.Input((din,)),
        layers.Dense(48, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(24, activation="relu"),
        layers.Dense(len(C.CLASSES), activation="softmax"),
    ])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def cnn(side_m, side_t):
    m = keras.Sequential([
        keras.Input((side_m, side_t, 1)),
        layers.Conv2D(8, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"), layers.Dropout(0.3),
        layers.Dense(len(C.CLASSES), activation="softmax"),
    ])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def evaluate(name, model, is_cnn, test, mu, sd, bg_te):
    siren_id = C.CLASS_TO_ID["siren"]
    print(f"\n########## {name} ##########")
    for label, noisy in [("クリーン", False), (f"雑音入り(SNR {TEST_SNR}dB)", True)]:
        X, y, s = [], [], []
        for (w, c, f, sv) in test:
            sig = mix_noise(w, bg_te[int(rng.integers(len(bg_te)))], TEST_SNR) if noisy else w
            X.append(feat(sig)); y.append(c); s.append(sv)
        X = ((np.stack(X) - mu) / sd); y = np.array(y); s = np.array(s)
        Xin = X.reshape(-1, F.N_MELS, _NFRAMES, 1) if is_cnn else X
        pred = model.predict(Xin, verbose=0).argmax(1)
        acc = (pred == y).mean()
        rep = classification_report(y, pred, target_names=C.CLASSES, digits=3, output_dict=True)
        print(f"[{label}] 正解率 {acc:.3f} | recall siren {rep['siren']['recall']:.2f} "
              f"car_horn {rep['car_horn']['recall']:.2f} other {rep['other']['recall']:.2f} "
              f"| precision car_horn {rep['car_horn']['precision']:.2f}")
        for nm, sv in [("日本", 1), ("ESC", 0), ("US8K", 2)]:
            m = (y == siren_id) & (s == sv)
            if m.sum():
                print(f"      サイレン{nm}: recall {(pred[m]==siren_id).mean():.2f}")


def main():
    items = gather()
    train = [it for it in items if it[2] != 5]
    test = [it for it in items if it[2] == 5]
    bg_tr = [w for (w, c, f, s) in train if c == C.CLASS_TO_ID["other"]]
    bg_te = [w for (w, c, f, s) in test if c == C.CLASS_TO_ID["other"]]

    Xtr, ytr, _ = build_features(train, augment=True, bg=bg_tr)
    print(f"学習窓(拡張後) {len(ytr)} / 特徴量次元 {Xtr.shape[1]}")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    cwd = {int(k): float(v) for k, v in zip(np.unique(ytr), cw)}

    print("\n=== MLP 学習 ===")
    m_mlp = mlp(Xtr_n.shape[1])
    m_mlp.fit(Xtr_n, ytr, epochs=30, batch_size=64, class_weight=cwd, verbose=0)
    print("=== CNN 学習 ===")
    m_cnn = cnn(F.N_MELS, _NFRAMES)
    m_cnn.fit(Xtr_n.reshape(-1, F.N_MELS, _NFRAMES, 1), ytr,
              epochs=30, batch_size=64, class_weight=cwd, verbose=0)

    evaluate("小型MLP", m_mlp, False, test, mu, sd, bg_te)
    evaluate("小型CNN", m_cnn, True, test, mu, sd, bg_te)

    print(f"\nMLPパラメータ数: {m_mlp.count_params()} / CNNパラメータ数: {m_cnn.count_params()}")
    C.MODELS.mkdir(exist_ok=True)
    m_mlp.save(C.MODELS / "mlp_embedded.keras")
    m_cnn.save(C.MODELS / "cnn_embedded.keras")
    np.savez(C.MODELS / "embedded_norm.npz", mu=mu, sd=sd)


if __name__ == "__main__":
    main()
