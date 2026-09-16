"""評価資料用: 5分割交差検証で全データを評価する。

train_device.py の単発評価は fold5 だけをテストに使うため、実機録音の
テストが14窓しかなく統計的に弱い。ここでは fold1〜5 を順にテストにして
5回学習し、全窓(公開1178×5相当/実機91)を一度ずつ評価する。

fold はファイル単位で割り当てられているので、同じ録音から切った窓が
train と test に分かれることはない(データリークなし)。

出力: docs/EVALUATION.md に貼る混同行列と指標。
実行時間の目安: 15〜25分。
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import config as C
import features_ref as F
from train_cnn import gather
from train_augmented import mix_noise
from ingest_device import gather_device
from train_device import feat, build_features, cnn, K_AUG_PUB, K_AUG_DEV, TEST_SNR

rng = np.random.default_rng(C.RANDOM_SEED)
_NFRAMES = (C.SR - F.N_FFT) // F.HOP + 1


def predict(model, items, mu, sd, bg=None, snr=None):
    X, y = [], []
    for (w, c, f, s) in items:
        sig = mix_noise(w, bg[int(rng.integers(len(bg)))], snr) if bg is not None else w
        X.append(feat(sig)); y.append(c)
    Xi = ((np.stack(X) - mu) / sd).reshape(-1, F.N_MELS, _NFRAMES, 1)
    return model.predict(Xi, verbose=0).argmax(1), np.array(y)


def show(title, y, p):
    labels = list(range(len(C.CLASSES)))
    print(f"\n### {title}  (n={len(y)})  正解率 {(p==y).mean():.3f}")
    print(classification_report(y, p, labels=labels, target_names=C.CLASSES,
                                digits=3, zero_division=0))
    cm = confusion_matrix(y, p, labels=labels)
    print("混同行列(行=正解, 列=予測)")
    print("           " + " ".join(f"{c[:8]:>9s}" for c in C.CLASSES))
    for i, c in enumerate(C.CLASSES):
        print(f"{c[:9]:>9s}  " + " ".join(f"{v:9d}" for v in cm[i]))


def main():
    pub = gather()
    dev = gather_device()
    other_id = C.CLASS_TO_ID["other"]

    acc = {"pub": ([], []), "pub_noisy": ([], []), "dev": ([], [])}

    for test_fold in range(1, 6):
        print(f"\n{'='*60}\nfold{test_fold} をテストにして学習中...\n{'='*60}")
        pub_tr = [it for it in pub if it[2] != test_fold]
        pub_te = [it for it in pub if it[2] == test_fold]
        dev_tr = [it for it in dev if it[2] != test_fold]
        dev_te = [it for it in dev if it[2] == test_fold]

        bg_pub = [w for (w, c, f, s) in pub_tr if c == other_id]
        bg_dev = [w for (w, c, f, s) in dev_tr if c == other_id] or bg_pub
        bg_te = [w for (w, c, f, s) in pub_te if c == other_id] or bg_pub

        Xp, yp, _ = build_features(pub_tr, K_AUG_PUB, bg_pub)
        Xd, yd, _ = build_features(dev_tr, K_AUG_DEV, bg_dev)
        Xtr = np.stack(Xp + Xd); ytr = np.array(yp + yd)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xi = ((Xtr - mu) / sd).reshape(-1, F.N_MELS, _NFRAMES, 1)

        cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
        cwd = {int(k): float(v) for k, v in zip(np.unique(ytr), cw)}

        model = cnn()
        model.fit(Xi, ytr, epochs=30, batch_size=64, class_weight=cwd, verbose=0)

        p, y = predict(model, pub_te, mu, sd)
        acc["pub"][0].append(y); acc["pub"][1].append(p)
        p, y = predict(model, pub_te, mu, sd, bg_te, TEST_SNR)
        acc["pub_noisy"][0].append(y); acc["pub_noisy"][1].append(p)
        if dev_te:
            p, y = predict(model, dev_te, mu, sd)
            acc["dev"][0].append(y); acc["dev"][1].append(p)
        print(f"  fold{test_fold} 完了")

    print(f"\n\n{'#'*60}\n5分割交差検証の総合結果\n{'#'*60}")
    show("公開データ・クリーン", np.concatenate(acc["pub"][0]), np.concatenate(acc["pub"][1]))
    show(f"公開データ・雑音入り(SNR {TEST_SNR}dB)",
         np.concatenate(acc["pub_noisy"][0]), np.concatenate(acc["pub_noisy"][1]))
    show("★実機録音データ", np.concatenate(acc["dev"][0]), np.concatenate(acc["dev"][1]))


if __name__ == "__main__":
    main()
