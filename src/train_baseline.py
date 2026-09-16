"""古典ML(ランダムフォレスト)で3クラス分類のベースラインを学習・評価する。

評価は ESC-50 の fold で分割する(同じ元クリップの窓が train/test に
混ざるのを防ぐ = データリーク防止)。
  - train: fold 1〜4
  - test : fold 5
混同行列で「サイレン/クラクション/その他」の取り違えを確認する。
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import config as C


def main():
    X = np.load(C.DATA_PROC / "X.npy")
    y = np.load(C.DATA_PROC / "y.npy")
    folds = np.load(C.DATA_PROC / "folds.npy")
    classes = json.loads((C.DATA_PROC / "classes.json").read_text(encoding="utf-8"))

    test_fold = 5
    tr = folds != test_fold
    te = folds == test_fold
    print(f"学習窓: {tr.sum()} / 評価窓: {te.sum()}")

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",   # クラス不均衡を補正
        random_state=C.RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X[tr], y[tr])

    pred = clf.predict(X[te])
    print("\n=== 分類レポート(評価fold) ===")
    print(classification_report(y[te], pred, target_names=classes, digits=3))

    print("=== 混同行列(行=正解, 列=予測) ===")
    cm = confusion_matrix(y[te], pred)
    header = "          " + " ".join(f"{c[:8]:>8s}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        print(f"{c[:8]:>8s}  " + " ".join(f"{v:8d}" for v in cm[i]))

    C.MODELS.mkdir(parents=True, exist_ok=True)
    out = C.MODELS / "rf_baseline.joblib"
    joblib.dump(clf, out)
    print(f"\nモデル保存: {out}")


if __name__ == "__main__":
    main()
