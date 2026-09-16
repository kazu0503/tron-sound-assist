"""ESC-50(欧米) + 日本の音源 を結合して学習・評価する。
日本のサイレンを足したことで siren の取りこぼし(recall)が改善したかを見る。
評価は fold5。test内のサイレンを「ESC-50産/日本産」に分けて正解率も表示する。
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import config as C


def load_all():
    X = [np.load(C.DATA_PROC / "X.npy")]
    y = [np.load(C.DATA_PROC / "y.npy")]
    folds = [np.load(C.DATA_PROC / "folds.npy")]
    src = [np.zeros(len(y[0]), dtype=np.int64)]            # 0 = ESC-50
    jp = C.DATA_PROC / "X_japan.npy"
    if jp.exists():
        Xj = np.load(jp)
        X.append(Xj)
        y.append(np.load(C.DATA_PROC / "y_japan.npy"))
        folds.append(np.load(C.DATA_PROC / "folds_japan.npy"))
        src.append(np.ones(len(Xj), dtype=np.int64))        # 1 = 日本
    return (np.concatenate(X), np.concatenate(y),
            np.concatenate(folds), np.concatenate(src))


def main():
    X, y, folds, src = load_all()
    classes = json.loads((C.DATA_PROC / "classes.json").read_text(encoding="utf-8"))
    siren_id = C.CLASS_TO_ID["siren"]

    print("=== データ構成(窓数) ===")
    for i, c in enumerate(classes):
        n_esc = int(((y == i) & (src == 0)).sum())
        n_jp = int(((y == i) & (src == 1)).sum())
        print(f"  {c:10s}: ESC-50 {n_esc:4d} + 日本 {n_jp:4d} = {n_esc+n_jp}")

    tr = folds != 5
    te = folds == 5
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=C.RANDOM_SEED, n_jobs=-1,
    ).fit(X[tr], y[tr])
    pred = clf.predict(X[te])

    print("\n=== 分類レポート(評価fold5) ===")
    print(classification_report(y[te], pred, target_names=classes, digits=3))

    print("=== 混同行列(行=正解, 列=予測) ===")
    cm = confusion_matrix(y[te], pred)
    print("          " + " ".join(f"{c[:8]:>8s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:8]:>8s}  " + " ".join(f"{v:8d}" for v in cm[i]))

    # サイレンを「ESC-50産 / 日本産」に分けて recall を確認
    print("\n=== サイレンの取りこぼし内訳(test) ===")
    for label, s in [("ESC-50産", 0), ("日本産", 1)]:
        mask = te & (y == siren_id) & (src == s)
        n = int(mask.sum())
        if n:
            hit = int((clf.predict(X[mask]) == siren_id).sum())
            print(f"  {label}: {hit}/{n} 正解 (recall {hit/n:.2f})")

    C.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, C.MODELS / "rf_combined.joblib")
    print(f"\nモデル保存: {C.MODELS / 'rf_combined.joblib'}")


if __name__ == "__main__":
    main()
