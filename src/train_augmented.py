"""データ拡張(雑音混入)あり版の学習・評価。

狙い:
  - 学習データを「ターゲット音 + 街頭雑音(=otherの音)」を様々なSNRで合成して水増し
    → 実環境(雑音下)での頑健性を上げ、誤報も減らす
  - テストも「クリーン」と「雑音入り」の2通りで評価
    → 雑音下の正直な精度が分かる(拡張なし版の楽観的な数字を補正)

雑音源(背景)は「other」クラスの音を使う。学習用背景は学習fold、
テスト用背景はテストfoldから取り、リークを避ける。
"""
import numpy as np
import librosa

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import config as C
from preprocess import load_metadata, select_clips, clip_to_windows, mel_feature

rng = np.random.default_rng(C.RANDOM_SEED)
K_AUG = 3              # 学習窓1つあたり追加する雑音入りコピー数
SNR_RANGE = (-5, 15)   # 学習時のSNR(dB)範囲
TEST_SNR = 5.0         # 雑音入りテストのSNR(dB)


def gather_windows():
    """ESC-50 + 日本 の全クリップを1秒窓にして集める。
    各窓: (waveform, class_id, fold, src)  src: 0=ESC-50, 1=日本"""
    items = []
    audio = C.ESC50_DIR / "audio"
    for fn, fold, cls in select_clips(load_metadata()):
        sig, _ = librosa.load(audio / fn, sr=C.SR, mono=True)
        for w in clip_to_windows(sig):
            items.append((w, C.CLASS_TO_ID[cls], fold, 0))
    clip_idx = 0
    for clsname in ["siren", "car_horn"]:
        for p in sorted((C.ROOT / "data" / "japan" / clsname).glob("*")):
            if p.suffix.lower() not in (".wav", ".mp3"):
                continue
            sig, _ = librosa.load(p, sr=C.SR, mono=True)
            fold = (clip_idx % 5) + 1
            for w in clip_to_windows(sig):
                items.append((w, C.CLASS_TO_ID[clsname], fold, 1))
            clip_idx += 1
    return items


def mix_noise(sig, noise, snr_db):
    """sig に noise を指定SNRで混ぜる"""
    n = len(sig)
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    start = int(rng.integers(0, len(noise) - n + 1)) if len(noise) > n else 0
    noise = noise[start:start + n]
    sp = np.mean(sig ** 2) + 1e-9
    npow = np.mean(noise ** 2) + 1e-9
    noise = noise * np.sqrt((sp / (10 ** (snr_db / 10))) / npow)
    out = sig + noise
    m = np.max(np.abs(out))
    return (out / m if m > 1 else out).astype(np.float32)


def main():
    items = gather_windows()
    other_id = C.CLASS_TO_ID["other"]
    siren_id = C.CLASS_TO_ID["siren"]

    train = [it for it in items if it[2] != 5]
    test = [it for it in items if it[2] == 5]
    bg_train = [w for (w, c, f, s) in train if c == other_id]
    bg_test = [w for (w, c, f, s) in test if c == other_id]
    print(f"学習窓(拡張前) {len(train)} / テスト窓 {len(test)} / 背景(学習) {len(bg_train)}")

    # --- 学習データ: 原音 + 雑音入りコピーK個 ---
    Xtr, ytr = [], []
    for (w, c, f, s) in train:
        Xtr.append(mel_feature(w)); ytr.append(c)
        for _ in range(K_AUG):
            bg = bg_train[int(rng.integers(len(bg_train)))]
            snr = float(rng.uniform(*SNR_RANGE))
            Xtr.append(mel_feature(mix_noise(w, bg, snr))); ytr.append(c)
    Xtr = np.stack(Xtr); ytr = np.array(ytr)
    print(f"学習窓(拡張後) {len(ytr)}")

    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=C.RANDOM_SEED, n_jobs=-1).fit(Xtr, ytr)

    # --- テスト: クリーン版 と 雑音入り版 ---
    classes = C.CLASSES
    for label, noisy in [("クリーン", False), (f"雑音入り(SNR {TEST_SNR}dB)", True)]:
        Xte, yte, ste = [], [], []
        for (w, c, f, s) in test:
            sig = mix_noise(w, bg_test[int(rng.integers(len(bg_test)))], TEST_SNR) if noisy else w
            Xte.append(mel_feature(sig)); yte.append(c); ste.append(s)
        Xte = np.stack(Xte); yte = np.array(yte); ste = np.array(ste)
        pred = clf.predict(Xte)
        print(f"\n========== テスト: {label} ==========")
        print(classification_report(yte, pred, target_names=classes, digits=3))
        cm = confusion_matrix(yte, pred)
        print("混同行列(行=正解,列=予測): " + " ".join(f"{c[:8]:>8s}" for c in classes))
        for i, c in enumerate(classes):
            print(f"{c[:8]:>8s}  " + " ".join(f"{v:8d}" for v in cm[i]))
        # サイレンを日本産/ESC産で分けて recall
        for name, sv in [("日本産", 1), ("ESC-50産", 0)]:
            m = (yte == siren_id) & (ste == sv)
            if m.sum():
                rec = (pred[m] == siren_id).mean()
                print(f"   サイレン {name}: recall {rec:.2f} ({int((pred[m]==siren_id).sum())}/{int(m.sum())})")


if __name__ == "__main__":
    main()
