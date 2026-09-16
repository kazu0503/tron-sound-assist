"""環境検証スクリプト
この企画で実際に使う処理（音声生成→メルスペクトログラム→ML）が
PC上で動くかを確認する。micro:bit は不要。
"""
import numpy as np
import scipy
import librosa
import sklearn
import soundfile as sf
import matplotlib

print("=== ライブラリ バージョン ===")
print("numpy        :", np.__version__)
print("scipy        :", scipy.__version__)
print("librosa      :", librosa.__version__)
print("scikit-learn :", sklearn.__version__)
print("soundfile    :", sf.__version__)
print("matplotlib   :", matplotlib.__version__)

# --- micro:bit 制約に合わせた特徴量パラメータ（仮確定） ---
SR = 8000        # サンプルレート 8kHz（micro:bit SAADC 想定）
N_FFT = 512      # FFTサイズ
HOP = 256        # フレームのずらし幅
N_MELS = 32      # メル帯域数

print("\n=== 1秒のテスト音（1000Hzのサイン波）でメル特徴量を計算 ===")
t = np.linspace(0, 1.0, SR, endpoint=False)
sine = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

mel = librosa.feature.melspectrogram(
    y=sine, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
)
mel_db = librosa.power_to_db(mel, ref=np.max)
print("メルスペクトログラム shape:", mel_db.shape, "(メル帯域 x フレーム数)")
print("→ この shape がモデルへの入力サイズになる")

# --- scikit-learn が動くか（ダミーデータで RandomForest 学習） ---
print("\n=== scikit-learn 動作確認（ダミーで RandomForest 学習） ===")
from sklearn.ensemble import RandomForestClassifier
X = np.random.rand(30, mel_db.size)   # 30サンプル, 特徴量=メル全体を平坦化
y = np.random.randint(0, 3, 30)        # 3クラス
clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
print("学習OK / 予測例:", clf.predict(X[:5]))

print("\n=== すべて正常。開発環境の準備完了 ===")
