"""プロジェクト共通設定
特徴量パラメータは micro:bit の制約に合わせて軽量に固定する。
Python側とC側(実機)で必ず同じ値を使うこと。
"""
from pathlib import Path

# --- パス ---
ROOT = Path(__file__).resolve().parent.parent      # tron_sound/
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ESC50_DIR = DATA_RAW / "ESC-50-master"             # zip展開後のフォルダ
US8K_DIR = DATA_RAW / "UrbanSound8K"               # tar.gz展開後のフォルダ
US8K_OTHER_CLIPS = 400   # UrbanSound8Kの非ターゲット8クラスから採用する"other"クリップ数

# --- 音声・特徴量パラメータ（micro:bit制約・仮確定） ---
SR = 8000            # サンプルレート 8kHz
WINDOW_SEC = 1.0     # 分類の窓（1秒）
N_FFT = 512          # FFTサイズ
HOP = 256            # フレームのずらし幅
N_MELS = 32          # メル帯域数
# → 1秒窓 = 32(メル) × 32(フレーム) のメルスペクトログラム

# --- クラス定義（A案で確定） ---
CLASSES = ["siren", "car_horn", "other"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# ESC-50 のカテゴリ名 → 本プロジェクトのクラスへの対応
# (ESC-50 の category 列の値。これ以外は全て "other" に入れる)
ESC50_TARGET_MAP = {
    "siren": "siren",
    "car_horn": "car_horn",
}

# "other" クラスの作り方:
#   狙う音以外の全48カテゴリから、各カテゴリ N 個ずつクリップを採用して
#   多様性を確保する（負例が偏ると実環境で誤報が増えるため）。
OTHER_CLIPS_PER_CATEGORY = 6   # 負例(その他)の多様性を増やし誤報を抑制

RANDOM_SEED = 0
