"""micro:bitに載せる「組み込み版メル特徴量」の Python 参照実装。
librosaは実機に載らないので、C言語と1対1で対応する素朴な定義に揃える。
このPython版とC版(melc.c)が数値一致することを検証する(validate)。

定義(C版と完全一致):
  - 入力: 8kHz・モノラル・[-1,1]の1秒(8000サンプル)
  - フレーム: 長さN_FFT=512, ホップHOP=256, センタリングなし → 30フレーム
  - 窓: Hann (periodic)  w[n] = 0.5 - 0.5*cos(2*pi*n/N)
  - FFT: 512点 → パワースペクトル(re^2+im^2), 0..256bin
  - メル: 32帯域, 0〜4000Hz, HTK式 mel=2595*log10(1+f/700), 三角フィルタ
  - 出力: log(mel_energy + 1e-10) を 32(mel) × 30(frame)

使い方:
  python features_ref.py gen       # 検証用入力(samples.txt)と参照出力(ref.txt)を生成
  python features_ref.py compare   # ref.txt と c_out.txt を比較
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent     # tron_sound/
SAMPLES = ROOT / "samples.txt"
REF = ROOT / "ref.txt"
COUT = ROOT / "c_out.txt"

SR = 8000
N_FFT = 512
HOP = 256
N_MELS = 32
FMIN = 0.0
FMAX = 4000.0
EPS = 1e-10

# --- レベル不変化(RMS正規化) ---
# 実機マイクの入力レベルは、マイクのゲイン・音源の距離・音量で大きく変わる。
# 固定ゲイン(旧MIC_GAIN=0.01)のままだと対数メルの分布がまるごとズレて誤判定する。
# そこで窓ごとに「DC除去 → 基準RMSへスケーリング」してから特徴量を作り、
# 絶対音量に影響されないようにする。Python(学習)とC(実機)で同一定義であること。
RMS_TARGET = 0.1     # 正規化後の目標RMS(float音声[-1,1]での標準的な値)
RMS_FLOOR = 1e-9     # 無音でゼロ割りしないための下限


def hann(N):
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / N)


def rms_normalize(sig):
    """DC除去 → RMSを RMS_TARGET に揃える。C版 mb_normalize と同一定義。"""
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    r = np.sqrt(np.mean(x * x))
    if r < RMS_FLOOR:
        return x                      # ほぼ無音: 増幅しない(ノイズを持ち上げない)
    return x * (RMS_TARGET / r)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank():
    n_bins = N_FFT // 2 + 1
    mpts = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2)
    fpts = mel_to_hz(mpts)
    binf = fpts / (SR / N_FFT)            # FFTビン番号(浮動小数)
    fb = np.zeros((N_MELS, n_bins))
    for m in range(N_MELS):
        l, c, r = binf[m], binf[m + 1], binf[m + 2]
        for k in range(n_bins):
            if c > l and l <= k <= c:
                fb[m, k] = (k - l) / (c - l)
            elif r > c and c < k <= r:
                fb[m, k] = (r - k) / (r - c)
    return fb


def melspec(sig, normalize=True):
    """1秒窓 → 対数メル(32×30)。normalize=True で RMS正規化を先に適用。"""
    if normalize:
        sig = rms_normalize(sig)
    w = hann(N_FFT)
    fb = mel_filterbank()
    nframes = (len(sig) - N_FFT) // HOP + 1
    out = np.zeros((N_MELS, nframes))
    for t in range(nframes):
        frame = sig[t * HOP: t * HOP + N_FFT] * w
        spec = np.fft.rfft(frame)
        power = spec.real ** 2 + spec.imag ** 2
        out[:, t] = np.log(fb @ power + EPS)
    return out


def make_test_signal():
    """検証用のテスト信号(複数トーン+ノイズ)を作る"""
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    sig = (0.5 * np.sin(2 * np.pi * 700 * t)
           + 0.3 * np.sin(2 * np.pi * 1500 * t)
           + 0.1 * rng.standard_normal(SR))
    return sig.astype(np.float64)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if mode == "gen":
        sig = make_test_signal()
        np.savetxt(SAMPLES, sig, fmt="%.10f")
        feat = melspec(sig)
        np.savetxt(REF, feat.flatten(), fmt="%.10f")
        print(f"samples.txt ({len(sig)}サンプル) と ref.txt ({feat.size}値, shape {feat.shape}) を生成")
    elif mode == "compare":
        ref = np.loadtxt(REF)
        cout = np.loadtxt(COUT)
        if ref.shape != cout.shape:
            print(f"NG: 要素数が違う ref={ref.shape} c={cout.shape}")
            sys.exit(1)
        diff = np.abs(ref - cout)
        print(f"要素数 {ref.size}")
        print(f"最大絶対誤差 : {diff.max():.3e}")
        print(f"平均絶対誤差 : {diff.mean():.3e}")
        print("判定: " + ("一致(OK) ✅" if diff.max() < 1e-6 else "不一致(NG) ❌"))
