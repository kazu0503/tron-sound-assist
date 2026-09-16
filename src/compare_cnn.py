"""C版CNN推論(cnn_c_out.txt)とKeras(cnn_ref.txt)の予測確率を比較する。"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ref = np.loadtxt(ROOT / "cnn_ref.txt").reshape(-1, 3)
cout = np.loadtxt(ROOT / "cnn_c_out.txt").reshape(-1, 3)

print(f"サンプル数 {ref.shape[0]}")
diff = np.abs(ref - cout)
print(f"確率の最大絶対誤差 : {diff.max():.3e}")
print(f"確率の平均絶対誤差 : {diff.mean():.3e}")
agree = (ref.argmax(1) == cout.argmax(1)).mean()
print(f"argmax(予測クラス)一致率 : {agree*100:.1f}%")
ok = diff.max() < 2e-3 and agree == 1.0
print("判定: " + ("一致(OK) ✅" if ok else "不一致(NG) ❌"))
