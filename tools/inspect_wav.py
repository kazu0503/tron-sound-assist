"""録音WAVの中身を解析: サンプル数・振幅・DC・無音判定。"""
import sys
import wave
import struct

path = sys.argv[1]
with wave.open(path, "rb") as w:
    sr = w.getframerate()
    n = w.getnframes()
    ch = w.getnchannels()
    sw = w.getsampwidth()
    data = w.readframes(n)

samples = struct.unpack("<%dh" % n, data)
mn = min(samples)
mx = max(samples)
mean = sum(samples) / len(samples)
pp = mx - mn
# RMS(DC除去後)
ac = [s - mean for s in samples]
rms = (sum(v * v for v in ac) / len(ac)) ** 0.5

print(f"file   : {path}")
print(f"format : {sr}Hz {ch}ch {sw*8}bit  samples={n} ({n/sr:.2f}s)")
print(f"min={mn}  max={mx}  mean(DC)={mean:.1f}")
print(f"peak-to-peak = {pp}")
print(f"RMS(AC) = {rms:.1f}")
print(f"int16最大に対する割合: pp {pp/655.36:.2f}%  rms {rms/327.68:.3f}%")
if pp < 50:
    print(">>> ほぼ無音。マイクが拾えていない/電源OFFの可能性。")
else:
    print(">>> 信号あり。振幅が小さいだけ(プレーヤーでは聞こえにくい)。")
