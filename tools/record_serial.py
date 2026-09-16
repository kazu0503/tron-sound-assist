#!/usr/bin/env python3
"""
record_serial.py : micro:bit 録音モードファームからシリアル受信して WAV 保存。

ファーム(app_main.c 録音モード)は、ボタンA押下で
    #BEGIN sr=8000 n=24000
    <int16>
    ... (n行) ...
    #END
をシリアルへ出力する。本スクリプトはそれを受信し、生int16をそのまま
16bit PCM WAV (8kHz mono) として保存する。

使い方:
    # ラベルを付けて連続録音(推奨)。ボタンAを押すたびに1ファイル保存。
    python tools/record_serial.py --port COM3 --label siren
    python tools/record_serial.py --port COM3 --label car_horn
    python tools/record_serial.py --port COM3 --label other

保存先: data/device/<label>/<label>_YYYYmmdd_HHMMSS_NN.wav
Ctrl+C で終了。
"""
import argparse
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial 未導入: venv で `python -m pip install pyserial`")

ROOT = Path(__file__).resolve().parents[1]   # tron_sound/


def save_wav(path: Path, samples, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    import struct
    # int16 にクリップして書き込み
    clipped = []
    for s in samples:
        if s > 32767:
            s = 32767
        elif s < -32768:
            s = -32768
        clipped.append(s)
    data = struct.pack("<%dh" % len(clipped), *clipped)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True,
                    help="siren / car_horn / other など。保存フォルダ名になる")
    ap.add_argument("--outdir", default=None,
                    help="保存先ルート(既定: tron_sound/data/device)")
    args = ap.parse_args()

    out_root = Path(args.outdir) if args.outdir else (ROOT / "data" / "device")
    label_dir = out_root / args.label

    print(f"[record] port={args.port} baud={args.baud} label={args.label}")
    print(f"[record] 保存先: {label_dir}")
    print("[record] A=手動録音(1秒) / B=オートトリガON・OFF。Ctrl+C で終了。\n")

    ser = serial.Serial(args.port, args.baud, timeout=2)
    # 起動直後のゴミを少し捨てる
    time.sleep(0.3)
    ser.reset_input_buffer()

    count = 0
    while True:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("#BEGIN"):
            # 例: #BEGIN sr=8000 n=24000
            sr, n = 8000, 0
            for tok in line.split():
                if tok.startswith("sr="):
                    sr = int(tok[3:])
                elif tok.startswith("n="):
                    n = int(tok[2:])
            print(f"\n[record] 受信開始 sr={sr} n={n} ...", end="", flush=True)
            samples = []
            t0 = time.time()
            while len(samples) < n:
                l2 = ser.readline().decode("ascii", errors="ignore").strip()
                if l2 == "" :
                    continue
                if l2.startswith("#END"):
                    break
                try:
                    samples.append(int(l2))
                except ValueError:
                    # ノイズ行は無視
                    continue
            # #END をまだ読んでいなければ読み飛ばす
            if not l2.startswith("#END"):
                # 末尾合わせ
                pass
            dt = time.time() - t0
            count += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{args.label}_{ts}_{count:02d}.wav"
            path = label_dir / fname
            save_wav(path, samples, sr)
            print(f" 受信 {len(samples)}/{n} samples ({dt:.1f}s) -> {path.name}")
            if len(samples) != n:
                print(f"   [警告] サンプル数が一致しません({len(samples)} != {n})。"
                      "ボーレート/接続を確認、もう一度録ってください。")
            print("[record] 次を録るにはボタンA。終了は Ctrl+C。")
        elif line.startswith("level"):
            # 例: level pp=156  → 同じ行を上書きしてメーター表示
            pp = line.split("=")[-1].strip()
            try:
                bars = "#" * min(40, int(pp) // 10)
            except ValueError:
                bars = ""
            print(f"\r  音量 pp={pp:>5}  {bars:<40}", end="", flush=True)
        elif line.startswith("==="):
            print(f"\n[mb] {line}")
        elif line.startswith("REC"):
            print("\n[record] 録音中(1秒)...")
        # それ以外は無視


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[record] 終了しました。")
