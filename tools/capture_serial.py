#!/usr/bin/env python3
"""
capture_serial.py : micro:bit をリセットして、起動直後からのシリアル出力を一定時間記録する。

miniterm を手で開くと起動直後の数行を取りこぼしやすいので、先にシリアルを開いてから
pyocd でリセットをかけ、指定秒数ぶんの出力をそのまま表示する。

使い方:
    python tools/capture_serial.py --port COM3 --seconds 15
    python tools/capture_serial.py --port COM3 --seconds 15 --no-reset
"""
import argparse
import subprocess
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial 未導入: venv で `python -m pip install pyserial`")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--no-reset", action="store_true",
                    help="リセットせずに今流れている出力だけを記録する")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    ser.reset_input_buffer()

    if not args.no_reset:
        # シリアルを開いた後にリセットするので、起動直後の行も取りこぼさない
        r = subprocess.run(["pyocd", "reset"], capture_output=True, text=True)
        if r.returncode != 0:
            print("[capture] pyocd reset に失敗:", r.stderr.strip()[-300:])

    end = time.time() + args.seconds
    buf = b""
    while time.time() < end:
        buf += ser.read(512)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            print(line.decode("ascii", errors="replace").rstrip("\r"), flush=True)
    if buf:
        print(buf.decode("ascii", errors="replace"))
    ser.close()


if __name__ == "__main__":
    main()
