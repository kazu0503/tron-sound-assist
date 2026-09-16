#!/usr/bin/env python3
"""
build_playback.py : 実機録音のための「PCスピーカー再生用ループ音源」を作る。

実機マイクはスマホ音量では届かない(実測 pp≈40=暗騒音)ため、PCスピーカーで
大音量再生する。そのために:
  - 各素材をピーク正規化(0.95)して音量を揃え、最大音量で鳴らせるようにする
  - 何本も連続で録れるよう1ファイルに連結する(再生しっぱなしにして何度もボタンA)
  - car_horn は無音区間があると3秒窓に音が入らないので、各クリップの
    「一番鳴っている区間」を切り出して詰めて並べる

出力: tron_sound/playback/siren_loop.wav / car_horn_loop.wav (44.1kHz mono)
"""
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "playback"
PB_SR = 44100          # 再生用(PCスピーカーの素の音質で鳴らす)
PEAK = 0.95


def norm(y):
    """ピーク正規化。最大音量で鳴らすため。"""
    m = np.max(np.abs(y))
    return y * (PEAK / m) if m > 1e-6 else y


def loudest_window(y, sr, sec):
    """一番エネルギーが高い sec 秒の窓を切り出す(無音部を避ける)。"""
    n = int(sec * sr)
    if len(y) <= n:
        return y
    # 0.1秒刻みでRMSを見て最大の窓を選ぶ
    hop = max(1, int(0.1 * sr))
    best, best_e = 0, -1.0
    for s in range(0, len(y) - n + 1, hop):
        e = float(np.sum(y[s:s + n] ** 2))
        if e > best_e:
            best_e, best = e, s
    return y[best:best + n]


def build(files, out_path, clip_sec, gap_sec, label):
    if not files:
        print(f"[skip] {label}: 素材が見つかりません")
        return
    parts = []
    gap = np.zeros(int(gap_sec * PB_SR), dtype=np.float32)
    used = 0
    for f in files:
        try:
            y, _ = librosa.load(str(f), sr=PB_SR, mono=True)
        except Exception as e:
            print(f"  [warn] 読込失敗 {f.name}: {e}")
            continue
        if len(y) < int(0.5 * PB_SR):
            continue
        y = loudest_window(y, PB_SR, clip_sec)
        parts.append(norm(y).astype(np.float32))
        if gap_sec > 0:
            parts.append(gap)
        used += 1
    if not parts:
        print(f"[skip] {label}: 有効な素材がありません")
        return
    out = np.concatenate(parts)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, PB_SR, subtype="PCM_16")
    print(f"[ok] {label}: {used}本 -> {out_path.name}  ({len(out)/PB_SR:.1f}秒)")


def pick_siren():
    """日本のサイレン(OtoLogic)。Close/通常を優先し種類を混ぜる。"""
    d = ROOT / "data" / "japan" / "siren"
    files = sorted(d.glob("*.mp3")) + sorted(d.glob("*.wav"))
    # Far/Dry(遠い・無響)は後回しにして、近い音を先頭に
    near = [f for f in files if "Far" not in f.name and "Dry" not in f.name]
    far = [f for f in files if f not in near]
    # 種類(Ambulance/Fire/Police)を散らすため名前順で間引く
    sel = near[:10] + far[:2]
    return sel


def pick_car_horn(n=14):
    """UrbanSound8K の car_horn。"""
    us8k = ROOT / "data" / "raw" / "UrbanSound8K"
    meta = us8k / "metadata" / "UrbanSound8K.csv"
    if not meta.exists():
        return []
    import csv
    rows = []
    with open(meta, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if r["class"] == "car_horn":
                rows.append(r)
    files = []
    for r in rows:
        p = us8k / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
        if p.exists():
            files.append(p)
    # 長さのあるものを優先(短すぎるスライスは避ける)
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[:n]


def main():
    print("再生用ループ音源を作成します...")
    # サイレン: 連続音なので8秒ずつ、間は空けない
    build(pick_siren(), OUT / "siren_loop.wav",
          clip_sec=8.0, gap_sec=0.0, label="siren")
    # クラクション: 短い音なので鳴っている2.5秒を詰めて並べる
    build(pick_car_horn(), OUT / "car_horn_loop.wav",
          clip_sec=2.5, gap_sec=0.4, label="car_horn")
    print(f"\n保存先: {OUT}")
    print("この2ファイルをPCの音楽プレーヤーで「リピート再生」し、音量を上げて録音してください。")


if __name__ == "__main__":
    sys.exit(main())
