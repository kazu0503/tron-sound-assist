# ビルド・書き込み・動作手順

審査用の動作確認手順です。**「4. 動作確認」だけでも作品の動きは確認できます。**

---

## 1. 必要なもの

### ハードウェア
| 品目 | 備考 |
|---|---|
| micro:bit v2 | 内蔵MEMSマイクを使うため **v2 必須**（v1 は不可） |
| USBケーブル（micro-B） | 書き込み・シリアル通信用 |
| 振動モーターモジュール | ドライバ内蔵・3.3V対応の3ピン品。**なくてもLEDで動作します** |
| ワニ口⇔オスジャンパー線 3本 | 振動モーター接続用 |

### ソフトウェア
| 品目 | バージョン |
|---|---|
| GNU Arm Embedded Toolchain | 10.3-2021.10 |
| GNU Make | 4.3 以降（Windows は xPack Windows Build Tools） |
| pyOCD | 書き込み用 |
| Python | 3.10（学習をやり直す場合のみ） |

```bash
pip install pyocd pyserial
```

### μT-Kernel 3.0
「IoTエッジノード実践キット/micro:bit」（パーソナルメディア株式会社）付属CDの
`mtk3.zip` を展開して使用します。**再配布不可のため本リポジトリには含まれていません。**

> 展開先のパスに**日本語やスペースを含めないでください**。
> ビルドが通らない原因になります（例: `C:\Users\Public\microbit_mtk3\`）。

---

## 2. ソースの配置

展開した μT-Kernel のツリーの `app_sample/` に、本リポジトリの `firmware/` から**次の4ファイルだけ**をコピーします。

```
mtkernel_3/
├── app_sample/          ← ここに次の4ファイルを入れる
│   ├── app_main.c       （本作品のアプリ本体：3タスク構成）
│   ├── infer_mb.c       （特徴量抽出＋CNN推論）
│   ├── infer_mb.h
│   └── cnn_weights.h    （学習済みの重み。自動生成物）
├── build_make/
└── ...
```

> `app_sample/` にある元のサンプル `app_main.c` は上書きされます。
> 必要なら退避しておいてください。
> また `app_sample/*.c` は**すべてビルド対象**になるため、`usermain()` を持つ
> ファイルを2つ置かないでください（リンクエラーになります）。
> `firmware/` にある `app_main_record.c`（録音用）・`app_motortest.c`（モーター配線の確認用）・
> `app_pulsetest.c`（振動の長さの確認用）も `usermain()` を持つ別アプリなので、
> **通常のビルドではコピーしないでください**。録音用を使うときは `app_main.c` と入れ替えます（5章参照）。

## 3. ビルド設定

`build_make/microbit.mk` を2か所変更します。

```make
CFLAGS := -mcpu=cortex-m4 -mthumb -ffreestanding \
    -std=gnu11 \
    -O2 -g3 \                    ← -O0 から -O2 へ（推論速度のため必須）
    -MMD -MP \
    -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
    -DFIRMWARE_BUILD             ← 追加（PC検証用 main を除外する）
```

`build_make/makefile` のリンク行に `-lm` を追加します（`logf`/`expf`/`sqrt` を使うため）。

```make
$(LINK) $(LFLAGS) -T $(LNKFILE) -Wl,-Map,"$(EXE_FILE).map" -o "$(EXE_FILE).elf" $(OBJS) -lm
```

### ビルド実行

```bash
make -C <展開先>/mtkernel_3/build_make clean
make -C <展開先>/mtkernel_3/build_make all
```

`Finished building target: mtkernel_3.elf` が出れば成功です。

サイズ確認：
```bash
arm-none-eabi-size <展開先>/mtkernel_3/build_make/mtkernel_3.elf
#   text    data     bss     dec
#  86804    1152  104516  192472      ← bss(RAM) が 128KB 未満であること
```

## 4. 書き込みと動作確認

micro:bit を USB 接続して：

```bash
pyocd load <展開先>/mtkernel_3/build_make/mtkernel_3.elf
```

シリアル接続（ボーレート **115200**）：
```bash
python -m serial.tools.miniterm COM3 115200      # Windows
python -m serial.tools.miniterm /dev/ttyACM0 115200   # Linux/macOS
```

micro:bit 裏面の**リセットボタン**を押すと起動します。

```
=== Sound Assist (RTOS tasks) ===
tasks started: audio(2) notify(3) infer(4)
=> other conf=652 (340ms) run=1
```

### 動作の確認手順

> **音は大きめに鳴らしてください。** 小さな音は周りの雑音に埋もれ、「その他」と判定されることがあります。
> スマートフォンなどの音量を最大に近づけ、スピーカーを micro:bit 表面のマイクの穴（ロゴの右横）に近づけてください。

| 手順 | 期待する動作 |
|---|---|
| ① 静かにする | `=> other` が続く。LEDは消灯のまま |
| ② 緊急車両のサイレンを2〜3秒以上鳴らす<br>（スマートフォンでの再生で可。**音量は大きめにし、マイクに近づける**） | `=> SIREN ... *** ALERT ***` と表示され、**LEDマトリクスの一番上の段が点灯**＋長い振動2回 |
| ③ クラクションを鳴らす<br>（同じく音量は大きめ。0.3秒ほどの短い音には反応しないことがある） | `=> CAR_HORN ... *** ALERT ***`、**LEDの真ん中の段が点灯**＋短い振動3回 |
| ④ 声を出す・手を叩く | `=> other` のまま。**ALERTは出ない**（誤報しないこと） |
| ⑤ 音を止めて3秒待つ | LEDが自動消灯 |

表示の意味：
- `conf` … 確信度（1000が最大）
- `(340ms)` … 推論にかかった実測時間
- `run` … 同じ判定が連続した回数（サイレンは確信度600以上が2回以上、クラクションは確信度700以上なら1回でALERT）

### 振動モーターを接続する場合

| モジュール側 | micro:bit 側 |
|---|---|
| VCC | `3V` 端子 |
| GND | `GND` 端子 |
| IN (SIG) | `P0` 端子 |

> **モーターを GPIO に直結しないでください。** micro:bit のピンは 1本あたり約5mA
> しか流せず、振動モーター（60〜100mA）を直結するとマイコンが破損します。
> 必ずトランジスタ（ドライバ）内蔵のモジュールを使ってください。

---

## 5. 学習をやり直す場合（任意）

審査に必須の手順ではありません。モデルを再学習したい場合のみ実施してください。

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
```

データセットを `data/` に配置します（出典は [../CREDITS.md](../CREDITS.md)）。

```bash
cd src
python train_device.py          # 1. 学習（公開データ＋実機録音、RMS正規化あり）
python export_cnn_weights.py    # 2. 重みを ../c_impl/cnn_weights.h に書き出す
cp ../c_impl/cnn_weights.h ../firmware/cnn_weights.h   # 3. 実機用にコピー（Windows は copy）
cd ../c_impl
gcc -O2 -o infer_mb infer_mb.c -lm   # 4. PC用の検証プログラムを新しい重みで作り直す
cd ../src
python validate_rms.py          # 5. Python と C が一致するか検証
```

4 を飛ばすと、`validate_rms.py` が古い重みの検証プログラムと比較してしまうので注意してください。

`validate_rms.py` は次の2点を確認します。
1. Python（学習側）と C（実機側）の出力が一致するか（誤差 1e-5 未満）
2. 入力音量を変えても出力が変わらないか（レベル不変性）

### 実機マイクで学習データを録音する

`firmware/app_main_record.c` を `app_sample/app_main.c` として配置し、
`infer_mb.c` を一時的に退避（拡張子を変える）してビルドします。
※ 録音用アプリは推論を行わないため、RAM 節約のために推論コードを外します。

```bash
python tools/record_serial.py --port COM3 --label siren
```

micro:bit のボタンAで手動録音、ボタンBで自動録音（音を検知したら自動で録る）の
切り替えができます。録音は `data/device/<ラベル>/` に WAV で保存されます。

品質確認：
```bash
python tools/check_window.py data/device/siren/*.wav
```
「ピーク突出量」が +20dB 以上であれば学習に使える品質です。
