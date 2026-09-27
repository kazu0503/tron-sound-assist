# クレジット表記（音源素材と既存ソフトウェア）

本プロジェクトで使用した外部の音源素材と、他者が開発した既存ソフトウェアの一覧です。

## OtoLogic（CC BY 4.0 — 表記必須）
効果音: **OtoLogic**（https://otologic.jp/）

使用素材（フリー効果音「乗り物」より、CC BY 4.0）:
- 救急車サイレン: Ambulance-Siren01 / 02 / 03（Close/Far/Dry 各種）
- 消防車サイレン: Fire_Truck-Siren01（Close/Far/Dry 各種）
- パトカーサイレン: Police_Car-Siren03（Close/Far/Dry 各種）
- 出典ページ: https://otologic.jp/free/se/transportation01.html , https://otologic.jp/free/se/transportation02.html

※ ライセンス: Creative Commons Attribution 4.0 International (CC BY 4.0)。
　クレジット表記により再配布・改変が可能。

## 学習から除いた素材
- DOVA-SYNDROME（現 OpenTracks）「救急車サイレン1」 作者：稿屋 隆
  当初は学習に使っていましたが、同サイトの利用規約が音源のAIトレーニングへの使用を禁止しているため、
  2026年9月27日にこの素材を除いて学習し直しました。現在の学習済みモデル（v2）には使っていません。

## 公開データセット
- **ESC-50** (https://github.com/karolpiczak/ESC-50) — CC BY-NC 3.0。学習に使用。
  K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015.
- **UrbanSound8K** (https://urbansounddataset.weebly.com/urbansound8k.html) — CC BY-NC 3.0。
  siren 929 / car_horn 429 クリップ、および非ターゲット8クラスから400クリップを "other" として使用。
  J. Salamon, C. Jacoby, J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", ACM Multimedia 2014.

## 自己録音データ（応募者が作成）
- `data/device/` … micro:bit v2 の内蔵マイクで応募者が録音した 67ファイル / 91秒。
  実機のマイク特性に合わせるため学習に使用（siren 9 / car_horn 11 / other 47）。
  サイレンとクラクションは、次のYouTubeの効果音をスマートフォンで再生し、micro:bitのマイクで録音しました。
  利用条件は、各動画の概要欄の記載です（2026年9月27日確認）。
  - サイレン：kamada CH の再生リスト（https://www.youtube.com/playlist?list=PLT21gaEOBQw7VQ-q0LW02ldo-hT7m5bGb）の緊急車両サイレンの動画
    （例：https://www.youtube.com/watch?v=qL7o1hU9wKE 、https://www.youtube.com/watch?v=S7NpBgtxSak）。
    タイトルに「フリー素材」とあり、概要欄には「YouTube動画・TikTok・自主制作映像・イベント演出など、さまざまな場面でご活用いただけます」
    と書かれています。利用条件の詳しい記載はなく、AI学習への利用を許可または禁止する記載もありません。
  - クラクション：【著作権フリー】効果音ライブラリー「クラクション2」（https://www.youtube.com/watch?v=mhnREDHc5q8）。
    概要欄に「フリー素材、著作権フリーとして皆様に提供しているので、…ご自由にお使いください」とあり、
    禁止事項は「効果音だけを他の動画にアップロードする2次使用」のみです。
  どちらも音をリポジトリに含めず、学習（情報解析）にのみ使っています。
  録音内容には上記の音源を再生したものが含まれるため、
  再配布の可否を厳密に判断できない。よって**リポジトリには含めない**（.gitignore済み）。

## ソフトウェア（他者が開発した既存ソフトウェア）

| 名称 | 権利者 | 入手方法 | 用途 |
|---|---|---|---|
| μT-Kernel 3.0 と micro:bit 向け BSP | トロンフォーラム / パーソナルメディア株式会社 | 「IoTエッジノード実践キット/micro:bit」付属のCD | 実機で動かすリアルタイムOS |
| GNU Arm Embedded Toolchain（arm-none-eabi-gcc、newlib の数学ライブラリ） | Arm Limited / Free Software Foundation ほか | Arm の公式サイト | ファームウェアのビルド、`expf`・`logf` などの数学関数 |
| pyOCD | pyOCD の開発者（Apache License 2.0） | `pip install pyocd` | micro:bit への書き込み |
| pySerial | Chris Liechti（BSD License） | `pip install pyserial` | シリアル出力の受信、録音データの取り込み |
| TensorFlow / Keras | Google LLC ほか（Apache License 2.0） | `pip install tensorflow` | PC でのモデルの学習（実機では使わない） |
| NumPy・SciPy・scikit-learn・librosa・soundfile・matplotlib | 各プロジェクトの開発者（BSD / ISC 系） | `pip install -r requirements.txt` | PC での音声処理・評価・図の作成 |

μT-Kernel 3.0 と BSP はキットの利用条件により再配布できないため、リポジトリには含めていません。

**実機（micro:bit）で動くアプリケーション部分は、すべて応募者が作成しました。**
SAADC のドライバ、μT-Kernel のタスク構成、特徴量の抽出（FFT・メルフィルタ）、
CNN の推論は手書きの C で実装しており、推論ライブラリは使っていません。

上記の既存ソフトウェアは、それぞれのライセンスと利用条件に従って利用しており、
著作権などの権利処理を行っていることを保証します。

---

## 音声データを公開しない方針について

学習に使用した音声ファイルそのものは、以下の理由から本リポジトリに含めていません。

- 公開データセット（ESC-50 / UrbanSound8K）は **CC BY-NC** であり再配布に制約がある
- 日本のフリー効果音サイト由来の素材は、規約上**再配布が禁止**されているものがある
- 実機録音にはそれらを再生した音が含まれる

一方、**学習コード・特徴量抽出の実装・学習済みモデル・評価結果は全て公開**しており、
データセットは上記の出典から各自入手すれば学習を再現できます（手順は docs/BUILD.md）。
