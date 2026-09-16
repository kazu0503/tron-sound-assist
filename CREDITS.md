# クレジット表記（使用した音源素材）

本プロジェクトの学習に使用した外部音源の出典。
CC BY 等で表記が必要な素材はここに記載する。GitHub公開時はこのファイルを掲載する。

## OtoLogic（CC BY 4.0 — 表記必須）
効果音: **OtoLogic**（https://otologic.jp/）

使用素材（フリー効果音「乗り物」より、CC BY 4.0）:
- 救急車サイレン: Ambulance-Siren01 / 02 / 03（Close/Far/Dry 各種）
- 消防車サイレン: Fire_Truck-Siren01（Close/Far/Dry 各種）
- パトカーサイレン: Police_Car-Siren03（Close/Far/Dry 各種）
- 出典ページ: https://otologic.jp/free/se/transportation01.html , https://otologic.jp/free/se/transportation02.html

※ ライセンス: Creative Commons Attribution 4.0 International (CC BY 4.0)。
　クレジット表記により再配布・改変が可能。

## その他のフリー効果音（利用無料・生ファイルは再配布しない＝学習専用）
- 救急車サイレン.mp3（日本のフリー効果音サイト由来。効果音ラボ / DOVA-SYNDROME 等）
  ※ 生音声はリポジトリに含めない（.gitignore済み）。学習にのみ使用。
  ※ 正確な出典が分かれば追記する。

## 公開データセット
- **ESC-50** (https://github.com/karolpiczak/ESC-50) — CC BY-NC 3.0。学習に使用。
  K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015.
- **UrbanSound8K** (https://urbansounddataset.weebly.com/urbansound8k.html) — CC BY-NC 3.0。
  siren 929 / car_horn 429 クリップ、および非ターゲット8クラスから400クリップを "other" として使用。
  J. Salamon, C. Jacoby, J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", ACM Multimedia 2014.

## 自己録音データ（応募者が作成）
- `data/device/` … micro:bit v2 の内蔵マイクで応募者が録音した 67ファイル / 91秒。
  実機のマイク特性に合わせるため学習に使用（siren 9 / car_horn 11 / other 47）。
  録音内容には上記フリー音源をスピーカーで再生したものが含まれるため、
  再配布の可否を厳密に判断できない。よって**リポジトリには含めない**（.gitignore済み）。

## ソフトウェア
- **μT-Kernel 3.0** および micro:bit 向け BSP
  … 「IoTエッジノード実践キット/micro:bit」（パーソナルメディア株式会社）付属のものを使用。
  再配布不可のためリポジトリには含めていない。

---

## 音声データを公開しない方針について

学習に使用した音声ファイルそのものは、以下の理由から本リポジトリに含めていません。

- 公開データセット（ESC-50 / UrbanSound8K）は **CC BY-NC** であり再配布に制約がある
- 日本のフリー効果音サイト由来の素材は、規約上**再配布が禁止**されているものがある
- 実機録音にはそれらを再生した音が含まれる

一方、**学習コード・特徴量抽出の実装・学習済みモデル・評価結果は全て公開**しており、
データセットは上記の出典から各自入手すれば学習を再現できます（手順は docs/BUILD.md）。
