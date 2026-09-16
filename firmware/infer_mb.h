/* micro:bit向け推論API (infer_mb.c)
 *
 * 特徴量の前段で「DC除去 + RMS正規化」を行うため、入力の絶対音量に
 * 影響されない(マイクゲインの手動較正は不要)。
 */
#ifndef INFER_MB_H
#define INFER_MB_H

/* float音声[8000] -> probs[3] (PC検証用。内部でRMS正規化) */
void mb_classify(const float *sig, float *probs);

/* int16生サンプル[8000] -> probs[3] (実機用。DC除去・RMS正規化は内部)
 * 戻り値: 1=推論した / 0=音量不足(無音)のため推論せず"その他"を返した */
int mb_classify_i16(const short *raw, float *probs);

#endif
