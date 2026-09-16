/*
 *  infer_mb.c : micro:bit向け 推論(特徴量+CNN)の float最適化版
 *
 *  predict.c(double, dense mel)はRAM 128KBに収まらないため:
 *   - 全てfloat化(M4のFPU活用・メモリ半減)
 *   - メルフィルタバンクをスパース化(64KB行列→約数KB)
 *   - CNNの中間バッファを2枚(bufA/bufB)で使い回す
 *  を行い、数値はKeras/predict.cと一致することをPCで検証する。
 *
 *  公開API: mb_classify(sig8000, probs3)
 *  PC検証: gcc -O2 -o infer_mb infer_mb.c -lm ; infer_mb in.txt out.txt
 */
#include <math.h>
#include "cnn_weights.h"      /* MU,SD,C1_W.. (float const), N_FEAT=960 */

#define SR        8000
#define N_FFT     512
#define HOP       256
#define N_BINS    257
#define N_MELS    32
#define NFRAMES   30
#define NCLASS    3
#define MEL_FMIN  0.0f
#define MEL_FMAX  4000.0f
#define MEL_EPS   1e-10f
#define PI_F      3.14159265358979f

/* --- レベル不変化(RMS正規化) : features_ref.py と同一定義 ---
 * 窓ごとに DC除去 → RMSを RMS_TARGET に揃えてから特徴量化する。
 * 入力の絶対音量が特徴量に影響しなくなるので、マイクゲインの手動較正が不要。
 * 実装上は「スケール係数を窓関数に畳み込む」ことで、正規化用の
 * 32KB floatバッファを作らずに済ませている。 */
#define RMS_TARGET  0.1f
#define RMS_FLOOR   1e-9f
/* 実機(SAADC生値)でこのRMS未満なら推論せず"その他"扱い。
 * 無音を増幅してノイズを誤判定するのを防ぐ。静かな室内でRMS約6なので
 * 3は「ほぼ完全な無音」だけを弾く保守的な値。 */
#define GATE_RMS    3.0f

/* ---------------- FFT (float, 基数2) ---------------- */
static void fft(float *re, float *im, int n)
{
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) { float t; t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; }
	}
	for (int len = 2; len <= n; len <<= 1) {
		float ang = -2.0f * PI_F / len, wr = cosf(ang), wi = sinf(ang);
		for (int i = 0; i < n; i += len) {
			float cwr = 1.0f, cwi = 0.0f;
			for (int k = 0; k < len/2; k++) {
				int a = i+k, b = i+k+len/2;
				float tr = re[b]*cwr - im[b]*cwi, ti = re[b]*cwi + im[b]*cwr;
				re[b]=re[a]-tr; im[b]=im[a]-ti; re[a]+=tr; im[a]+=ti;
				float nwr = cwr*wr - cwi*wi; cwi = cwr*wi + cwi*wr; cwr = nwr;
			}
		}
	}
}

/* ---------------- スパース・メルフィルタバンク ---------------- */
static int   m_start[N_MELS];
static int   m_len[N_MELS];
static int   m_off[N_MELS];
static float m_w[1024];
static float s_win[N_FFT];
static int   s_inited = 0;

static float hz2mel(float f) { return 2595.0f * log10f(1.0f + f / 700.0f); }
static float mel2hz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

static void infer_init(void)
{
	for (int i = 0; i < N_FFT; i++) s_win[i] = 0.5f - 0.5f * cosf(2.0f * PI_F * i / N_FFT);

	float binf[N_MELS + 2];
	float mlo = hz2mel(MEL_FMIN), mhi = hz2mel(MEL_FMAX);
	for (int i = 0; i < N_MELS + 2; i++) {
		float mel = mlo + (mhi - mlo) * i / (N_MELS + 1);
		binf[i] = mel2hz(mel) / ((float)SR / N_FFT);
	}
	int off = 0;
	for (int m = 0; m < N_MELS; m++) {
		float l = binf[m], c = binf[m+1], r = binf[m+2];
		int ks = -1, ke = -1;
		for (int k = 0; k < N_BINS; k++) {
			float w = 0;
			if (c > l && k >= l && k <= c) w = (k - l) / (c - l);
			else if (r > c && k > c && k <= r) w = (r - k) / (r - c);
			if (w > 0) { if (ks < 0) ks = k; ke = k; }
		}
		m_start[m] = (ks < 0) ? 0 : ks;
		m_len[m]   = (ks < 0) ? 0 : (ke - ks + 1);
		m_off[m]   = off;
		for (int k = m_start[m]; k < m_start[m] + m_len[m]; k++) {
			float w = 0;
			if (c > l && k >= l && k <= c) w = (k - l) / (c - l);
			else if (r > c && k > c && k <= r) w = (r - k) / (r - c);
			m_w[off++] = w;
		}
	}
	s_inited = 1;
}

/* DC(平均)を返す */
static float mean_f(const float *sig)
{
	double sum = 0;
	for (int i = 0; i < SR; i++) sum += sig[i];
	return (float)(sum / SR);
}

/* DC除去後のRMSを返す */
static float rms_f(const float *sig, float dc)
{
	double acc = 0;
	for (int i = 0; i < SR; i++) { double d = (double)sig[i] - dc; acc += d * d; }
	return (float)sqrt(acc / SR);
}

/* RMS→正規化スケール係数(無音時は1.0=増幅しない。Python版と同じ挙動) */
static float norm_scale(float rms)
{
	return (rms < RMS_FLOOR) ? 1.0f : (RMS_TARGET / rms);
}

/* sig[8000] -> out[960] (mel優先順, log mel)。DC除去+RMS正規化を内部で行う */
static void mb_feature(const float *sig, float *out)
{
	if (!s_inited) infer_init();
	static float re[N_FFT], im[N_FFT], pw[N_BINS];
	float dc = mean_f(sig);
	float k  = norm_scale(rms_f(sig, dc));
	for (int t = 0; t < NFRAMES; t++) {
		for (int i = 0; i < N_FFT; i++) { re[i] = (sig[t*HOP + i] - dc) * k * s_win[i]; im[i] = 0; }
		fft(re, im, N_FFT);
		for (int k = 0; k < N_BINS; k++) pw[k] = re[k]*re[k] + im[k]*im[k];
		for (int m = 0; m < N_MELS; m++) {
			float e = 0; int off = m_off[m], ks = m_start[m];
			for (int j = 0; j < m_len[m]; j++) e += m_w[off + j] * pw[ks + j];
			out[m*NFRAMES + t] = logf(e + MEL_EPS);
		}
	}
}

/* 実機用: int16生サンプルからDC除去+RMS正規化を窓掛けに畳んで特徴量に
 * (32KBのfloatバッファを作らずに済ませる) */
static void mb_feature_i16(const short *raw, float dc, float k, float *out)
{
	if (!s_inited) infer_init();
	static float re[N_FFT], im[N_FFT], pw[N_BINS];
	for (int t = 0; t < NFRAMES; t++) {
		for (int i = 0; i < N_FFT; i++) {
			re[i] = (((float)raw[t*HOP + i]) - dc) * k * s_win[i];
			im[i] = 0;
		}
		fft(re, im, N_FFT);
		for (int k = 0; k < N_BINS; k++) pw[k] = re[k]*re[k] + im[k]*im[k];
		for (int m = 0; m < N_MELS; m++) {
			float e = 0; int off = m_off[m], ks = m_start[m];
			for (int j = 0; j < m_len[m]; j++) e += m_w[off + j] * pw[ks + j];
			out[m*NFRAMES + t] = logf(e + MEL_EPS);
		}
	}
}

/* ---------------- CNN (float, バッファ使い回し) ---------------- */
static void conv_relu(const float *in, int H, int W, int IC,
                      const float *ker, const float *b, int OC, float *out)
{
	for (int oy = 0; oy < H; oy++)
		for (int ox = 0; ox < W; ox++)
			for (int oc = 0; oc < OC; oc++) {
				float a = b[oc];
				for (int ky = 0; ky < 3; ky++)
					for (int kx = 0; kx < 3; kx++) {
						int iy = oy+ky-1, ix = ox+kx-1;
						if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
						for (int ic = 0; ic < IC; ic++)
							a += in[(iy*W+ix)*IC+ic] * ker[(((ky*3+kx)*IC)+ic)*OC+oc];
					}
				out[(oy*W+ox)*OC+oc] = a > 0 ? a : 0;
			}
}
static void pool(const float *in, int H, int W, int C, float *out, int *OH, int *OW)
{
	*OH = H/2; *OW = W/2;
	for (int oy = 0; oy < *OH; oy++)
		for (int ox = 0; ox < *OW; ox++)
			for (int c = 0; c < C; c++) {
				float mx = -1e30f;
				for (int dy = 0; dy < 2; dy++)
					for (int dx = 0; dx < 2; dx++) {
						float v = in[((2*oy+dy)*W+(2*ox+dx))*C+c];
						if (v > mx) mx = v;
					}
				out[(oy*(*OW)+ox)*C+c] = mx;
			}
}
static void gap(const float *in, int H, int W, int C, float *out)
{
	for (int c = 0; c < C; c++) {
		float s = 0;
		for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) s += in[(y*W+x)*C+c];
		out[c] = s / (H*W);
	}
}
static void dense(const float *in, const float *w, const float *b, int IN, int OUT, int relu, float *out)
{
	for (int o = 0; o < OUT; o++) {
		float a = b[o];
		for (int i = 0; i < IN; i++) a += in[i] * w[i*OUT+o];
		out[o] = (relu && a < 0) ? 0 : a;
	}
}

static float s_bufA[N_MELS*NFRAMES*8];   /* 7680 = conv1出力が最大 */
static float s_bufB[16*15*8];            /* 1920 */
static float s_x[N_FEAT];

static float s_feat[N_FEAT];

/* feat[960] -> probs[3] (CNN本体・正規化込み) */
static void mb_cnn(const float *feat, float *probs)
{
	for (int i = 0; i < N_FEAT; i++) s_x[i] = (feat[i] - MU[i]) / SD[i];

	int h, w, h2, w2;
	conv_relu(s_x, 32, 30, 1, C1_W, C1_B, 8, s_bufA);      /* b1 32x30x8 */
	pool(s_bufA, 32, 30, 8, s_bufB, &h, &w);               /* p1 16x15x8 */
	conv_relu(s_bufB, h, w, 8, C2_W, C2_B, 16, s_bufA);    /* b2 16x15x16 */
	pool(s_bufA, h, w, 16, s_bufB, &h2, &w2);              /* p2 8x7x16 */
	conv_relu(s_bufB, h2, w2, 16, C3_W, C3_B, 32, s_bufA); /* b3 8x7x32 */
	float g[32];  gap(s_bufA, h2, w2, 32, g);
	float d1[32]; dense(g, D1_W, D1_B, 32, 32, 1, d1);
	float lg[NCLASS]; dense(d1, D2_W, D2_B, 32, NCLASS, 0, lg);
	float mx = lg[0]; for (int i = 1; i < NCLASS; i++) if (lg[i] > mx) mx = lg[i];
	float s = 0; for (int i = 0; i < NCLASS; i++) { probs[i] = expf(lg[i]-mx); s += probs[i]; }
	for (int i = 0; i < NCLASS; i++) probs[i] /= s;
}

/* sig[8000] float -> probs[3] (PC検証用) */
void mb_classify(const float *sig, float *probs)
{
	mb_feature(sig, s_feat);
	mb_cnn(s_feat, probs);
}

/* raw[8000] int16 -> probs[3] (実機用)
 * DC除去・RMS正規化は内部で実施するのでゲイン指定は不要。
 * 戻り値: 1=推論した / 0=音量がGATE_RMS未満のため推論せず"その他"を返した */
int mb_classify_i16(const short *raw, float *probs)
{
	long sum = 0;
	for (int i = 0; i < SR; i++) sum += raw[i];
	float dc = (float)sum / SR;

	double acc = 0;
	for (int i = 0; i < SR; i++) { double d = (double)raw[i] - dc; acc += d * d; }
	float rms = (float)sqrt(acc / SR);

	if (rms < GATE_RMS) {                 /* ほぼ無音: 増幅せず"その他"確定 */
		probs[0] = 0.0f; probs[1] = 0.0f; probs[2] = 1.0f;
		return 0;
	}
	mb_feature_i16(raw, dc, norm_scale(rms), s_feat);
	mb_cnn(s_feat, probs);
	return 1;
}

#ifndef FIRMWARE_BUILD
/* PC検証用 main */
#include <stdio.h>
int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: infer_mb in.txt out.txt\n"); return 1; }
	FILE *fi = fopen(argv[1], "r");
	if (!fi) return 1;
	static float sig[SR];
	int n = 0;
	while (n < SR && fscanf(fi, "%f", &sig[n]) == 1) n++;
	fclose(fi);
	float probs[NCLASS];
	mb_classify(sig, probs);
	FILE *fo = fopen(argv[2], "w");
	for (int i = 0; i < NCLASS; i++) fprintf(fo, "%.8e\n", probs[i]);
	fclose(fo);
	return 0;
}
#endif
