/* 組み込み版メル特徴量の C 実装 (features_ref.py と1対1対応)
 *
 * micro:bit に載せる前に、PC(gcc)で計算してPython参照と数値一致するか検証する。
 * 検証は double で行い「アルゴリズムの移植が正しい」ことを確認する。
 * (実機では float でも可。FFTは512点・基数2。)
 *
 * 入力 : argv[1] のファイル(1行1サンプル) ※ASCIIパスで渡すこと(日本語パス回避)
 * 出力 : argv[2] のファイルに 32(mel)×30(frame)=960値(mel優先順)
 *
 * ビルド: gcc -O2 -o melc melc.c -lm
 * 実行 : melc.exe in.txt out.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SR     8000
#define N_FFT  512
#define HOP    256
#define N_MELS 32
#define N_BINS (N_FFT/2 + 1)
#define FMIN   0.0
#define FMAX   4000.0
#define EPS    1e-10
#define PI     3.14159265358979323846

/* ---- 基数2 FFT (反復・ビット反転) : re[],im[] を破壊的に変換 ---- */
static void fft(double *re, double *im, int n) {
    /* ビット反転並べ替え */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * PI / len;        /* 順変換 */
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1.0, cwi = 0.0;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                double tr = re[b] * cwr - im[b] * cwi;
                double ti = re[b] * cwi + im[b] * cwr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
                double ncwr = cwr * wr - cwi * wi;
                cwi = cwr * wi + cwi * wr; cwr = ncwr;
            }
        }
    }
}

static double hz_to_mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel_to_hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

/* メルフィルタバンク fb[N_MELS][N_BINS] を生成 (Python版と同一式) */
static void make_filterbank(double fb[N_MELS][N_BINS]) {
    double binf[N_MELS + 2];
    double mlo = hz_to_mel(FMIN), mhi = hz_to_mel(FMAX);
    for (int i = 0; i < N_MELS + 2; i++) {
        double mel = mlo + (mhi - mlo) * i / (N_MELS + 1);
        binf[i] = mel_to_hz(mel) / ((double)SR / N_FFT);
    }
    for (int m = 0; m < N_MELS; m++)
        for (int k = 0; k < N_BINS; k++) {
            double l = binf[m], c = binf[m + 1], r = binf[m + 2];
            double v = 0.0;
            if (c > l && k >= l && k <= c) v = (k - l) / (c - l);
            else if (r > c && k > c && k <= r) v = (r - k) / (r - c);
            fb[m][k] = v;
        }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: melc in.txt out.txt\n"); return 1; }
    FILE *fi = fopen(argv[1], "r");
    if (!fi) { fprintf(stderr, "cannot open input %s\n", argv[1]); return 1; }
    double *sig = malloc(sizeof(double) * SR * 2);
    int n = 0;
    while (n < SR * 2 && fscanf(fi, "%lf", &sig[n]) == 1) n++;
    fclose(fi);

    static double fb[N_MELS][N_BINS];
    make_filterbank(fb);

    double win[N_FFT];
    for (int i = 0; i < N_FFT; i++) win[i] = 0.5 - 0.5 * cos(2.0 * PI * i / N_FFT);

    int nframes = (n - N_FFT) / HOP + 1;
    double re[N_FFT], im[N_FFT], power[N_BINS];
    static double out[N_MELS][64];     /* out[m][t] に格納 */

    for (int t = 0; t < nframes; t++) {
        for (int i = 0; i < N_FFT; i++) {
            re[i] = sig[t * HOP + i] * win[i];
            im[i] = 0.0;
        }
        fft(re, im, N_FFT);
        for (int k = 0; k < N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
        for (int m = 0; m < N_MELS; m++) {
            double e = 0.0;
            for (int k = 0; k < N_BINS; k++) e += fb[m][k] * power[k];
            out[m][t] = log(e + EPS);
        }
    }
    /* Python の out.flatten() と同じ mel優先順(m外側, t内側)で出力 */
    FILE *fo = fopen(argv[2], "w");
    if (!fo) { fprintf(stderr, "cannot open output %s\n", argv[2]); return 1; }
    for (int m = 0; m < N_MELS; m++)
        for (int t = 0; t < nframes; t++)
            fprintf(fo, "%.10f\n", out[m][t]);
    fclose(fo);

    free(sig);
    return 0;
}
