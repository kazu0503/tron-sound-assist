/* 小型CNN推論の C 実装 (cnn_embedded.keras と一致)
 *
 * 構成(学習時と同一):
 *   入力 32(mel)×30(frame)×1
 *   Conv2D(8,3x3,same,relu) -> MaxPool2x2 -> Conv2D(16,same,relu) -> MaxPool2x2
 *   -> Conv2D(32,same,relu) -> GlobalAveragePooling -> Dense(32,relu) -> Dense(3,softmax)
 *
 * 入力 : argv[1] のファイル(raw特徴量を1行1値, 960×Kサンプル分)
 * 出力 : argv[2] に各サンプルのsoftmax確率3値(計3×K)
 * raw特徴量は内部で (x-MU)/SD 正規化してから推論する。
 *
 * ビルド: gcc -O2 -o cnn_infer cnn_infer.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cnn_weights.h"

#define H0 32
#define W0 30
#define NCLASS 3

/* Conv2D 3x3 same + ReLU. in:[H][W][IC], ker:[3][3][IC][OC](flat), out:[H][W][OC] */
static void conv_same_relu(const float *in, int H, int W, int IC,
                           const float *ker, const float *bias, int OC, float *out) {
    for (int oy = 0; oy < H; oy++)
        for (int ox = 0; ox < W; ox++)
            for (int oc = 0; oc < OC; oc++) {
                float acc = bias[oc];
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++) {
                        int iy = oy + ky - 1, ix = ox + kx - 1;
                        if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                        for (int ic = 0; ic < IC; ic++)
                            acc += in[(iy * W + ix) * IC + ic] *
                                   ker[(((ky * 3 + kx) * IC) + ic) * OC + oc];
                    }
                out[(oy * W + ox) * OC + oc] = acc > 0.0f ? acc : 0.0f;
            }
}

/* MaxPool 2x2 valid. out dims = H/2, W/2 */
static void maxpool(const float *in, int H, int W, int Cc, float *out, int *OH, int *OW) {
    *OH = H / 2; *OW = W / 2;
    for (int oy = 0; oy < *OH; oy++)
        for (int ox = 0; ox < *OW; ox++)
            for (int c = 0; c < Cc; c++) {
                float mx = -1e30f;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        float v = in[((2 * oy + dy) * W + (2 * ox + dx)) * Cc + c];
                        if (v > mx) mx = v;
                    }
                out[(oy * (*OW) + ox) * Cc + c] = mx;
            }
}

/* Global Average Pooling: out[c] = mean over H*W */
static void gap(const float *in, int H, int W, int Cc, float *out) {
    for (int c = 0; c < Cc; c++) {
        float s = 0.0f;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                s += in[(y * W + x) * Cc + c];
        out[c] = s / (H * W);
    }
}

/* Dense. W:[IN][OUT](flat). relu指定でReLU */
static void dense(const float *in, const float *w, const float *b,
                  int IN, int OUT, int relu, float *out) {
    for (int o = 0; o < OUT; o++) {
        float acc = b[o];
        for (int i = 0; i < IN; i++) acc += in[i] * w[i * OUT + o];
        out[o] = (relu && acc < 0.0f) ? 0.0f : acc;
    }
}

static void forward(const float *raw, float *probs) {
    static float x[H0 * W0];            /* 正規化後 [32][30][1] */
    for (int i = 0; i < N_FEAT; i++) x[i] = (raw[i] - MU[i]) / SD[i];

    static float b1[H0 * W0 * 8];
    conv_same_relu(x, H0, W0, 1, C1_W, C1_B, 8, b1);
    static float p1[16 * 15 * 8]; int h, w;
    maxpool(b1, H0, W0, 8, p1, &h, &w);                 /* 16x15 */
    static float b2[16 * 15 * 16];
    conv_same_relu(p1, h, w, 8, C2_W, C2_B, 16, b2);
    static float p2[8 * 7 * 16]; int h2, w2;
    maxpool(b2, h, w, 16, p2, &h2, &w2);                /* 8x7 */
    static float b3[8 * 7 * 32];
    conv_same_relu(p2, h2, w2, 16, C3_W, C3_B, 32, b3);
    float g[32];
    gap(b3, h2, w2, 32, g);
    float d1[32];
    dense(g, D1_W, D1_B, 32, 32, 1, d1);
    float logits[NCLASS];
    dense(d1, D2_W, D2_B, 32, NCLASS, 0, logits);
    /* softmax */
    float mx = logits[0];
    for (int i = 1; i < NCLASS; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < NCLASS; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
    for (int i = 0; i < NCLASS; i++) probs[i] /= sum;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: cnn_infer in.txt out.txt\n"); return 1; }
    FILE *fi = fopen(argv[1], "r");
    if (!fi) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    FILE *fo = fopen(argv[2], "w");
    if (!fo) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }

    float *raw = malloc(sizeof(float) * N_FEAT);
    float probs[NCLASS];
    int got;
    while (1) {
        got = 0;
        for (int i = 0; i < N_FEAT; i++) { if (fscanf(fi, "%f", &raw[i]) == 1) got++; else break; }
        if (got < N_FEAT) break;
        forward(raw, probs);
        for (int i = 0; i < NCLASS; i++) fprintf(fo, "%.8e\n", probs[i]);
    }
    free(raw); fclose(fi); fclose(fo);
    return 0;
}
