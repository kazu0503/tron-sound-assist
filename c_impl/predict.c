/* 音声(1秒8kHz) → メル特徴量 → 小型CNN → クラス判定 までC完結。
 * micro:bit に載る推論コードそのもの(マイク取り込み部分を除く)。
 * melc.c の特徴量抽出 + cnn_infer.c の推論を統合し、両方とも検証済み。
 *
 * 入力 : argv[1] 音声サンプル(1行1値, 8000行) ※ASCIIパス
 * 出力 : argv[2] に softmax確率3値(siren/car_horn/other)
 * ビルド: gcc -O2 -o predict predict.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cnn_weights.h"

#define SR 8000
#define N_FFT 512
#define HOP 256
#define N_MELS 32
#define N_BINS (N_FFT/2+1)
#define NFRAMES 30
#define FMIN 0.0
#define FMAX 4000.0
#define EPS 1e-10
#define PI 3.14159265358979323846
#define NCLASS 3

/* ---------- 特徴量(melc.cと同一) ---------- */
static void fft(double *re, double *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { double t; t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * PI / len, wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1.0, cwi = 0.0;
            for (int k = 0; k < len/2; k++) {
                int a = i+k, b = i+k+len/2;
                double tr = re[b]*cwr - im[b]*cwi, ti = re[b]*cwi + im[b]*cwr;
                re[b]=re[a]-tr; im[b]=im[a]-ti; re[a]+=tr; im[a]+=ti;
                double n2 = cwr*wr - cwi*wi; cwi = cwr*wi + cwi*wr; cwr = n2;
            }
        }
    }
}
static double hz2mel(double f){return 2595.0*log10(1.0+f/700.0);}
static double mel2hz(double m){return 700.0*(pow(10.0,m/2595.0)-1.0);}

/* 音声(8000) → raw特徴量(960, mel優先順) */
void feature(const double *sig, float *out960) {
    static double fb[N_MELS][N_BINS];
    double binf[N_MELS+2], mlo = hz2mel(FMIN), mhi = hz2mel(FMAX);
    for (int i=0;i<N_MELS+2;i++) binf[i]=mel2hz(mlo+(mhi-mlo)*i/(N_MELS+1))/((double)SR/N_FFT);
    for (int m=0;m<N_MELS;m++) for(int k=0;k<N_BINS;k++){
        double l=binf[m],c=binf[m+1],r=binf[m+2],v=0;
        if(c>l&&k>=l&&k<=c)v=(k-l)/(c-l); else if(r>c&&k>c&&k<=r)v=(r-k)/(r-c);
        fb[m][k]=v;
    }
    double win[N_FFT];
    for(int i=0;i<N_FFT;i++) win[i]=0.5-0.5*cos(2.0*PI*i/N_FFT);
    double re[N_FFT],im[N_FFT],pw[N_BINS];
    static double mel[N_MELS][NFRAMES];
    for(int t=0;t<NFRAMES;t++){
        for(int i=0;i<N_FFT;i++){re[i]=sig[t*HOP+i]*win[i];im[i]=0;}
        fft(re,im,N_FFT);
        for(int k=0;k<N_BINS;k++) pw[k]=re[k]*re[k]+im[k]*im[k];
        for(int m=0;m<N_MELS;m++){double e=0;for(int k=0;k<N_BINS;k++)e+=fb[m][k]*pw[k];mel[m][t]=log(e+EPS);}
    }
    for(int m=0;m<N_MELS;m++)for(int t=0;t<NFRAMES;t++) out960[m*NFRAMES+t]=(float)mel[m][t];
}

/* ---------- CNN(cnn_infer.cと同一) ---------- */
static void conv(const float*in,int H,int W,int IC,const float*ker,const float*b,int OC,float*out){
    for(int oy=0;oy<H;oy++)for(int ox=0;ox<W;ox++)for(int oc=0;oc<OC;oc++){
        float a=b[oc];
        for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++){
            int iy=oy+ky-1,ix=ox+kx-1; if(iy<0||iy>=H||ix<0||ix>=W)continue;
            for(int ic=0;ic<IC;ic++) a+=in[(iy*W+ix)*IC+ic]*ker[(((ky*3+kx)*IC)+ic)*OC+oc];
        }
        out[(oy*W+ox)*OC+oc]=a>0?a:0;
    }
}
static void pool(const float*in,int H,int W,int C,float*out,int*OH,int*OW){
    *OH=H/2;*OW=W/2;
    for(int oy=0;oy<*OH;oy++)for(int ox=0;ox<*OW;ox++)for(int c=0;c<C;c++){
        float mx=-1e30f;
        for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){float v=in[((2*oy+dy)*W+(2*ox+dx))*C+c];if(v>mx)mx=v;}
        out[(oy*(*OW)+ox)*C+c]=mx;
    }
}
static void gap(const float*in,int H,int W,int C,float*out){
    for(int c=0;c<C;c++){float s=0;for(int y=0;y<H;y++)for(int x=0;x<W;x++)s+=in[(y*W+x)*C+c];out[c]=s/(H*W);}
}
static void dense(const float*in,const float*w,const float*b,int IN,int OUT,int relu,float*out){
    for(int o=0;o<OUT;o++){float a=b[o];for(int i=0;i<IN;i++)a+=in[i]*w[i*OUT+o];out[o]=(relu&&a<0)?0:a;}
}
void predict(const float*raw,float*probs){
    static float x[N_MELS*NFRAMES];
    for(int i=0;i<N_FEAT;i++) x[i]=(raw[i]-MU[i])/SD[i];
    static float b1[32*30*8],p1[16*15*8],b2[16*15*16],p2[8*7*16],b3[8*7*32];
    int h,w,h2,w2; float g[32],d1[32],lg[NCLASS];
    conv(x,32,30,1,C1_W,C1_B,8,b1);
    pool(b1,32,30,8,p1,&h,&w);
    conv(p1,h,w,8,C2_W,C2_B,16,b2);
    pool(b2,h,w,16,p2,&h2,&w2);
    conv(p2,h2,w2,16,C3_W,C3_B,32,b3);
    gap(b3,h2,w2,32,g);
    dense(g,D1_W,D1_B,32,32,1,d1);
    dense(d1,D2_W,D2_B,32,NCLASS,0,lg);
    float mx=lg[0];for(int i=1;i<NCLASS;i++)if(lg[i]>mx)mx=lg[i];
    float s=0;for(int i=0;i<NCLASS;i++){probs[i]=expf(lg[i]-mx);s+=probs[i];}
    for(int i=0;i<NCLASS;i++)probs[i]/=s;
}

#ifndef FIRMWARE_BUILD   /* PC検証用main。実機ビルド時は FIRMWARE_BUILD を定義して除外 */
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: predict in.txt out.txt\n");return 1;}
    FILE*fi=fopen(argv[1],"r"); if(!fi){fprintf(stderr,"open in fail\n");return 1;}
    static double sig[SR]; int n=0;
    while(n<SR && fscanf(fi,"%lf",&sig[n])==1) n++;
    fclose(fi);
    static float raw[N_MELS*NFRAMES]; float probs[NCLASS];
    feature(sig,raw);
    predict(raw,probs);
    const char*names[]={"siren","car_horn","other"};
    int best=0; for(int i=1;i<NCLASS;i++) if(probs[i]>probs[best])best=i;
    FILE*fo=fopen(argv[2],"w");
    for(int i=0;i<NCLASS;i++) fprintf(fo,"%.8e\n",probs[i]);
    fclose(fo);
    fprintf(stderr,"判定: %s (siren=%.3f car_horn=%.3f other=%.3f)\n",
            names[best],probs[0],probs[1],probs[2]);
    return 0;
}
#endif /* FIRMWARE_BUILD */
