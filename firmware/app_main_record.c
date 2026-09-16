/*
 *  環境音安全アシスト : 録音モードファーム (実機マイク較正用データ収集)
 *
 *  目的: 推論モデルが「データセット音源」で学習されているため、実機マイクの
 *        サイレンが CAR_HORN に誤判定される(ドメインギャップ)。
 *        実機マイクで本番の音を録音し、そのデータで再学習して較正する。
 *
 *  動作:
 *    - 常時: 0.1秒ぶんの音量(peak-to-peak)を "level pp=NN" としてシリアル出力。
 *      スマホで音を鳴らしながら数値が上がるのを見て、上がった瞬間に録音できる。
 *    - ボタンA(P0.14)を押すと 3秒(24000サンプル, 8kHz) を取り込み、生サンプル
 *      (int16)を #BEGIN..#END で囲んでダンプ。PC側 record_serial.py が WAV 保存。
 *
 *  SAADC: ゲイン×2・取得時間40us(スマホ再生音をしっかり拾うため。推論側も同じ
 *         設定に合わせてから再学習する)。
 *
 *  ※推論用appは app_main_infer.bak、推論本体は infer_mb.c.bak に退避済み。
 *    推論に戻すときは app_main_infer.bak→app_main.c、infer_mb.c.bak→infer_mb.c。
 */
#include <tk/tkernel.h>
#include <tm/tmonitor.h>

/* nRF52833 SAADC レジスタ */
#define SAADC_TASKS_START   0x40007000
#define SAADC_TASKS_SAMPLE  0x40007004
#define SAADC_TASKS_STOP    0x40007008
#define SAADC_EVENTS_END    0x40007104
#define SAADC_ENABLE        0x40007500
#define SAADC_CH0_PSELP     0x40007510
#define SAADC_CH0_PSELN     0x40007514
#define SAADC_CH0_CONFIG    0x40007518
#define SAADC_RESOLUTION    0x400075F0
#define SAADC_SAMPLERATE    0x400075F8
#define SAADC_RESULT_PTR    0x4000762C
#define SAADC_RESULT_MAXCNT 0x40007630

#define AIN3        4
#define BTN_A       14            /* micro:bit v2 ボタンA = P0.14 (押下=0) */
#define BTN_B       23            /* micro:bit v2 ボタンB = P0.23 (押下=0) */
#define SR          8000
#define REC_SEC     1             /* 1秒 = モデルの判定窓とちょうど同じ */
#define NSAMP       (SR * REC_SEC) /* 8000 */
#define NMON        1000           /* 音量メーター用(0.125秒) */

/*
 * 自動録音(オートトリガ):
 *   クラクションのような短い音はボタン操作とタイミングを合わせにくいので、
 *   音量が TRIG_PP を超えたら自動で1秒録音する。鳴らすだけで録れる。
 *   同じ音を二重に録らないよう、一度 REARM_PP を下回るまで再発火しない。
 *   ボタンB で オン/オフ を切り替え(静かな環境音"other"を録るときは切る)。
 */
#define TRIG_PP     150
#define REARM_PP    80

/*
 * CH0_CONFIG:
 *   GAIN  bits[10:8] = 6  → ×2         (0x600)
 *   TACQ  bits[18:16]= 5  → 40us       (0x50000)
 *   その他0 = 単極性/内部基準0.6V
 */
#define SAADC_CONFIG_VAL  (0x50000 | 0x600)

static short audio[NSAMP];         /* 48KB 取り込みバッファ */
static short mon[NMON];            /* 2KB メーター用 */

/* マイク・SAADC初期化 (8kHz内蔵タイマでDMA連続サンプリング) */
static void mic_setup(void)
{
	out_w(GPIO(P0, PIN_CNF(20)), 1);           /* P0.20 出力 */
	out_w(GPIO(P0, OUTSET), 1 << 20);          /* マイク電源ON */
	out_w(SAADC_RESOLUTION, 2);                /* 12bit */
	out_w(SAADC_CH0_PSELP, AIN3);
	out_w(SAADC_CH0_PSELN, 0);
	out_w(SAADC_CH0_CONFIG, SAADC_CONFIG_VAL); /* ×2 / 40us */
	out_w(SAADC_SAMPLERATE, (1 << 12) | 2000); /* 内蔵タイマ 8kHz */
	out_w(SAADC_ENABLE, 1);
}

/* ボタンA を入力(プルアップ)に設定 */
static void btn_setup(void)
{
	out_w(GPIO(P0, PIN_CNF(BTN_A)), 0x0C);     /* DIR=in, PULL=pullup(3<<2) */
	out_w(GPIO(P0, PIN_CNF(BTN_B)), 0x0C);
}

/* buf に n サンプル取り込む(ms = DMA充填待ち時間) */
static void capture_n(short *buf, W n, W ms)
{
	W to;
	out_w(SAADC_RESULT_PTR, (UW)buf);
	out_w(SAADC_RESULT_MAXCNT, n);
	out_w(SAADC_EVENTS_END, 0);
	out_w(SAADC_TASKS_START, 1);
	out_w(SAADC_TASKS_SAMPLE, 1);              /* 起動(以降は内蔵タイマで自動) */
	tk_dly_tsk(ms);
	to = 0;
	while (in_w(SAADC_EVENTS_END) == 0) { if (++to > 4000000) break; }
	out_w(SAADC_EVENTS_END, 0);
	out_w(SAADC_TASKS_STOP, 1);
}

/* peak-to-peak (DC差を見る簡易音量) */
static W peak_to_peak(short *buf, W n)
{
	short mn = buf[0], mx = buf[0];
	W i;
	for (i = 1; i < n; i++) {
		if (buf[i] < mn) mn = buf[i];
		if (buf[i] > mx) mx = buf[i];
	}
	return (W)(mx - mn);
}

static int btn_pressed(W pin)
{
	return (in_w(GPIO(P0, IN)) & (1 << pin)) == 0;
}

/* 1秒録音してシリアルへダンプ */
static void record_and_dump(void)
{
	W i;
	tm_putstring((UB*)"REC...\n");
	capture_n(audio, NSAMP, REC_SEC * 1000 + 100);
	tm_printf((UB*)"#BEGIN sr=%d n=%d\n", (W)SR, (W)NSAMP);
	for (i = 0; i < NSAMP; i++) {
		tm_printf((UB*)"%d\n", (W)audio[i]);
	}
	tm_putstring((UB*)"#END\n");
	tm_putstring((UB*)"--- saved. ---\n");
}

EXPORT INT usermain(void)
{
	W pp;
	int auto_on = 1;     /* オートトリガ既定ON */
	int armed = 1;       /* 発火可能か(音が静まると再装填) */

	tm_putstring((UB*)"\n=== Recorder ready (gain x2, 1s) ===\n");
	tm_putstring((UB*)"A=record  B=auto on/off  (auto: sound over threshold records itself)\n");
	mic_setup();
	btn_setup();

	for (;;) {
		/* 音量メーター(0.125秒) */
		capture_n(mon, NMON, 160);
		pp = peak_to_peak(mon, NMON);
		tm_printf((UB*)"level pp=%d\n", pp);

		/* ボタンB: オートトリガ切替 */
		if (btn_pressed(BTN_B)) {
			tk_dly_tsk(30);
			if (btn_pressed(BTN_B)) {
				auto_on = !auto_on;
				tm_printf((UB*)"=== auto trigger %s ===\n",
					  auto_on ? (UB*)"ON" : (UB*)"OFF");
				while (btn_pressed(BTN_B)) tk_dly_tsk(20);
			}
		}

		/* ボタンA: 手動録音 */
		if (btn_pressed(BTN_A)) {
			tk_dly_tsk(30);
			if (btn_pressed(BTN_A)) {
				record_and_dump();
				while (btn_pressed(BTN_A)) tk_dly_tsk(20);
				armed = 0;   /* 直後の自動発火を防ぐ */
			}
			continue;
		}

		/* オートトリガ: しきい値超えで自動録音 */
		if (auto_on) {
			if (armed && pp >= TRIG_PP) {
				record_and_dump();
				armed = 0;
			} else if (!armed && pp < REARM_PP) {
				armed = 1;   /* 静まったので次の音を待てる状態へ */
			}
		}
	}
}
