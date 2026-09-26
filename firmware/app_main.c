/*
 *  環境音シーン認識・安全アシスト : RTOSタスク分割版 (M2)
 *
 *  μT-Kernel 3.0 の優先度付きマルチタスクで、録音・推論・通知を分離する。
 *
 *    T_audio (優先度2) 内蔵マイクをSAADC+EasyDMAの「ピンポンバッファ」で
 *                      連続サンプリング。片面が埋まるたびイベントフラグで通知。
 *                      録音に切れ目が出ないことを保証する(取りこぼし厳禁)。
 *    T_infer (優先度4) 埋まった面を特徴量→CNN推論。重い処理だが優先度を下げて
 *                      あるので、推論中もT_audioが割り込んで録音を続けられる。
 *                      結果はメッセージバッファでT_notifyへ渡す。
 *    T_notify(優先度3) 後処理(確信度しきい値＋連続N回一致)を通し、危険音と
 *                      確定したときだけLEDを点灯する。単発の誤判定では光らない。
 *
 *  この「録音を止めずに推論する」構造こそRTOSを使う必然性であり、
 *  同時に短い音の取りこぼしと単発誤検出の対策にもなっている。
 */
#include <tk/tkernel.h>
#include <tm/tmonitor.h>
#include "infer_mb.h"

/* ---------------- nRF52833 SAADC ---------------- */
#define SAADC_TASKS_START   0x40007000
#define SAADC_TASKS_SAMPLE  0x40007004
#define SAADC_TASKS_STOP    0x40007008
#define SAADC_EVENTS_STARTED 0x40007100
#define SAADC_EVENTS_END    0x40007104
#define SAADC_ENABLE        0x40007500
#define SAADC_CH0_PSELP     0x40007510
#define SAADC_CH0_PSELN     0x40007514
#define SAADC_CH0_CONFIG    0x40007518
#define SAADC_RESOLUTION    0x400075F0
#define SAADC_SAMPLERATE    0x400075F8
#define SAADC_RESULT_PTR    0x4000762C
#define SAADC_RESULT_MAXCNT 0x40007630

#define AIN3      4
#define NSAMP     8000              /* 1秒 = モデルの判定窓 */
/* 録音モードファームと同一にすること(学習データと同じ録音条件にするため) */
#define SAADC_CONFIG_VAL  (0x50000 | 0x600)   /* GAIN x2 / TACQ 40us */

/* ---------------- 通知の後処理パラメータ ---------------- */
#define CONF_TH     600             /* 確信度しきい値 (/1000) */
#define AGREE_N     2               /* 連続何回一致したら通知するか */
/* クラクションは1秒以内に鳴り終わる短い音で、2回連続の判定がほぼ起きない。
 * そのため確信度がこの値以上なら1回で通知する(実測: はっきり鳴らした
 * クラクション=987、静音時の単発誤判定=500前後)。 */
#define HORN_SINGLE_TH  700
#define HOLD_MS     3000            /* 通知を保持する時間 */

/* ---------------- micro:bit v2 LEDマトリクス ---------------- */
/* 行をHIGH・列をLOWにするとそのLEDが点く。1行まるごと点けるので
 * ダイナミック点灯(多重化)は不要＝T_notifyが単純に保てる。
 * 列は P0 にある4本だけを使う(残る1列は P1.05。P0.00 は32.768kHz水晶の
 * 入力ピンなので絶対に触らないこと)。1行あたり4個点けば視認には十分。 */
#define N_LED_ROW  5
#define N_LED_COL  4
static const unsigned char led_rows[N_LED_ROW] = { 21, 22, 15, 24, 19 };  /* P0 */
static const unsigned char led_cols[N_LED_COL] = { 28, 11, 31, 30 };      /* P0 */

/* ---------------- 振動モーター ----------------
 * エッジコネクタの「P0」端子 = nRF52833 の P0.02。
 * マイク(P0.05)ともLED(P0.21等)とも衝突しない。
 * ※必ずドライバ(トランジスタ)内蔵モジュールを使うこと。GPIOは1本5mAしか
 *   流せないため、振動モーター(60〜100mA)の直結はマイコンを壊す。
 * モジュール未接続でもこのピンを叩くだけなので動作に影響はない。 */
#define MOTOR_PIN   2

/* 100msごとの ON/OFF パターン。クラスごとに振動を変えて区別できるようにする */
static const unsigned char VIB_SIREN[] = { 1,1,1,1,1, 0,0, 1,1,1,1,1, 0,0 }; /* 長く2回 */
static const unsigned char VIB_HORN[]  = { 1,1, 0,0, 1,1, 0,0, 1,1, 0,0 };   /* 短く3回 */

/* ---------------- バッファ・カーネルオブジェクト ---------------- */
static short buf[2][NSAMP];         /* ピンポン: 16KB×2 */
static volatile int ready_idx = -1; /* 埋まった面の番号 */

static ID flg_audio;                /* 録音完了イベント */
static ID mbf_result;               /* 推論結果 T_infer -> T_notify */
#define FLG_FILLED  (1U << 0)

typedef struct {
	W cls;                      /* 0=siren 1=car_horn 2=other */
	W conf;                     /* 確信度 (/1000) */
	W ms;                       /* 推論所要時間(ms) */
} RESULT;

static const char *names[3] = { "SIREN", "CAR_HORN", "other" };

/* ================= LED ================= */
static void led_setup(void)
{
	int i;
	for (i = 0; i < N_LED_ROW; i++) out_w(GPIO(P0, PIN_CNF(led_rows[i])), 1);
	for (i = 0; i < N_LED_COL; i++) out_w(GPIO(P0, PIN_CNF(led_cols[i])), 1);
	/* 全消灯: 行LOW・列HIGH */
	for (i = 0; i < N_LED_ROW; i++) out_w(GPIO(P0, OUTCLR), 1U << led_rows[i]);
	for (i = 0; i < N_LED_COL; i++) out_w(GPIO(P0, OUTSET), 1U << led_cols[i]);
}

/* row行を丸ごと点灯(row<0で全消灯) */
static void led_row(int row)
{
	int i;
	for (i = 0; i < N_LED_ROW; i++) out_w(GPIO(P0, OUTCLR), 1U << led_rows[i]);
	if (row < 0) {
		for (i = 0; i < N_LED_COL; i++) out_w(GPIO(P0, OUTSET), 1U << led_cols[i]);
		return;
	}
	for (i = 0; i < N_LED_COL; i++) out_w(GPIO(P0, OUTCLR), 1U << led_cols[i]);
	out_w(GPIO(P0, OUTSET), 1U << led_rows[row]);
}

/* ================= 振動モーター ================= */
static void motor_setup(void)
{
	out_w(GPIO(P0, PIN_CNF(MOTOR_PIN)), 1);        /* 出力 */
	out_w(GPIO(P0, OUTCLR), 1U << MOTOR_PIN);      /* 停止 */
}

static void motor(int on)
{
	if (on) out_w(GPIO(P0, OUTSET), 1U << MOTOR_PIN);
	else    out_w(GPIO(P0, OUTCLR), 1U << MOTOR_PIN);
}

/* ================= T_audio ================= */
static void mic_setup(void)
{
	out_w(GPIO(P0, PIN_CNF(20)), 1);
	out_w(GPIO(P0, OUTSET), 1 << 20);           /* マイク電源ON */
	out_w(SAADC_RESOLUTION, 2);                 /* 12bit */
	out_w(SAADC_CH0_PSELP, AIN3);
	out_w(SAADC_CH0_PSELN, 0);
	out_w(SAADC_CH0_CONFIG, SAADC_CONFIG_VAL);
	out_w(SAADC_SAMPLERATE, (1 << 12) | 2000);  /* 内蔵タイマ 8kHz */
	out_w(SAADC_ENABLE, 1);
}

static void audio_task(INT stacd, void *exinf)
{
	int fill = 0;               /* いまDMAが書いている面 */
	W to;

	mic_setup();

	/* 1面目を開始し、STARTED後に「次の面」を予約しておく(EasyDMAの
	 * ダブルバッファ。これにより END の直後に切れ目なく継続できる) */
	out_w(SAADC_RESULT_PTR, (UW)buf[fill]);
	out_w(SAADC_RESULT_MAXCNT, NSAMP);
	out_w(SAADC_EVENTS_STARTED, 0);
	out_w(SAADC_EVENTS_END, 0);
	out_w(SAADC_TASKS_START, 1);
	to = 0;
	while (in_w(SAADC_EVENTS_STARTED) == 0) { if (++to > 1000000) break; }
	out_w(SAADC_EVENTS_STARTED, 0);
	out_w(SAADC_RESULT_PTR, (UW)buf[1 - fill]);   /* 次の面を予約 */
	out_w(SAADC_TASKS_SAMPLE, 1);                 /* 内蔵タイマ開始 */

	for (;;) {
		/* 1面が埋まるのを待つ。tk_dly_tskで待つ間はT_inferが走れる */
		while (in_w(SAADC_EVENTS_END) == 0) tk_dly_tsk(2);
		out_w(SAADC_EVENTS_END, 0);

		/* 予約済みの面へ即座に継続(ここが切れ目ゼロの肝) */
		out_w(SAADC_TASKS_START, 1);
		out_w(SAADC_TASKS_SAMPLE, 1);

		ready_idx = fill;                     /* 埋まった面を渡す */
		tk_set_flg(flg_audio, FLG_FILLED);
		fill = 1 - fill;

		/* さらに次の面を予約 */
		to = 0;
		while (in_w(SAADC_EVENTS_STARTED) == 0) { if (++to > 1000000) break; }
		out_w(SAADC_EVENTS_STARTED, 0);
		out_w(SAADC_RESULT_PTR, (UW)buf[1 - fill]);
	}
}

/* ================= T_infer ================= */
static void infer_task(INT stacd, void *exinf)
{
	float probs[3];
	UINT ptn;
	RESULT r;
	SYSTIM t0, t1;
	W i, best, idx;

	for (;;) {
		tk_wai_flg(flg_audio, FLG_FILLED, TWF_ORW | TWF_CLR, &ptn, TMO_FEVR);
		idx = ready_idx;
		if (idx < 0) continue;

		tk_get_otm(&t0);
		mb_classify_i16(buf[idx], probs);      /* ゲイン不要(内部でRMS正規化) */
		tk_get_otm(&t1);

		best = 0;
		for (i = 1; i < 3; i++) if (probs[i] > probs[best]) best = i;
		r.cls  = best;
		r.conf = (W)(probs[best] * 1000);
		r.ms   = (W)(t1.lo - t0.lo);
		tk_snd_mbf(mbf_result, &r, sizeof(r), TMO_POL);   /* 詰まっても待たない */
	}
}

/* ================= T_notify ================= */
/* 通知するかどうか(音の種類ごとに条件を変える)
 *   サイレン     : 鳴り続ける音なので、確信度600以上が2回連続したら
 *   クラクション : 短い音なので、確信度700以上なら1回で。
 *                  それ未満でも600以上が2回連続したら */
static int should_alert(const RESULT *r, W run)
{
	if (r->cls == 0) return r->conf >= CONF_TH && run >= AGREE_N;
	if (r->cls == 1) return r->conf >= HORN_SINGLE_TH || (r->conf >= CONF_TH && run >= AGREE_N);
	return 0;
}

static void notify_task(INT stacd, void *exinf)
{
	RESULT r;
	INT sz;
	W last = 2, run = 0;        /* 直前クラスと連続一致回数 */
	W alarm = -1, hold = 0;
	const unsigned char *vib = 0;
	W vlen = 0, vpos = 0;

	for (;;) {
		sz = tk_rcv_mbf(mbf_result, &r, 100);   /* 100ms待ち */
		if (sz == sizeof(r)) {
			/* --- 後処理: 音の種類ごとの条件を満たしたら通知 --- */
			int alert;
			if (r.cls == last) run++; else { last = r.cls; run = 1; }
			alert = should_alert(&r, run);

			tm_printf((UB*)"=> %s conf=%d (%dms) run=%d%s\n",
				  names[r.cls], r.conf, r.ms, run,
				  alert ? (UB*)"  *** ALERT ***" : (UB*)"");

			if (alert) {
				alarm = r.cls;
				hold = HOLD_MS;
				led_row(r.cls == 0 ? 0 : 2);    /* siren=上段 horn=中段 */
				/* 同じ警報が続いている間は途中で頭出ししない(毎秒リセットすると
				 * 「長く2回」が最後まで再生されず区別しにくくなるため)。
				 * パターンを鳴らし終えたか、クラスが変わったときだけ頭から再生。 */
				if (vib != (r.cls == 0 ? VIB_SIREN : VIB_HORN) || vpos >= vlen) {
					if (r.cls == 0) { vib = VIB_SIREN; vlen = sizeof(VIB_SIREN); }
					else            { vib = VIB_HORN;  vlen = sizeof(VIB_HORN); }
					vpos = 0;
				}
			}
		}

		/* 振動パターンを100msごとに1コマ進める(鳴らし終えたら停止) */
		if (vib && vpos < vlen) motor(vib[vpos++]);
		else                    motor(0);

		/* 通知の保持と自動解除 */
		if (alarm >= 0) {
			hold -= 100;
			if (hold <= 0) { alarm = -1; led_row(-1); vib = 0; motor(0); }
		}
	}
}

/* ================= 起動 ================= */
EXPORT INT usermain(void)
{
	T_CFLG cflg;
	T_CMBF cmbf;
	T_CTSK ctsk;
	ID tid;

	tm_putstring((UB*)"\n=== Sound Assist (RTOS tasks) ===\n");
	led_setup();
	led_row(-1);
	motor_setup();

	cflg.flgatr = TA_TFIFO | TA_WMUL;
	cflg.iflgptn = 0;
	flg_audio = tk_cre_flg(&cflg);
	if (flg_audio < 0) { tm_printf((UB*)"!! tk_cre_flg failed %d\n", (W)flg_audio); return 0; }

	cmbf.mbfatr = TA_TFIFO;
	cmbf.bufsz  = sizeof(RESULT) * 4;
	cmbf.maxmsz = sizeof(RESULT);
	mbf_result = tk_cre_mbf(&cmbf);
	if (mbf_result < 0) { tm_printf((UB*)"!! tk_cre_mbf failed %d\n", (W)mbf_result); return 0; }

	ctsk.tskatr = TA_HLNG | TA_RNG0;

	ctsk.task = audio_task;  ctsk.itskpri = 2; ctsk.stksz = 1024;
	tid = tk_cre_tsk(&ctsk);
	if (tid < 0) { tm_printf((UB*)"!! cre audio %d\n", (W)tid); return 0; }
	tk_sta_tsk(tid, 0);

	ctsk.task = notify_task; ctsk.itskpri = 3; ctsk.stksz = 1024;
	tid = tk_cre_tsk(&ctsk);
	if (tid < 0) { tm_printf((UB*)"!! cre notify %d\n", (W)tid); return 0; }
	tk_sta_tsk(tid, 0);

	ctsk.task = infer_task;  ctsk.itskpri = 4; ctsk.stksz = 2048;
	tid = tk_cre_tsk(&ctsk);
	if (tid < 0) { tm_printf((UB*)"!! cre infer %d\n", (W)tid); return 0; }
	tk_sta_tsk(tid, 0);

	tm_putstring((UB*)"tasks started: audio(2) notify(3) infer(4)\n");

	/* 初期タスク(優先度1)は眠り続ける。
	 * usermainがreturnするとカーネルがshutdownしてしまうため。 */
	tk_slp_tsk(TMO_FEVR);
	return 0;
}
