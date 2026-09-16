/*
 *  振動モーター配線テスト (組み立て確認用・本番アプリとは別)
 *
 *  1) 短絡チェック: リング0(P0.02)の隣の細いピン P3(P0.31=LED列3) / P4(P0.28=LED列1)
 *     と接触していないかを電気的に調べる。隣接ピンは弱いプルダウン付き入力にし、
 *     P0だけをHIGHにしたとき隣接ピンがHIGHに引きずられたら短絡と判定する。
 *     本番アプリと違いLED列を出力にしないので、仮に短絡していてもピン同士が
 *     ぶつからず安全に確認できる。
 *  2) 電源電圧: SAADCでVDD(=3V端子の電圧)を測り、モーターON時の電圧降下も表示する。
 *     モジュールの起動電圧は仕様上3.7Vのため、実際に回るかの判断材料にする。
 *  3) 振動: 短絡が無ければ P0 を 1秒ON / 1秒OFF で繰り返す。
 */
#include <tk/tkernel.h>
#include <tm/tmonitor.h>

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
#define PSEL_VDD            9

#define MOTOR_PIN   2       /* リング0 */
#define NEIGH_P3    31      /* 細いピンP3 = LED列3 */
#define NEIGH_P4    28      /* 細いピンP4 = LED列1 */

static short vdd_buf[1];

static int pin_in(int pin)
{
	return (in_w(GPIO(P0, IN)) >> pin) & 1;
}

static void motor(int on)
{
	if (on) out_w(GPIO(P0, OUTSET), 1U << MOTOR_PIN);
	else    out_w(GPIO(P0, OUTCLR), 1U << MOTOR_PIN);
}

/* VDD を mV で返す (GAIN 1/6, 内部基準0.6V → フルスケール3.6V, 12bit) */
static W read_vdd_mv(void)
{
	W to = 0;
	out_w(SAADC_ENABLE, 0);
	out_w(SAADC_RESOLUTION, 2);
	out_w(SAADC_CH0_PSELP, PSEL_VDD);
	out_w(SAADC_CH0_PSELN, 0);
	out_w(SAADC_CH0_CONFIG, 0x20000);          /* GAIN1/6, TACQ 10us */
	out_w(SAADC_SAMPLERATE, 0);                /* タスク駆動(1回だけ) */
	out_w(SAADC_ENABLE, 1);
	out_w(SAADC_RESULT_PTR, (UW)vdd_buf);
	out_w(SAADC_RESULT_MAXCNT, 1);
	out_w(SAADC_EVENTS_END, 0);
	out_w(SAADC_TASKS_START, 1);
	out_w(SAADC_TASKS_SAMPLE, 1);
	while (in_w(SAADC_EVENTS_END) == 0) { if (++to > 1000000) break; }
	out_w(SAADC_EVENTS_END, 0);
	out_w(SAADC_TASKS_STOP, 1);
	if (vdd_buf[0] < 0) return 0;
	return (W)vdd_buf[0] * 3600 / 4096;
}

EXPORT INT usermain(void)
{
	int short_found = 0;
	int i, p3, p4;
	W mv;

	tm_putstring((UB*)"\n=== Motor wiring test ===\n");

	/* モーター出力は必ずLOWから始める */
	out_w(GPIO(P0, OUTCLR), 1U << MOTOR_PIN);
	out_w(GPIO(P0, PIN_CNF(MOTOR_PIN)), 1);    /* 出力 */

	/* 隣接ピンはプルダウン付き入力(出力にはしない＝短絡していても安全) */
	out_w(GPIO(P0, PIN_CNF(NEIGH_P3)), 0x04);
	out_w(GPIO(P0, PIN_CNF(NEIGH_P4)), 0x04);

	/* ---- 1) 短絡チェック (5回) ---- */
	for (i = 0; i < 5; i++) {
		motor(1);
		tk_dly_tsk(30);
		p3 = pin_in(NEIGH_P3);
		p4 = pin_in(NEIGH_P4);
		motor(0);
		tk_dly_tsk(30);
		if (p3 || p4) short_found = 1;
		tm_printf((UB*)"check %d: P0=HIGH -> P3=%d P4=%d\n", i + 1, p3, p4);
	}

	if (short_found) {
		motor(0);
		tm_putstring((UB*)"!! SHORT: ring0 touches a neighbor pin. Unplug USB and fix.\n");
		for (;;) tk_dly_tsk(1000);
	}
	tm_putstring((UB*)"OK: no short between ring0 and P3/P4\n");

	/* ---- 2)3) 電源電圧と振動 ---- */
	for (;;) {
		mv = read_vdd_mv();
		tm_printf((UB*)"motor OFF  VDD=%d mV\n", mv);
		motor(1);
		tk_dly_tsk(500);
		mv = read_vdd_mv();                     /* 回っている最中の電圧 */
		tm_printf((UB*)"motor ON   VDD=%d mV\n", mv);
		tk_dly_tsk(500);
		motor(0);
		tk_dly_tsk(1000);
	}
}
