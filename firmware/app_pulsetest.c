/*
 *  振動モーター パルス長テスト (本番の振動パターン長を決めるための計測用)
 *
 *  本番アプリでは ALERT とLEDは毎回出るのに、振動は最初の1回しか起きなかった。
 *  一方、1秒ON/1秒OFFのテストでは毎回振動する。
 *  → 本番の振動パターンのON時間(0.2〜0.5秒)が短すぎて、3.3Vではモーターが
 *    回り出す前にOFFになっている可能性がある(モジュールの起動電圧は仕様3.7V)。
 *
 *  そこでON時間を変えながら順番に鳴らし、何ms以上なら確実に回り出すかを調べる。
 *  各パルスの直前に長さを表示し、パルス間は2秒止めるので、体感と表示を対応づけられる。
 *
 *  LED列(P3など)は使わない。
 */
#include <tk/tkernel.h>
#include <tm/tmonitor.h>

#define MOTOR_PIN   2

static const W pulse_ms[] = { 100, 200, 300, 400, 500, 700, 1000 };
#define N_PULSE  (sizeof(pulse_ms) / sizeof(pulse_ms[0]))

static void motor(int on)
{
	if (on) out_w(GPIO(P0, OUTSET), 1U << MOTOR_PIN);
	else    out_w(GPIO(P0, OUTCLR), 1U << MOTOR_PIN);
}

EXPORT INT usermain(void)
{
	W round = 0;
	UW i;

	out_w(GPIO(P0, OUTCLR), 1U << MOTOR_PIN);
	out_w(GPIO(P0, PIN_CNF(MOTOR_PIN)), 1);

	tm_putstring((UB*)"\n=== Motor pulse length test ===\n");
	tm_putstring((UB*)"Feel the motor. Each line is printed right before the pulse.\n");

	for (;;) {
		round++;
		tm_printf((UB*)"\n--- round %d ---\n", round);
		for (i = 0; i < N_PULSE; i++) {
			tm_printf((UB*)"ON %d ms\n", pulse_ms[i]);
			motor(1);
			tk_dly_tsk(pulse_ms[i]);
			motor(0);
			tk_dly_tsk(2000);
		}
	}
}
