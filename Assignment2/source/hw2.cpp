#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
// #include <curses.h>
#include <termios.h>
#include <fcntl.h>

#include <iostream>

#define ROW 10
#define COLUMN 50
#define LOG_NUM 9
struct Node
{
	int x, y;
	Node(int _x, int _y) : x(_x), y(_y){};
	Node(){};
} frog;

char map[ROW + 10][COLUMN];
Node logs_pos[LOG_NUM];

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void)
{
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if (ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void *logs_move(void *index)
{
	int a = *((int *)index);
	while (true)
	{
		usleep(50000);
		if (logs_pos[a].x % 2 == 0)
		{
			map[logs_pos[a].x][logs_pos[a].y] = ' ';
			map[logs_pos[a].x][(logs_pos[a].y + 5) % (COLUMN - 1)] = '=';
			logs_pos[a].y = (logs_pos[a].y + 1) % (COLUMN - 1);
		}
		else if ((logs_pos[a].x % 2) != 0)
		{
			map[logs_pos[a].x][(logs_pos[a].y + 48) % (COLUMN - 1)] = '=';
			map[logs_pos[a].x][(logs_pos[a].y + 4) % (COLUMN - 1)] = ' ';
			logs_pos[a].y = (logs_pos[a].y + 48) % (COLUMN - 1);
		}
	}

	/*  Move the logs  */

	/*  Check keyboard hits, to change frog's position or quit the game. */

	/*  Check game's status  */

	/*  Print the map on the screen  */
}

int main(int argc, char *argv[])
{

	// Initialize the river map and frog's starting position
	memset(map, 0, sizeof(map));
	int i, j;
	for (i = 1; i < ROW; ++i)
	{
		for (j = 0; j < COLUMN - 1; ++j)
			map[i][j] = ' ';
	}

	for (j = 0; j < COLUMN - 1; ++j)
		map[ROW][j] = map[0][j] = '|';

	for (j = 0; j < COLUMN - 1; ++j)
		map[0][j] = map[0][j] = '|';

	frog = Node(ROW, (COLUMN - 1) / 2);
	map[frog.x][frog.y] = '0';

	for (i = 0; i < LOG_NUM; ++i)
	{
		logs_pos[i] = Node(i + 1, 0);
		for (j = 0; j < 5; j++)
			map[logs_pos[i].x][logs_pos[i].y + j] = '=';
	}

	for (i = 0; i <= ROW; ++i)
		puts(map[i]);
	int indexs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	/*  Create pthreads for wood move and frog control.  */
	pthread_t frog_ctl, logs[LOG_NUM];
	for (i = 0; i < LOG_NUM; i++)
		pthread_create(&logs[i], NULL, logs_move, &indexs[i]);

	// pthread_create(&logs[i], NULL, logs_move, &indexs[0]);
	/*  Display the output for user: win, lose or quit.  */
	// Print the map into screen
	while (true)
	{
		usleep(50000);
		for (i = 0; i <= ROW; ++i)
			puts(map[i]);
	}

	return 0;
}
