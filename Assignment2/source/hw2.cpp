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
#define UPDATE_INTERVAL 70000

struct Node
{
	int x, y;
	Node(int _x, int _y) : x(_x), y(_y){};
	Node(){};
} frog, old_frog;

char map[ROW + 10][COLUMN];
Node logs_pos[LOG_NUM];
int logs_len[LOG_NUM] = {11, 14, 13, 19, 12, 18, 19, 14, 17};
int logs_init_pos[LOG_NUM] = {6, 23, 27, 14, 3, 19, 19, 20, 8};

int is_quit = 0;
int is_exist = 0;

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
		usleep(UPDATE_INTERVAL);
		int is_move = 0;

		if (logs_pos[a].x % 2 == 0)
		{

			map[logs_pos[a].x][logs_pos[a].y] = ' ';
			// map[logs_pos[a].x][(logs_pos[a].y + logs_len[a] - 1) % (COLUMN - 1)] = '=';
			if (logs_pos[a].y == frog.y && logs_pos[a].x == frog.x)
				is_move = 1;
			logs_pos[a].y = (logs_pos[a].y + 1) % (COLUMN - 1);
			for (int i = 0; i < logs_len[a]; i++)
			{
				if ((logs_pos[a].y + i) % (COLUMN - 1) == frog.y && logs_pos[a].x == frog.x)
					is_move = 1;
				map[logs_pos[a].x][(logs_pos[a].y + i) % (COLUMN - 1)] = '=';
			}
			if (is_move)
			{
				frog.y++;
			}
		}
		else if ((logs_pos[a].x % 2) != 0)
		{
			// map[logs_pos[a].x][(logs_pos[a].y + 48) % (COLUMN - 1)] = '=';
			map[logs_pos[a].x][(logs_pos[a].y + logs_len[a]) % (COLUMN - 1)] = ' ';
			if (logs_pos[a].y == frog.y && logs_pos[a].x == frog.x)
				is_move = 1;
			logs_pos[a].y = (logs_pos[a].y + 48) % (COLUMN - 1);
			for (int i = 0; i < logs_len[a]; i++)
			{
				if ((logs_pos[a].y + i) % (COLUMN - 1) == frog.y && logs_pos[a].x == frog.x)
					is_move = 1;
				map[logs_pos[a].x][(logs_pos[a].y + i) % (COLUMN - 1)] = '=';
			}
			if (is_move)
			{
				frog.y--;
			}
		}
		if (is_quit)
			return NULL;
	}

	/*  Move the logs  */

	/*  Check keyboard hits, to change frog's position or quit the game. */

	/*  Check game's status  */

	/*  Print the map on the screen  */
}
void *frog_move(void *)
{
	while (1)
	{
		// usleep(UPDATE_INTERVAL);
		if (kbhit())
		{
			char ch = (char)getchar();
			// switch (map[frog.x][frog.y])
			// {
			// case /* constant-expression */:
			// 	/* code */
			// 	break;

			// default:
			// 	break;
			// }
			switch (ch)
			{
			case 'w':
			case 'W':
				frog.x--;
				break;
			case 'a':
			case 'A':
				frog.y--;
				break;
			case 's':
			case 'S':
				frog.x++;
				break;
			case 'd':
			case 'D':
				frog.y++;
				break;
			case 'q':
			case 'Q':
				is_quit = 1;
				break;
			default:
				// if (frog.y < 0 || frog.y >= COLUMN - 1)
				// {
				// 	is_quit = 1;
				// 	break;
				// }
				// map[frog.x][frog.y] = '0';
				break;
			}
		}
		if (is_quit)
			return NULL;
	}
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
	{
		map[ROW][j] = '|';
		map[0][j] = '|';
	}

	frog = Node(ROW, (COLUMN - 1) / 2);
	old_frog = Node(ROW, (COLUMN - 1) / 2);
	map[frog.x][frog.y] = '0';

	for (i = 0; i < LOG_NUM; ++i)
	{
		logs_pos[i] = Node(i + 1, logs_init_pos[i]);
		for (j = 0; j < logs_len[i]; j++)
			map[logs_pos[i].x][logs_pos[i].y + j] = '=';
	}

	for (i = 0; i <= ROW; ++i)
		puts(map[i]);
	int indexs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

	/*  Create pthreads for wood move and frog control.  */
	pthread_t frog_ctl, logs[LOG_NUM];
	for (i = 0; i < LOG_NUM; i++)
		pthread_create(&logs[i], NULL, logs_move, &indexs[i]);

	pthread_create(&frog_ctl, NULL, frog_move, NULL);
	// pthread_create(&logs[i], NULL, logs_move, &indexs[0]);
	/*  Display the output for user: win, lose or quit.  */
	// Print the map into screen
	while (true)
	{
		usleep(UPDATE_INTERVAL);
		if (is_quit)
		{
			std::cout << "You exit the game." << std::endl;
			break;
		}

		for (j = 0; j < COLUMN - 1; ++j)
		{
			map[0][j] = '|';
			map[ROW][j] = '|';
		}

		if (frog.y < 0 || frog.y >= COLUMN - 1 || frog.x > ROW)
		{
			// is_quit = 1;
			std::cout << "You out of bound" << std::endl;
			break;
		}

		map[frog.x][frog.y] = '0';
		for (i = 0; i <= ROW; ++i)
			puts(map[i]);
		if (frog.x <= 0)
		{
			std::cout << "You win the game!" << std::endl;
			break;
		}
		if (frog.x != ROW)
		{
			int a = frog.x - 1;
			int pos = frog.y;
			int start = logs_pos[a].y;
			int end = (logs_pos[a].y + logs_len[a] - 1) % (COLUMN - 1);
			if ((logs_pos[a].y + logs_len[a] > 49 && pos < start && pos > end) || (logs_pos[a].y + logs_len[a] <= 49 && (pos > end || pos < start)))
			{
				std::cout << frog.y << "  " << logs_pos[frog.x - 1].y;
				std::cout << "You drop in river" << std::endl;
				break;
			}
		};
	};
	is_quit = 1;
	usleep(UPDATE_INTERVAL);
	usleep(UPDATE_INTERVAL);
	return 0;
}
