#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
void output_info(int status)
{
	switch (status) {
	case 1:
		printf("child process get SIGUP signal");
		break;
	case 2:
		printf("child process get SIGINT signal");
		break;
	case 131:
		printf("child process get SIGQUIT signal");
		break;
	case 132:
		printf("child process get SIGILL signal");
		break;
	case 133:
		printf("child process get SIGTRAP signal");
		break;
	case 134:
		printf("child process get SIGABRT signal");
		break;
	case 135:
		printf("child process get SIGBUS signal");
		break;
	case 136:
		printf("child process get SIGFPE signal");
		break;
	case 9:
		printf("child process get SIGKILL signal");
		break;
	case 139:
		printf("child process get SIGSEGV signal");
		break;
	case 13:
		printf("child process get SIGPIPE signal");
		break;
	case 14:
		printf("child process get SIGALRM signal");
		break;
	case 15:
		printf("child process get SIGTERM signal");
		break;
	case 0:
		printf("Normal termination with EXIT STATUS = 0");
		break;
	case 4991:
		printf("child process get SIGSTOP signal");
		break;
	default:
		printf("[program2] : a signal not contained in the signal list\n");
		printf("[program2] : The return signal is %d", status);
		break;
	}
	printf("\n");
	return;
}

int main(int argc, char *argv[])
{
	int status;
	pid_t pid;
	/* fork a child process */
	printf("Process start to fork\n");
	pid = fork();
	char *arg[argc];
	for (int i = 0; i < argc - 1; i++) {
		arg[i] = argv[i + 1];
	}
	arg[argc - 1] = NULL;
	if (pid == -1) {
		perror("fork failed");
		exit(1);
	}
	if (pid != 0) {
		printf("I am the parent ");
		printf("my pid: %d\n", getpid());
		/* wait for child process terminates */
		waitpid(pid, &status, WUNTRACED);
		printf("Parent process receives SIGCHLD signal\n");
		/* check child process'  termination status */
		output_info(status);
	} else {
		printf("I am the child ");
		printf("my pid: %d\n", getpid());
		printf("Child process start to execute test program:\n");

		/* execute test program */
		execve(arg[0], arg, NULL);
	}
	return 0;
}
