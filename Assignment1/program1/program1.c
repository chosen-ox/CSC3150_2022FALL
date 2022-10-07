#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int status;
    pid_t pid;
    /* fork a child process */
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
    if (pid != 0)//返回子进程
    {
        printf("I am the parent ");
        printf("my pid: %d\n", getpid());
        /* wait for child process terminates */
        waitpid(pid, &status, WUNTRACED);
        printf("Parent process receives the signal\n");
        /* check child process'  termination status */
        if (WIFEXITED(status)) {
            printf("Normal termination with EXIT STATUS = %d\n", status);
        } else if (WIFSIGNALED((status))) {
            printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
        } else if (WIFSTOPPED(status)) {
            printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
        } else {
            exit(0);
        }
    } else {
        printf("I am the child ");
        printf("my pid: %d\n", getpid());
        printf("Child process start to execute test program:\n");

        /* execute test program */
        execve(arg[0], arg, NULL);
    }
    return 0;
}
