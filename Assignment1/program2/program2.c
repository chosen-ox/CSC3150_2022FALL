#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/signal.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

struct wait_opts {
    enum pid_type wo_type;
    int wo_flags;
    struct pid *wo_pid;

    struct waitid_info *wo_info;
    int wo_stat;
    struct rusage *wo_rusage;

    wait_queue_entry_t child_wait;
    int notask_error;
};
static struct task_struct *task;
extern struct filename *getname_kernel(const char *filename);
extern pid_t kernel_clone(struct kernel_clone_args *args);
extern int do_execve(struct filename *filename,
                     const char __user *const __user *__argv,
                     const char __user *const __user *__envp);
extern long do_wait(struct wait_opts *wo);

void my_output(int signal) {
    switch (signal) {
        case 1:
            printk("[program2] : get SIGHUP signal\n");
            printk("[program2] : child process is hung up\n");
            printk("[program2] : The return signal is 1\n");
            break;
        case 2:
            printk("[program2] : get SIGINT signal\n");
            printk("[program2] : terminal interrupt\n");
            printk("[program2] : The return signal is 2\n");
            break;
        case 131:
            printk("[program2] : get SIGQUIT signal\n");
            printk("[program2] : terminal quit\n");
            printk("[program2] : The return signal is 3\n");
            break;
        case 132:
            printk("[program2] : get SIGILL signal\n");
            printk("[program2] : child process has illegal instruction error\n");
            printk("[program2] : The return signal is 4\n");
            break;
        case 133:
            printk("[program2] : get SIGTRAP signal\n");
            printk("[program2] : child process has trap error\n");
            printk("[program2] : The return signal is 5\n");
            break;
        case 134:
            printk("[program2] : get SIGABRT signal\n");
            printk("[program2] : child process has abort error\n");
            printk("[program2] : The return signal is 6\n");
            break;
        case 135:
            printk("[program2] : get SIGBUS signal\n");
            printk("[program2] : child process has bus error\n");
            printk("[program2] : The return signal is 7\n");
            break;
        case 136:
            printk("[program2] : get SIGFPE signal\n");
            printk("[program2] : child process has float error\n");
            printk("[program2] : The return signal is 8\n");
            break;
        case 9:
            printk("[program2] : get SIGKILL signal\n");
            printk("[program2] : child process killed\n");
            printk("[program2] : The return signal is 9\n");
            break;
        case 139:
            printk("[program2] : get SIGSEGV signal\n");
            printk("[program2] : child process has segmentation fault error\n");
            printk("[program2] : The return signal is 11\n");
            break;
        case 13:
            printk("[program2] : get SIGPIPE signal\n");
            printk("[program2] : child process has pipe error\n");
            printk("[program2] : The return signal is 13\n");
            break;
        case 14:
            printk("[program2] : get SIGALARM signal\n");
            printk("[program2] : child process has alarm error\n");
            printk("[program2] : The return signal is 14\n");
            break;
        case 15:
            printk("[program2] : get SIGTERM signal\n");
            printk("[program2] : child process terminated\n");
            printk("[program2] : The return signal is 15\n");
            break;
        case 0:
            printk("[program2] : child process exit normally\n");
            printk("[program2] : The return signal is 0\n");
            break;
        case 4991:
            printk("[program2] : child process stop\n");
            printk("[program2] : The return signal is 19\n");
            break;
    }
    return;
}

void my_wait(pid_t pid) {
    int a;
    struct wait_opts wo;
    struct pid *wo_pid = NULL;
    enum pid_type type;
    type = PIDTYPE_PID;
    wo_pid = find_get_pid(pid);

    wo.wo_type = type;
    wo.wo_pid = wo_pid;
    wo.wo_flags = WEXITED| WUNTRACED;
    wo.wo_info = NULL;
    wo.wo_rusage = NULL;
    printk("[program2] : look at me %d", wo.wo_stat);


    printk("[program2] : receive signal");

    a = do_wait(&wo);
    printk("[program2] :do_wait return value is %d\n", a);

    my_output(wo.wo_stat);
    put_pid(wo_pid);

    return;
}


int my_exec(void) {
    int result;
    const char __user path[] = "/home/vagrant/CSC3150_2022FALL/Assignment1/program2/test";

    /* execute a test program in child process */
    printk("[program2] : child process");

    result = do_execve(getname_kernel(path), NULL, NULL);

    if (!result) {
        return 0;
    } else {
        do_exit(result);
    }
}
//implement fork function
int my_fork(void *argc) {

    struct kernel_clone_args args = {
            .exit_signal = SIGCHLD,
            .stack = (unsigned long) &my_exec,
            .stack_size = 0,
            .parent_tid = NULL,
            .child_tid = NULL,
            .tls = 0,
    };


    //set default sigaction for current process
    pid_t pid;
    int i;
    struct k_sigaction *k_action = &current->sighand->action[0];

    // const char *const argv[]={path, NULL};

    // struct filename * result;


    for (i = 0; i < _NSIG; i++) {
        k_action->sa.sa_handler = SIG_DFL;
        k_action->sa.sa_flags = 0;
        k_action->sa.sa_restorer = NULL;
        sigemptyset(&k_action->sa.sa_mask);
        k_action++;
    }

    /* fork a process using kernel_clone or kernel_thread */
    pid = kernel_clone(&args);

    if (pid == -1) {
        printk("fork failed");
        return -1;
    }
    printk("[program2] : I am parent my pid is %d\n", (int) current->pid);

    printk("[program2] : I am child my pid is %d\n", pid);


    /* execute a test program in child process */

    /* wait until child process terminates */
    my_wait(pid);

    return 0;
}


static int __init program2_init(void) {

    printk("[program2] : Module_init Jiakun Fan\n");

    printk("[program2] : Module_init create kthread start");

    task = kthread_create(&my_fork, NULL, "MyThread");

    if (!IS_ERR(task)) {
        printk("Kthread starts\n");
        wake_up_process(task);
    }
    return 0;

    /* create a kernel thread to run my_fork */

    return 0;
}

static void __exit program2_exit(void) {
    printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
