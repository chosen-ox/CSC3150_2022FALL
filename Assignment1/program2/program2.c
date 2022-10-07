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

void my_wait(pid_t pid) {
    int status = 6;
    struct wait_opts wo;
    struct pid *wo_pid = NULL;
    enum pid_type type;
    type = PIDTYPE_PID;
    wo_pid = find_get_pid(pid);

    wo.wo_type = type;
    wo.wo_pid = wo_pid;
    wo.wo_flags = WEXITED| WUNTRACED;
    wo.wo_info = NULL;
    wo.wo_stat = status;
    wo.wo_rusage = NULL;
    printk("[program2] : look at me %d", wo.wo_stat);


    printk("[program2] : receive signal");
    int a;

    a = do_wait(&wo);
    printk("[program2] :do_wait return value is %d\n", a);

    printk("[program2] : The return signal is %d\n", wo.wo_stat);
    put_pid(wo_pid);

    return;
}

int my_exec(void) {
    int result;
    const char __user path[] = "/home/vagrant/CSC3150_2022FALL/Assignment1/program2/test";
    const char *const argv[] = {path, NULL, NULL};
    const char *const envp[] = {"HOME=/", "PATH=/sbin:/user/sbin:/bin:/usr/bin", NULL};

    struct filename *my_filename = getname_kernel(path);
    // printk("[program2] : here am i%s",my_filename->name);

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
            .stack = my_exec,
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
