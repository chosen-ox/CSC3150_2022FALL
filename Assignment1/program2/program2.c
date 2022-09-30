#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

static struct task_struct *task;
static struct kernel_clone_args args = {
        .exit_signal = SIGCHLD,
};

extern struct filename *getname(const char __user *);
extern pid_t kernel_clone(struct kernel_clone_args *args);
//implement fork function
int my_fork(void *argc){


    //set default sigaction for current process
    pid_t pid;
    int i;
    struct k_sigaction *k_action = &current->sighand->action[0];
    char path[] = "/etc/test";

    const char *const argv[]={path, NULL};

    struct filename * result;


    for(i=0;i<_NSIG;i++){
        k_action->sa.sa_handler = SIG_DFL;
        k_action->sa.sa_flags = 0;
        k_action->sa.sa_restorer = NULL;
        sigemptyset(&k_action->sa.sa_mask);
        k_action++;
    }

    /* fork a process using kernel_clone or kernel_thread */
    pid = kernel_clone(&args);

    if (pid==-1) {
        printk("fork failed");
        return -1;
    }

    if (pid != 0 ) {
        printk("I am parent\n");
    }
    else {
        printk("I am child\n");
        result = getname(path);
        do_execve(path, argv, NULL);
    }



    /* execute a test program in child process */

    /* wait until child process terminates */

    return 0;
}


static int __init program2_init(void){

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

static void __exit program2_exit(void){
    printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);