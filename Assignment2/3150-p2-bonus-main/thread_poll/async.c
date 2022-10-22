
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"
#include <stdio.h>

my_queue_t task_queue;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
void async_init(int num_threads)
{

    pthread_t threads[num_threads];
    task_queue.size = 0;
    for (int i = 0; i < num_threads; i++)
    {
        pthread_create(&threads[i], NULL, wait_to_wakeup, NULL);
        // my_item_t item;
        // item.thread_id = threads[i];
        // DL_APPEND(thread_poll.head, (my_item_t *)&item);
        // thread_poll.size++;
    }

    // printf("hello, %d\n", thread_poll.size);
    return;
    /** TODO: create num_threads threads and initialize the thread pool **/
}

void async_run(void (*hanlder)(int), int args)
{
    my_item_t item = {
        .handler_ptr = hanlder,
        .args = args,
    };
    pthread_mutex_lock(&lock);
    DL_APPEND(task_queue.head, &item);
    task_queue.size++;
    pthread_cond_signal(&cond);
    printf("size increase: %d\n", task_queue.size);
    pthread_mutex_unlock(&lock);
    return;
    /** TODO: rewrite it to support thread pool **/
}
void *wait_to_wakeup(void *args)
{

    void (*handler)(int);
    int arg;
    for (;;)
    {
        // pthread_mutex_lock(&lock);

        printf("I got the lock\n");
        while (task_queue.size == 0)
        {
            printf("waiting\n");
            pthread_cond_wait(&cond, &lock);
        }

        printf("size delete:%d\n", task_queue.size);
        task_queue.size--;

        handler = task_queue.head->handler_ptr;
        arg = task_queue.head->args;
        // if (task_queue.head->prev == NULL)
        // {

        //     printf("fuck the null\n");
        // };

        printf("the args%d\n", task_queue.head->args);

        printf("size decrease:%d\n", task_queue.size);
        DL_DELETE(task_queue.head, task_queue.head);
        pthread_mutex_unlock(&lock);
        (*handler)(arg);
    }
}