
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"
#include <stdio.h>

my_queue_t *task_queue;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
void async_init(int num_threads)
{
    pthread_t threads[num_threads];
    task_queue = (my_queue_t *)malloc(sizeof(my_queue_t));
    task_queue->head = NULL;
    task_queue->size = 0;
    for (int i = 0; i < num_threads; i++)
    {
        pthread_create(&threads[i], NULL, wait_to_wakeup, NULL);
    }
    return;
}

void async_run(void (*hanlder)(int), int args)
{
    my_item_t *item_ptr = (my_item_t *)malloc(sizeof(my_item_t));
    item_ptr->args = args;
    item_ptr->handler_ptr = hanlder;

    pthread_mutex_lock(&lock);
    if (task_queue->head != NULL)
    {
        item_ptr->prev = task_queue->head;
        task_queue->head->next = item_ptr;
        task_queue->head = task_queue->head->next;
        (task_queue->head)->next = NULL;
    }
    else
    {
        (task_queue->head) = (item_ptr);
        (task_queue->head)->prev = (task_queue->head);
        (task_queue->head)->next = NULL;
    }

    task_queue->size++;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&lock);
    return;
}
void *wait_to_wakeup(void *args)
{

    int arg;
    my_item_t *item_ptr;
    void (*handler)(int);
    for (;;)
    {
        pthread_mutex_lock(&lock);
        while (task_queue->size == 0)
            pthread_cond_wait(&cond, &lock);

        task_queue->size--;
        handler = task_queue->head->handler_ptr;
        arg = task_queue->head->args;

        item_ptr = task_queue->head;

        if ((task_queue->head)->prev == (task_queue->head))
        {
            (task_queue->head) = NULL;
        }
        else
        {
            task_queue->head = task_queue->head->prev;
            task_queue->head->next = NULL;
        };

        pthread_mutex_unlock(&lock);

        (*handler)(arg);
        free(item_ptr);
    }
}