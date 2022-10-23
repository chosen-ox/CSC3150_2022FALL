
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"
#include <stdio.h>

#include <assert.h>
my_queue_t *task_queue;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
int threads = 0;
void async_init(int num_threads)
{
    if (pthread_mutex_init(&lock, NULL) != 0)
        printf("gg\n");

    if (pthread_cond_init(&cond, NULL) != 0)
        printf("gg\n");
    exit;
    pthread_t threads[num_threads];
    task_queue = (my_queue_t *)malloc(sizeof(my_queue_t));

    task_queue->head = NULL;
    task_queue->size = 0;
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
    my_item_t *item_ptr = (my_item_t *)malloc(sizeof(my_item_t));
    item_ptr->args = args;
    item_ptr->handler_ptr = hanlder;
    pthread_mutex_lock(&lock);

    // DL_APPEND(task_queue->head, item_ptr);
    if (task_queue->head != NULL)
    {
        // (item_ptr)->prev = (task_queue->head)->prev;
        item_ptr->prev = task_queue->head;
        assert(task_queue->head != NULL);

        task_queue->head->next = item_ptr;
        task_queue->head = task_queue->head->next;
        // (task_queue->head)->prev->next = (item_ptr);
        // (task_queue->head)->prev = (item_ptr);
        (task_queue->head)->next = NULL;
    }
    else
    {
        (task_queue->head) = (item_ptr);
        (task_queue->head)->prev = (task_queue->head);
        (task_queue->head)->next = NULL;
    }

    task_queue->size += 1;
    assert(task_queue->size >= 0);
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&lock);
    return;
    /** TODO: rewrite it to support thread pool **/
}
void *wait_to_wakeup(void *args)
{

    int arg;
    my_item_t *item_ptr;
    void (*handler)(int);
    for (;;)
    {
        threads++;
        pthread_mutex_lock(&lock);
        //
        printf("I got the lock\n");
        while (task_queue->size == 0)
        {
            printf("waiting\n");
            pthread_cond_wait(&cond, &lock);
        }
        assert(task_queue->head != NULL);

        // printf("size before delete:%d\n", task_queue->size);
        // printf("size before delete111:%d\n", task_queue->size);
        assert(task_queue->size > 0);
        task_queue->size -= 1;
        // printf("size decrease111:%d\n", task_queue->size);

        handler = task_queue->head->handler_ptr;
        arg = task_queue->head->args;
        threads--;

        // printf("size decrease:%d\n", task_queue->size);
        item_ptr = task_queue->head;
        // DL_DELETE(task_queue->head, task_queue->head);
        assert((task_queue->head)->prev != NULL);
        if ((task_queue->head)->prev == (task_queue->head))
        {
            assert(task_queue->size == 0);
            (task_queue->head) = NULL;
        }
        else
        {
            task_queue->head = task_queue->head->prev;
            task_queue->head->next = NULL;
        };

        // (*handler)(arg);

        pthread_mutex_unlock(&lock);
        (*handler)(arg);
        // free(item_ptr);
    }
}