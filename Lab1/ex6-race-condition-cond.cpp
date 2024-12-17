/*******************************************************************
 * ex3456-race-condition.cpp
 * Demonstrates a race condition.
 *******************************************************************/

#include <cstdio>
#include <cstdlib>

#include <chrono>
#include <pthread.h> // include the pthread library
#include <unistd.h>

#define ADD_THREADS 4
#define SUB_THREADS 4

int global_counter = 0;

int add_done_count = 0;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;


void *add(void *threadid)
{
	long tid = *(long *)threadid;
	global_counter++;
	sleep(rand() % 2);
	printf("add thread #%ld incremented global_counter!\n", tid);

	add_done_count++;

	if (add_done_count == ADD_THREADS)
	{
		printf("All add threads completed, signalling cond var\n");
		pthread_cond_broadcast(&count_threshold_cv);
	}
	pthread_exit(NULL); // terminate thread
}

void *sub(void *threadid)
{
	long tid = *(long *)threadid;

	pthread_mutex_lock(&count_mutex);
	while(add_done_count < ADD_THREADS)
	{
		pthread_cond_wait(&count_threshold_cv, &count_mutex);
	}
	

	global_counter--;
	sleep(rand() % 2);
	printf("sub thread #%ld decremented global_counter! \n", tid);


	pthread_mutex_unlock(&count_mutex);
	pthread_exit(NULL); // terminate thread
}

int main(int argc, char *argv[])
{
	global_counter = 10;
	pthread_t add_threads[ADD_THREADS];
	pthread_t sub_threads[SUB_THREADS];
	long add_threadid[ADD_THREADS];
	long sub_threadid[SUB_THREADS];

	int rc;
	long t1, t2;
	for (t1 = 0; t1 < ADD_THREADS; t1++)
	{
		int tid = t1;
		add_threadid[tid] = tid;
		printf("main thread: creating add thread %d\n", tid);
		rc = pthread_create(&add_threads[tid], NULL, add,
							(void *)&add_threadid[tid]);
		if (rc)
		{
			printf("Return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	for (t2 = 0; t2 < SUB_THREADS; t2++)
	{
		int tid = t2;
		sub_threadid[tid] = tid;
		printf("main thread: creating sub thread %d\n", tid);
		rc = pthread_create(&sub_threads[tid], NULL, sub,
							(void *)&sub_threadid[tid]);
		if (rc)
		{
			printf("Return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}


	for(long i = 0; i < ADD_THREADS; i++)
	{
		pthread_join(add_threads[i], NULL); // This assumes add and sub threads are of the same number
		pthread_join(sub_threads[i], NULL);
	}


	printf("### global_counter final value = %d ###\n", global_counter);
	pthread_exit(NULL);
}
