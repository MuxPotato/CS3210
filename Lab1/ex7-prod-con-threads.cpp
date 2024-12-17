/*******************************************************************
 * ex789-prod-con-threads.cpp
 * Producer-consumer synchronisation problem in C++
 *******************************************************************/

#include <cstdio>
#include <cstdlib>
#include <pthread.h>

constexpr int PRODUCERS = 2;
constexpr int CONSUMERS = 1;

int producer_buffer[10];


int curr_size = 0;
int front_index = 0;
int back_index = 0;

// If we use one counter tgt for both producer and consumer, there is a situation where if the counter
// is too low, producer can increment it so fast that consumer waits for a signal that never comes from
// producer as producer thread has killed itself
int item_consumed = 0;
int item_produced = 0;


const int EXIT_AMOUNT_CONSUMER = 100000000;
const int EXIT_AMOUNT_PRODUCER = 100000000;


int consumer_sum = 0;

pthread_mutex_t mutex;
pthread_cond_t full_cv;

pthread_cond_t empty_cv;

void *producer(void *threadid)
{
	while(item_produced < EXIT_AMOUNT_PRODUCER)
	{
		// Write producer code here
		pthread_mutex_lock(&mutex);

		//if buffer full, wait
		while(curr_size == 10)
		{
			//printf("buffer full, waiting...\n");
			pthread_cond_wait(&full_cv, &mutex);
		}
		

		int num_to_add = 1 + (rand() % 11);


		producer_buffer[back_index] = num_to_add;
		back_index = (back_index + 1) % 10;



		curr_size++;
		item_produced++;

		//printf("Adding into buffer num: %d \n", num_to_add);
		//printf("exit count: %d\n", exit_counter );
		

		// Broadcast empty condition var, no side effects if no threads are waiting
		pthread_cond_broadcast(&empty_cv);

		pthread_mutex_unlock(&mutex);	
	}
	pthread_exit(NULL);
	
}

void *consumer(void *threadid)
{
	// Write consumer code here
	while(item_consumed < EXIT_AMOUNT_CONSUMER)
	{
		pthread_mutex_lock(&mutex);

		// if buffer is empty, wait
		while(curr_size == 0)
		{
			//printf("here in while loop ");
			//printf("buffer empty, waiting...\n");
			pthread_cond_wait(&empty_cv, &mutex);
		}


		int num_to_read = producer_buffer[front_index];
		front_index = (front_index + 1) % 10;

		consumer_sum += num_to_read;
		curr_size--;
		item_consumed++;

		//printf("Adding into consumer sum: %d \n", num_to_read);
		//printf("consumer sum: %d\n", consumer_sum );
		//printf("exit count: %d\n", exit_counter );

		// Broadcast Full condition var, no side effects if no threads are waiting
		pthread_cond_broadcast(&full_cv);

		pthread_mutex_unlock(&mutex);
	}
	pthread_exit(NULL);
}

int main(int argc, char *argv[])
{	
	clock_t start = clock();

	pthread_t producer_threads[PRODUCERS];
	pthread_t consumer_threads[CONSUMERS];
	int producer_threadid[PRODUCERS];
	int consumer_threadid[CONSUMERS];

	int rc;
	int t1, t2;
	for (t1 = 0; t1 < PRODUCERS; t1++)
	{
		int tid = t1;
		producer_threadid[tid] = tid;
		printf("Main: creating producer %d\n", tid);
		rc = pthread_create(&producer_threads[tid], NULL, producer,
							(void *)&producer_threadid[tid]);
		if (rc)
		{
			printf("Error: Return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	for (t2 = 0; t2 < CONSUMERS; t2++)
	{
		int tid = t2;
		consumer_threadid[tid] = tid;
		printf("Main: creating consumer %d\n", tid);
		rc = pthread_create(&consumer_threads[tid], NULL, consumer,
							(void *)&consumer_threadid[tid]);
		if (rc)
		{
			printf("Error: Return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}


	// Add pthread_join here
	pthread_join(producer_threads[0], NULL);
	pthread_join(producer_threads[1], NULL);
	pthread_join(consumer_threads[0], NULL);
	
	printf("Final Consumer Sum: %d\n", consumer_sum);
	printf("items produced: %d\n", item_produced);
	printf("items consumed: %d\n", item_consumed);
	clock_t end = clock();

	double duration = double(end - start) / CLOCKS_PER_SEC;
	printf("Time taken for script: %f \n", duration);

	pthread_exit(NULL);

	/*
					some tips for this exercise:

					1. you may want to handle SIGINT (ctrl-C) so that your program
									can exit cleanly (by killing all threads, or just calling
		 exit)

					1a. only one thread should handle the signal (POSIX does not define
									*which* thread gets the signal), so it's wise to mask out the
		 signal on the worker threads (producer and consumer) and let the main
		 thread handle it
	*/
}
