/*******************************************************************
 * ex789-prod-con-threads.cpp
 * Producer-consumer synchronisation problem in C++
 *******************************************************************/

#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <semaphore.h>
#include <string>

#include <ctime>

constexpr int PRODUCERS = 2;
constexpr int CONSUMERS = 1;

int item_consumed = 0;

const int EXIT_AMOUNT_CONSUMER = 100000000;
const int EXIT_AMOUNT_PRODUCER = 100000000;
int consumer_sum = 0;



void producer(sem_t *sem_mutex, sem_t *sem_empty, sem_t *sem_full, int *producer_buffer, int *consumer_sum, int *producer_count, int *front_index, int *back_index)
{	
	printf("created 1 producer \n");
	while(*producer_count < EXIT_AMOUNT_PRODUCER)
	{	
		sem_wait(sem_full);
		sem_wait(sem_mutex);


		int num_to_add = 1 + (rand() % 11);


		producer_buffer[*back_index] = num_to_add;
		*back_index = (*back_index + 1) % 10; 
	

		(*producer_count)++;

		//printf("Adding into buffer num: %d \n", num_to_add);
		

		sem_post(sem_empty);
		sem_post(sem_mutex);
	}

	printf("producer finished\n");
	printf("items produced: %d\n", *producer_count);

	exit(0);
}
void consumer(sem_t *sem_mutex, sem_t *sem_empty, sem_t *sem_full, int *producer_buffer, int *consumer_sum, int *front_index, int *back_index)
{
	printf("created 1 consumer \n");
	sem_post(sem_mutex);
	// Write consumer code here
	while(item_consumed < EXIT_AMOUNT_CONSUMER)
	{
		sem_wait(sem_empty);
		sem_wait(sem_mutex);
		
		
		int num_to_read = producer_buffer[*front_index];
		*front_index = (*front_index + 1) % 10;
		*consumer_sum += num_to_read;
	
		item_consumed++;

		//printf("Adding into consumer sum: %d \n", num_to_read);
		//printf("consumer sum: %d\n", *consumer_sum );
		//printf("exit count: %d\n", exit_counter );

		
		sem_post(sem_full);
		sem_post(sem_mutex);
	}
	printf("consumer finished\n");
		
	printf("items consumed: %d\n", item_consumed);

	// exit(0); not needed as it returns back to the main() function
}

int main(int argc, char *argv[])
{
	clock_t start = clock();

	// Seed the random number generator with the current time
	srand((unsigned) time(NULL));
	// Get a random number to use as the shared memory key
	// This ensures that the shared memory key will be different across users and runs
	int shmrand = rand();
	// Generate the final shared memory key
	key_t shmkey = ftok("/dev/null", shmrand); /* valid directory name and a number */
	printf("shmkey for p = %d\n", shmkey);

	int shmid = shmget(shmkey, 14 * sizeof(int), 0644 | IPC_CREAT); // two shared integers 
	if (shmid < 0)
	{
		perror("shmget\n");
		exit(1);
	}

	// attach p to shared memory
	int *shared_memory = reinterpret_cast<int *>(shmat(shmid, NULL, 0));

	int* producer_count = &shared_memory[0];  
    int* consumer_sum = &shared_memory[1];

	int* front_index = &shared_memory[2];  
    int* back_index = &shared_memory[3];
	int* producer_buffer = &shared_memory[4];

	*producer_count = 0;
	*consumer_sum = 0;
	*front_index = 0;
	*back_index = 0;

	printf("shared_memory = %d is allocated in shared memory.\n\n", *shared_memory);


	// Compute the semaphore name for this run as a string containing the random number
	// i.e., pSem and shmrand appended together
	// This also ensures that the semaphore name will be different across users and runs
	auto semaphore_string = "pSem-" + std::to_string(shmrand);

	auto semaphore_mutex_string = semaphore_string + "-mutex";
	auto semaphore_empty_string = semaphore_string + "-empty";
	auto semaphore_full_string = semaphore_string + "-full";

	const char* semaphore_mutex = semaphore_mutex_string.c_str();
	const char* semaphore_empty = semaphore_empty_string.c_str();
	const char* semaphore_full = semaphore_full_string.c_str();

	sem_t *sem_mutex = sem_open(semaphore_mutex, O_CREAT | O_EXCL, 0644, 1);
	sem_t *sem_empty = sem_open(semaphore_empty, O_CREAT | O_EXCL, 0644, 0);
	sem_t *sem_full = sem_open(semaphore_full, O_CREAT | O_EXCL, 0644, 10); // At the start it is empty, so there are 10 not full counter (ish)


	pid_t pids[2];
	int index = 0;
	

	sem_wait(sem_mutex); // The next sem_post is in consumer before it enters the while loop to ensure all 3 process are created


	if ( (pids[index++] = fork()) == 0) producer(sem_mutex,sem_empty,sem_full,producer_buffer, consumer_sum, producer_count,  front_index,back_index); // producer 1
	if ( (pids[index++] = fork()) == 0) producer(sem_mutex,sem_empty,sem_full,producer_buffer, consumer_sum,  producer_count, front_index,back_index); // producer 2
	
	
	consumer(sem_mutex, sem_empty, sem_full, producer_buffer, consumer_sum, front_index, back_index);


	printf("index: %d \n", index);

	// TODO: HERE, handle the fork(), right now all threads print the final consumer sum
	for (int i = 0; i < 2; i++)
	{
		printf("waiting for chilld\n");
		waitpid(pids[i], NULL, 0);
		printf("Child process %d finished \n", pids[i]);
	}

	printf("Final Consumer Sum: %d\n", *consumer_sum);
	sem_close(sem_empty);
    sem_close(sem_full);
    sem_close(sem_mutex);

	// shared memory detach
	shmdt(shared_memory);
	shmctl(shmid, IPC_RMID, 0);


	sem_unlink(semaphore_mutex_string.c_str());
    sem_unlink(semaphore_empty_string.c_str());
    sem_unlink(semaphore_full_string.c_str());	


	
	clock_t end = clock();

	double duration = double(end - start) / CLOCKS_PER_SEC;
	printf("Time taken for script: %f \n", duration);

	return 0;
}
