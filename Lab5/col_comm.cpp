/**
 * CS3210 - Collective communication in MPI.
 */

#include <mpi.h>
#include <cstdio>
#define SIZE 4

int main(int argc, char *argv[])
{
	int numtasks, rank, sendcount, recvcount, source;
	float sendbuf[SIZE][SIZE] = 
	{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
		{13.0, 14.0, 15.0, 16.0}  
	};
	float recvbuf[SIZE];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	if (numtasks == SIZE) {
		source = 0;
		sendcount = SIZE;
		recvcount = SIZE;
  	
		MPI_Scatter(sendbuf,sendcount,MPI_FLOAT,recvbuf,recvcount,
			MPI_FLOAT,source,MPI_COMM_WORLD);
		printf("rank= %d  Results: %f %f %f %f\n",rank,recvbuf[0],
			recvbuf[1],recvbuf[2],recvbuf[3]);

		float sum = 0;

		// TODO: Exercise 3
		
		for(int i = 0; i < recvcount; i++)
		{
			sum += recvbuf[i];
		}
		float rootRecvBuf[SIZE]; // everyone sends data to root, even root itself
		MPI_Gather(&sum, 1, MPI_FLOAT, &rootRecvBuf, 1, MPI_FLOAT,source, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		float ans;
		if (rank == 0)
		{
			for(int i = 0; i < SIZE; i++)
			{
				printf("curr: %f\n", rootRecvBuf[i]);
				ans += rootRecvBuf[i];
			}
			printf("Final Sum: %f\n", ans);
		
		}

		
	} else {
		printf("Must specify %d processes. Terminating.\n",SIZE);
	}

	MPI_Finalize();
	
	return 0;
	
}
