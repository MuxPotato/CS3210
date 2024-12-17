# AY24/25 S1 CS3210 Parallel Computing 

This repository contains the Assignments, Tutorials and Labs of CS3210 Parallel Computing taken in AY24/25 Semester 1, exploring the power of parallelism using OpenMP, CUDA, and OpenMPI. 


Each project focuses on implementing efficient solutions to computationally intensive problems, leveraging multi-threading, GPU acceleration, and distributed memory paradigms. The assignments and projects demonstrate my hands-on experience with real-world parallel programming techniques, optimizing performance, and analyzing scalability across different architectures.


## Assignments Description

### Assignment 1: Parallel Particle Collision Simulator (OpenMP)
Develop a 2D particle simulation that models the movement and collisions of particles within a square area. The simulation involved handling particle-particle and particle-wall elastic collisions while conserving energy and momentum. Using OpenMP, the program was parallelized to efficiently process tens of thousands of particles over multiple timesteps. Key challenges included implementing spatial partitioning to optimize collision detection and ensuring correctness through step-by-step validation. The performance was evaluated by comparing execution times against benchmark implementations.


### Assignment 2: Virus Signature Scanning (CUDA)
Implementing a highly parallelized CUDA program to detect viral DNA sequences (signatures) within patient DNA samples. Given large datasets in FASTQ and FASTA formats, the program identifies matches between samples and signatures while accounting for wildcard nucleotides ('N') and calculates a match confidence score based on Phred quality scores. The implementation emphasized leveraging GPU capabilities through efficient memory management, thread-block configurations, and performance profiling. Correctness and speed were evaluated against benchmark solutions on NVIDIA A100 and H100 GPUs.

### Assignment 3: MRT Network Simulation (OpenMPI)
Simulate a simplified Mass Rapid Transit (MRT) network using MPI for distributed-memory parallel programming. The simulation involved trains moving across stations, managing loading/unloading passengers, and ensuring synchronization at shared tracks and platforms. The program handled train spawning, queuing, direction changes at terminal stations, and conflicts on shared links. Key challenges included implementing efficient communication between MPI processes to maintain correctness and achieving performance scalability across nodes. The output tracked the trains' positions over simulation ticks, ensuring correctness and speedup compared to a sequential reference implementation.

## Understanding the Assignments
Each assignment subfolder contains the code, input/output samples, and a PDF report explaining my approach, design decisions, and optimizations:

- [**Assignment 1: Parallel Particle Collision Simulator**](A1/A1_Report.pdf)  
- [**Assignment 2: Virus Signature Scanning with CUDA**](A2/A2_Report.pdf)  
- [**Assignment 3: MRT Network Simulation with MPI**](A3/A3_Report.pdf)


