#include "kseq/kseq.h"
#include "common.h"
#include <iostream>



/*
    Helper function to compare two strings of the same length
    Assumption: The two strings are of the same length
*/
__device__ bool isTwoStringsMatch(const char* s1, const char* s2, int len)
{
    for (int i = 0; i < len; i++)
    {
        char c1 = s1[i];
        char c2 = s2[i];
        bool isMatch = (c1 == c2) || (c1 == 'N') || (c2 == 'N');
        if (!isMatch) return false;
    }
    return true;
}

/*
    Compares a sample with a signature
    Each thread processes multiple starting positions.
*/
__device__ void isMatch(const char* sampleSeq, const char* signatureSeq, int sampleLen, int signatureLen, int* matchIndex) 
{
    __shared__ int earliestMatch;  // Shared memory to store the earliest match
    if (threadIdx.x == 0)
    {
        earliestMatch = sampleLen;  // Initialize to max possible index
        *matchIndex = -1;
    }
    __syncthreads();

    int validPositions = sampleLen - signatureLen + 1;  // Total valid starting positions
    //printf("valid position: %i", validPositions);

    if (validPositions > 0)
    {
        int positionsPerThread = (validPositions + blockDim.x - 1) / blockDim.x;  // Number of positions each thread will handle

        int startIdx = threadIdx.x * positionsPerThread;  // Starting position for this thread
        int endIdx = min(startIdx + positionsPerThread, validPositions);  // Ending position

        // Each thread checks multiple starting positions
        for (int idx = startIdx; idx < endIdx; ++idx) 
        {
            if (idx + signatureLen <= sampleLen)  // Add this boundary check
            {
                if (isTwoStringsMatch(&(sampleSeq[idx]), signatureSeq, signatureLen)) 
                {
                    atomicMin(&earliestMatch, idx);  // Update to the smallest (earliest) match found
                }
            }
        }
    }
    

    __syncthreads();  // Ensure all threads have finished updating earliestMatch

    if (threadIdx.x == 0)
    {  
        if (earliestMatch < sampleLen) *matchIndex = earliestMatch;
        else *matchIndex = -1;
    } 
}


__device__ double getMatchScore(const char* s , int len)
{
    double score = 0;
    for(int i = 0; i < len; i++)
    {
        score += (int)(s[i]) - 33;
    }
    score /= len; // I think can be improved
    return score;
}

/*
    Calls and compares each of the samples with all of the signatures
*/

__global__ void yourKernel(
    char* d_allSampleSeqs, size_t* d_sampleSeqOffsets,
    char* d_allSampleQuals, size_t* d_sampleQualOffsets,
    char* d_allSignatureSeqs, size_t* d_signatureSeqOffsets,
    char* d_allSignatureQuals, size_t* d_signatureQualOffsets,
    MatchResult* d_matches,
    int numSamples, int numSignatures)
{
    int sampleIdx = blockIdx.x;      // Block index determines the sample
    int signatureIdx = blockIdx.y;   // Block index determines the signature

    if (sampleIdx < numSamples && signatureIdx < numSignatures) 
    {
        // Access sample sequence and quality
        char* sampleSeq = &d_allSampleSeqs[d_sampleSeqOffsets[sampleIdx]];
        char* sampleQual = &d_allSampleQuals[d_sampleQualOffsets[sampleIdx]];
        int sampleLen = d_sampleSeqOffsets[sampleIdx + 1] - d_sampleSeqOffsets[sampleIdx] - 1;

        // Access signature sequence and quality
        char* signatureSeq = &d_allSignatureSeqs[d_signatureSeqOffsets[signatureIdx]];
        int signatureLen = d_signatureSeqOffsets[signatureIdx + 1] - d_signatureSeqOffsets[signatureIdx] - 1;

        __shared__ int matchIndex;  // This will store the result of isMatch

        if (threadIdx.x == 0)
        
        {
            matchIndex = -1;  // Initialize matchIndex to -1
        }
        __syncthreads();

        // Call isMatch to compare the sample with the signature
        isMatch(sampleSeq, signatureSeq, sampleLen, signatureLen, &matchIndex);

        __syncthreads();

        if (threadIdx.x == 0)
        {
            // Store the result in d_matches (one entry per sample-signature pair)
            if (matchIndex != -1) 
            {
                // Calculate match score using the correct portion of the quality string
                double match_score = getMatchScore(&sampleQual[matchIndex], signatureLen);
                d_matches[sampleIdx * numSignatures + signatureIdx].match_score = match_score;  // Match found
            } 
            else 
            {
                d_matches[sampleIdx * numSignatures + signatureIdx].match_score = 0.0;  // No match
            }
        }
    }
}


void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) 
{
    int numSamples = samples.size();
    int numSignatures = signatures.size();

    // Allocate managed memory for MatchResult as we want the result back on host
    MatchResult* d_matches;
    cudaMallocManaged(&d_matches, numSamples * numSignatures * sizeof(MatchResult));

    /*
        Idea here it to use continguous memory as one single array for all of the elements 
        So that when we do CudaMpyCpy its one chunk of data at once
    */

    // The extra +1 at the end is to store the totalSampleSeqSize and totalSampleQualSize which we did not really use
    std::vector<size_t> sampleSeqOffsets(numSamples + 1, 0);
    std::vector<size_t> sampleQualOffsets(numSamples + 1, 0); 
    size_t totalSampleSeqSize = 0;
    size_t totalSampleQualSize = 0;

    for (int i = 0; i < numSamples; i++) 
    {
        sampleSeqOffsets[i] = totalSampleSeqSize;
        totalSampleSeqSize += samples[i].seq.length() + 1; // +1 for null terminator when we convert it from std::string to a char* so that CUDA can work on it

        sampleQualOffsets[i] = totalSampleQualSize;
        totalSampleQualSize += samples[i].qual.length() + 1;
    }
    sampleSeqOffsets[numSamples] = totalSampleSeqSize;
    sampleQualOffsets[numSamples] = totalSampleQualSize;
    


    /*
        h_ -> host side
        d_ -> device side
    */
    // Allocate and fill contiguous arrays on host
    char* h_allSampleSeqs = (char*)malloc(totalSampleSeqSize);
    char* h_allSampleQuals = (char*)malloc(totalSampleQualSize);

    for (int i = 0; i < numSamples; i++) 
    {
        strcpy(&h_allSampleSeqs[sampleSeqOffsets[i]], samples[i].seq.c_str());
        strcpy(&h_allSampleQuals[sampleQualOffsets[i]], samples[i].qual.c_str());
    }

    // Doing the same thing for signature
    std::vector<size_t> signatureSeqOffsets(numSignatures + 1, 0);
    std::vector<size_t> signatureQualOffsets(numSignatures + 1, 0);
    size_t totalSignatureSeqSize = 0;
    size_t totalSignatureQualSize = 0;

    for (int i = 0; i < numSignatures; i++) 
    {
        signatureSeqOffsets[i] = totalSignatureSeqSize;
        totalSignatureSeqSize += signatures[i].seq.length() + 1; // +1 for null terminator

        signatureQualOffsets[i] = totalSignatureQualSize;
        totalSignatureQualSize += signatures[i].qual.length() + 1;
    }
    signatureSeqOffsets[numSignatures] = totalSignatureSeqSize;
    signatureQualOffsets[numSignatures] = totalSignatureQualSize;

    // Allocate and fill contiguous arrays on host
    char* h_allSignatureSeqs = (char*)malloc(totalSignatureSeqSize);
    char* h_allSignatureQuals = (char*)malloc(totalSignatureQualSize);

    for (int i = 0; i < numSignatures; i++) 
    {
        strcpy(&h_allSignatureSeqs[signatureSeqOffsets[i]], signatures[i].seq.c_str());
        strcpy(&h_allSignatureQuals[signatureQualOffsets[i]], signatures[i].qual.c_str());
    }

    // Allocate device memory for samples
    char* d_allSampleSeqs;
    cudaMalloc(&d_allSampleSeqs, totalSampleSeqSize);

    char* d_allSampleQuals;
    cudaMalloc(&d_allSampleQuals, totalSampleQualSize);

    size_t* d_sampleSeqOffsets;
    cudaMalloc(&d_sampleSeqOffsets, (numSamples + 1) * sizeof(size_t));

    size_t* d_sampleQualOffsets;
    cudaMalloc(&d_sampleQualOffsets, (numSamples + 1) * sizeof(size_t));

    // Allocate device memory for signatures
    char* d_allSignatureSeqs;
    cudaMalloc(&d_allSignatureSeqs, totalSignatureSeqSize);

    char* d_allSignatureQuals;
    cudaMalloc(&d_allSignatureQuals, totalSignatureQualSize);

    size_t* d_signatureSeqOffsets;
    cudaMalloc(&d_signatureSeqOffsets, (numSignatures + 1) * sizeof(size_t));

    size_t* d_signatureQualOffsets;
    cudaMalloc(&d_signatureQualOffsets, (numSignatures + 1) * sizeof(size_t));

    // Copy data to device
    cudaMemcpy(d_allSampleSeqs, h_allSampleSeqs, totalSampleSeqSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_allSampleQuals, h_allSampleQuals, totalSampleQualSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sampleSeqOffsets, sampleSeqOffsets.data(), (numSamples + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sampleQualOffsets, sampleQualOffsets.data(), (numSamples + 1) * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_allSignatureSeqs, h_allSignatureSeqs, totalSignatureSeqSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_allSignatureQuals, h_allSignatureQuals, totalSignatureQualSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatureSeqOffsets, signatureSeqOffsets.data(), (numSignatures + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatureQualOffsets, signatureQualOffsets.data(), (numSignatures + 1) * sizeof(size_t), cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockDim(512);  // Number of threads per block
    dim3 gridDim(numSamples, numSignatures);  // Grid of samples x signatures

    // Launch the kernel
    yourKernel<<<gridDim, blockDim>>>(
        d_allSampleSeqs, d_sampleSeqOffsets,
        d_allSampleQuals, d_sampleQualOffsets,
        d_allSignatureSeqs, d_signatureSeqOffsets,
        d_allSignatureQuals, d_signatureQualOffsets,
        d_matches,
        numSamples, numSignatures);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Process results
    matches.clear();

    // Loop through d_matches and add only positive matches
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) 
    {
        for (int signatureIdx = 0; signatureIdx < numSignatures; signatureIdx++) 
        {
            int idx = sampleIdx * numSignatures + signatureIdx;
            if (d_matches[idx].match_score > 0) 
            {
                // Create a new MatchResult object
                MatchResult match;
                match.match_score = d_matches[idx].match_score;
                match.sample_name = samples[sampleIdx].name;
                match.signature_name = signatures[signatureIdx].name;

                // Add the match to the matches vector
                matches.push_back(match);
            }
        }
    }

    // Free device memory
    cudaFree(d_allSampleSeqs);
    cudaFree(d_allSampleQuals);
    cudaFree(d_sampleSeqOffsets);
    cudaFree(d_sampleQualOffsets);

    cudaFree(d_allSignatureSeqs);
    cudaFree(d_allSignatureQuals);
    cudaFree(d_signatureSeqOffsets);
    cudaFree(d_signatureQualOffsets);

    cudaFree(d_matches);

    // Free host memory
    free(h_allSampleSeqs);
    free(h_allSampleQuals);
    free(h_allSignatureSeqs);
    free(h_allSignatureQuals);
}