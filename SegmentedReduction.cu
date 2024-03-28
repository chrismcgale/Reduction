__global__ void SegmentedSumReductionKernel(float * input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    // Performs seperate reductions on 'segments' of input and then atomically adds the results
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    // Stride halves each time. All threads in warp will be on or off until stride < Warp Size.
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(output, input_s[0]);
    }
}