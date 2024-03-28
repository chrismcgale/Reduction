// Used if execution resources aren't enough to fully parallelize
__global__ void CoarsenedSumReductionKernel(float * input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    // Performs seperate reductions on 'segments' of input and then atomically adds the results
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = input[i];
    // Each thread sums up COARSE_FACTOR * 2 elements before proceeding
    for (unsigned int tile = 1; tile < COARSE_FACTOR*2; tile++) {
        sum += input[i + tile * BLOCK_DIM]
    }
    input_s[t] = sum;
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