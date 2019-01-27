#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
    	// Define blocks and threads then calculate
    	int i = blockIdx.x * blockDim.x + threadIdx.x;
    	
	for(int j = 0; j < 100000; j++)
		if (i < n) y[i] = a*x[i] + y[i];

    	__syncthreads();
}

int main(void)
{
    // N = 1M
    int N = 1<<23;

    // Create host and device memory
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    // Assign local memory
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Copy memory to device
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    // Copy completed arithmetic back to host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up device
    cudaDeviceReset();

    // Check for errors
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i]-200002.0f));

    printf("Max error: %f\n", maxError);
    printf("%f\n", y[1]);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}
