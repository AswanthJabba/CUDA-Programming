/*
This is a simple CUDA program which performs addition using GPU (kernel function) 
*/
#include<stdio.h>
__global__ void add(int a, int b, int *d_c)
{
    *d_c = a + b;
}
int main()
{
    int a,b,c;
    int *d_c;
    a=3;
    b=4;
    // Allocating memory for device pointer of integer
    cudaMalloc((void**)&d_c, sizeof(int));
    // Calling Kernel function
    add<<<1,1>>>(a,b,d_c);
    // Copying the result from device to host
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d + %d is %d\n", a, b, c);
    // Free the device pointer
    cudaFree(d_c);
    return 0;
}
