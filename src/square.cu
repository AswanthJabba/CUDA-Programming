/*
This program performs sqaure of a number from 1 to N parallely using N threads and 1 Block
Warning : It is not advisable to practice using single block multiple threads, Please have a look at square2.cu for a better practice of blocks and threads
*/

#include <stdio.h>
#include <cuda.h>

#define N 128

__global__ void f(int *dev_a) {
    unsigned int tid = threadIdx.x;

    if(tid < N) {
        dev_a[tid] = tid * tid;
    }
}

int main(void) {

    int host_a[N];
    int *dev_a;
    //Memory allocation for device (GPU)
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    for(int i = 0 ; i < N ; i++) {
        host_a[i] = i;
    }
    for(int i = 0 ; i < N ; i++) {
        printf("%d\t ", host_a[i]);
    }
    printf("\n");
    //Copy data from Host(CPU) to device(GPU)
    cudaMemcpy(dev_a, host_a, N * sizeof(int), cudaMemcpyHostToDevice);
    //Calling GPU kernel (GPu function)
    f<<<1, N>>>(dev_a);
    //Copying back the result from Device(GPU) to Host(CPU)
    cudaMemcpy(host_a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0 ; i < N ; i++) {
        printf("%d\t ", host_a[i]);
    }
}
