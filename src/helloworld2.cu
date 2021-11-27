#include<stdio.h>
#include<stdlib.h>
__global__ void print_from_gpu(void) {
 printf("Hello World! from thread [%d,%d] From device\n", threadIdx.x,blockIdx.x);
}

int main(void) {
 printf("Hello World from host!\n");
 print_from_gpu<<<10,2>>>(); // first parameter is number of Blocks, second parameter is number of threads 
 cudaDeviceSynchronize();
 return 0;
}
