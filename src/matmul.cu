//Matrix Multiplication using CUDA

#include <stdio.h>
#include <math.h>
#define TILE_SIZE 16

//matrix multiplication kernel function 
__global__ void MatrixMultiplication( float *device_array1 , float *device_array2 , float *device_result_array , const int SIZE )
{

    __shared__ float Mds [TILE_SIZE][TILE_SIZE] ;
    __shared__ float Nds [TILE_SIZE][TILE_SIZE] ;

    int col = blockDim.x*blockIdx.x + threadIdx.x ;
    int row = blockDim.y*blockIdx.y + threadIdx.y ;
    

    for (int m = 0 ; m<SIZE/TILE_SIZE ; m++ ) 
    {
        Mds[threadIdx.y][threadIdx.x] =  device_array1[row*SIZE + (m*TILE_SIZE + threadIdx.x)]  ;
        Nds[threadIdx.y][threadIdx.x] =  device_array2[(m*TILE_SIZE + threadIdx.y) * SIZE + col] ;
         __syncthreads(); 
        for ( int k = 0; k<TILE_SIZE ; k++ )
            device_result_array[row*SIZE + col]+= Mds[threadIdx.x][k] * Nds[k][threadIdx.y] ;
            __syncthreads();

    }
}
int main ()
{
    const int SIZE = 512 ; 
    int i,j;
    // Host Vairables
    float host_array1[SIZE][SIZE], host_array2[SIZE][SIZE], host_result_array[SIZE][SIZE];
    // Device Variables
    float *device_array1 , *device_array2  ,*device_result_array ; 

    // Inserting values into Arrays
    for (i = 0 ; i<SIZE ; i++ )
    {
        for (j = 0 ; j<SIZE ; j++ )
        {
            host_array1[i][j] = 2;
            host_array2[i][j] = 1 ;
        }
    }

    // Allocate memory for GPU 
    cudaMalloc((void **) &device_array1 , SIZE*SIZE*sizeof (int)) ;
    cudaMalloc((void **) &device_array2 , SIZE*SIZE*sizeof (int)) ;
    cudaMalloc((void **) &device_result_array , SIZE*SIZE*sizeof (int)) ;
 
    // Copying yhe array from Host to Device Array
    cudaMemcpy ( device_array1 , host_array1 , SIZE*SIZE*sizeof (int) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy ( device_array2 , host_array2 , SIZE*SIZE*sizeof (int) , cudaMemcpyHostToDevice ) ;
 
    //calling kernal
    dim3 dimGrid ( SIZE/TILE_SIZE , SIZE/TILE_SIZE ) ;
    dim3 dimBlock( TILE_SIZE, TILE_SIZE ) ;
    MatrixMultiplication<<<dimGrid,dimBlock>>> ( device_array1 , device_array2 ,device_result_array , SIZE) ;

    // all gpu function blocked till kernel is working
    //copy back result_array_d to result_array_h
    cudaMemcpy(host_result_array , device_result_array , SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost) ;

    //printf the result array
    for ( i = 0 ; i<SIZE ; i++ )
    {
        for ( j = 0 ; j < SIZE ; j++ )
        {
            printf ("%.0f ",host_result_array[i][j] ) ;
        }
    printf ("\n") ;
    }
}
