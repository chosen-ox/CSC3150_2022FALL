#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  for (int i = 0; i < input_size; i++){
    if (threadIdx.x == 0) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads();
    if (threadIdx.x == 1) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads();
    if (threadIdx.x == 2) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads();
    if (threadIdx.x == 3) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads();
  }


  // for (int i = input_size - 1; i >= input_size - 32769; i--){
  //   if (threadIdx.x == 0) {
  //     vm_read(vm, i);
  //   }
  //   __syncthreads();
  //   if (threadIdx.x == 1) {
  //     vm_read(vm, i);
  //   }
  //   __syncthreads();
  //   if (threadIdx.x == 2) {
  //     vm_read(vm, i);
  //   }
  //   __syncthreads();
  //   if (threadIdx.x == 3) {
  //     vm_read(vm, i);
  //   }
  //   __syncthreads();

  //   // int value = vm_read(vm, i);
  // }
  __syncthreads();
//   if (threadIdx.x == 3) {
//     vm_snapshot(vm, results, 0, input_size);
// }
 
}

// __device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
//   int input_size) {
// // write the data.bin to the VM starting from address 32*1024
// for (int i = 0; i < input_size; i++)
//   vm_write(vm, 32*1024+i, input[i]);
// // write (32KB-32B) data  to the VM starting from 0
// for (int i = 0; i < 32*1023; i++)
//   vm_write(vm, i, input[i+32*1024]);
// // readout VM[32K, 160K] and output to snapshot.bin, which should be the same with data.bin
// __syncthreads();
// if (threadIdx.x = 3) {
//   vm_snapshot(vm, results, 32*1024, input_size);
// }
 
// }

// // expected page fault num: 9215