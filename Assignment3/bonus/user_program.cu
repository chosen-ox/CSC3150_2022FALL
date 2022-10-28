#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
  int index = threadIdx.x;
  for (int i = 0; i < input_size; i++){
    if (index == 0) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads();
    if (index == 1) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads();
    if (index == 2) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads(); 
    if (index == 3) {
      vm_write(vm, i, input[i]);
    }
    __syncthreads(); 

  }

  for (int i = input_size - 1; i >= input_size - 32769; i--){
    if (index == 0) {
      int value = vm_read(vm, i);
    }
    __syncthreads(); 
    if (index == 1) {
      int value = vm_read(vm, i);
    }
    __syncthreads(); 
    if (index == 2) {
      int value = vm_read(vm, i);
    }
    __syncthreads(); 
    if (index == 3) {
      int value = vm_read(vm, i);
    }
    __syncthreads(); 
  }
  if (index == 0) {
      vm_snapshot(vm, results, 0, input_size);
    }

}
