#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    // vm->invert_page_table[i + vm->PAGE_ENTRIES] = -1;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES, u32* LRU_ARRAY) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;
  vm->LRU_ARRAY = LRU_ARRAY;
  vm->time_counter = 0;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->LRU_ARRAY[i] = vm->time_counter++;
  }

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
/* Complete vm_write function to write value into data buffer */
  u32 offset = addr & 0x1f;
  u32 vpn = (addr & 0x0fffffff) >> 5;
  // u32 vpn = addr / 32;
  // printf("vpn%d\n", vpn);
  int page_entry = vm_search_vpn(vm, vpn);
  // printf("search %d", vm_search_vpn(vm, 3072));
  if (page_entry == -1) { 
    ++(*vm->pagefault_num_ptr);
    int lru_idx = vm_get_LRU_idx(vm);

    if ((vm->invert_page_table[lru_idx] & 0x80000000) == 0) {
      // printf("index%d\n",lru_idx);
      vm_swap_to_storage(vm, vm->invert_page_table[lru_idx], lru_idx);
    }
    vm_swap_to_data(vm, vpn, lru_idx);
    page_entry = vm_search_vpn(vm, vpn);
  }
  vm_update_queue(vm, page_entry);
  
  // printf("vpn:%d vm0%x  phy:%d, offset:%d\n", vpn, vm->invert_page_table[0], page_entry,offset);

  return vm->buffer[page_entry * vm->PAGESIZE + offset]; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 offset = addr & 0x1f;
  u32 vpn = (addr & 0x0fffffff) >> 5;
  // u32 vpn =  addr >> 5;
  // printf("vpn%d\n", vpn);
  int page_entry = vm_search_vpn(vm, vpn);
  if (page_entry == -1) { 
    ++(*vm->pagefault_num_ptr);
    int lru_idx = vm_get_LRU_idx(vm);


    if ((vm->invert_page_table[lru_idx] & 0x80000000) == 0) {
      vm_swap_to_storage(vm, vm->invert_page_table[lru_idx], lru_idx);
    }
    vm_swap_to_data(vm, vpn, lru_idx);
    page_entry = vm_search_vpn(vm, vpn);
  }
  vm_update_queue(vm, page_entry);
  
  vm->buffer[page_entry * vm->PAGESIZE + offset] = value; //TODO
  // printf("vpn:%d vm0%x val:%x, phy:%d, offset:%d\n", vpn, vm->invert_page_table[0], value, page_entry,offset);
}



__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  for (int i=0; i<input_size;i++){
    int value = vm_read(vm,i);
    results[i+offset] = value;
  }


}

__device__ int vm_search_vpn(VirtualMemory *vm, u32 vpn) {
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if ((vm->invert_page_table[i] & 0x80000000) == 0) {
      if ((vm->invert_page_table[i]>>28) == threadIdx.x) {
        if ((vm->invert_page_table[i] & 0x0fffffff) == vpn) {
          return i;
        }
      }
    }
  }
  return -1 ;
 
}
__device__ void vm_update_pt(VirtualMemory *vm, u32 vpn, int page_entry) {
    vm->invert_page_table[page_entry] = vpn | (threadIdx.x << 28);
}
__device__ void vm_update_queue(VirtualMemory *vm, int page_entry) {
  vm->time_counter++;
  vm->LRU_ARRAY[page_entry] = vm->time_counter;
}

__device__ int vm_get_LRU_idx(VirtualMemory *vm) {
  int min_idx = 0;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if (vm->LRU_ARRAY[i] < vm->LRU_ARRAY[min_idx]) {
      min_idx = i;
    }
  }
  return min_idx;
} 


__device__ void vm_swap_to_storage(VirtualMemory *vm, u32 vpn, int page_entry) {
  for (int i = 0; i < vm->PAGESIZE; i++) {
    vm->storage[vpn * vm->PAGESIZE + i] = vm->buffer[page_entry * vm->PAGESIZE + i];
  }
  // vm->invert_page_table[page_entry] = 0x80000000;
}
__device__ void vm_swap_to_data(VirtualMemory *vm, u32 vpn, int page_entry) {
  for (int i = 0; i < vm->PAGESIZE; i++) {
    vm->buffer[page_entry * vm->PAGESIZE + i] = vm->storage[vpn * vm->PAGESIZE + i];
  }
  vm_update_pt(vm, vpn, page_entry);
}