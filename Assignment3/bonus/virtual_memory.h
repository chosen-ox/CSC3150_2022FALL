#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

struct VirtualMemory {
  uchar *buffer;
  uchar *storage;
  u32 *invert_page_table;
  u32 *swap_page_table;
  int *pagefault_num_ptr;

  int PAGESIZE;
  int INVERT_PAGE_TABLE_SIZE;
  int PHYSICAL_MEM_SIZE;
  int STORAGE_SIZE;
  int PAGE_ENTRIES;
  u32 * LRU_ARRAY;
  int time_counter;
};

// TODO
__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES,
                        u32 *LRU_ARRAY,
                        u32 *swap_page_table);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size);
__device__ int vm_search_vpn(VirtualMemory *vm, u32 vpn);
__device__ void vm_update_pt(VirtualMemory *vm, u32 vpn, int page_entry);
__device__ void vm_update_queue(VirtualMemory *vm, int page_entry);
__device__ int vm_get_LRU_idx(VirtualMemory *vm);
__device__ void vm_swap_to_storage(VirtualMemory *vm, int vpn, int page_entry);
__device__ void vm_swap_to_data(VirtualMemory *vm, uchar * vpn, int page_entry);
__device__ int vm_search_swap_table(VirtualMemory *vm, u32 vpn);
__device__ int vm_find_empty_entry_disk(VirtualMemory *vm);
#endif
