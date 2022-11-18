#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  // fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  // super block 4096 byte
  // init free space management
  // 0-127 bytes
  for (int i = 0; i < 1024; i++) {
    fs->SUPERBLOCK[i] = 0;
  }
  // printf("%d\n", fs->SUPERBLOCK[0]);
  // 128-1151 bytes 
  // modification time sort
  // prev: >>10
  // next: &0x3ff 
  // head: 1152 bytes
  // tail: 1153 bytes
  


  // FCB *ptr = (FCB *) &fs->volume[fcb_base];

  for (int i = 0; i < 1024; i++) {
    set_address(&fs->FCBS[i], i);  
  }

}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  int empty_block = -1;
  for (int i = 0; i < 1024; i++) {
    if (VALID(fs->FCBS[i].address)) {
        if (cmp_str(fs->FCBS[i].name, s)) {
          return i;
        }
    }
    else {
      empty_block = i;
    }
  }

  if (op == G_READ) {
    printf("No such file!!!\n");
    return 0;
  }
  else if (op == G_WRITE) {
    if (empty_block == -1) {
      printf("No more space to create new file!!!\n");
      return 0;
    }
    else {
      // fs->SUPERBLOCK[empty_block] = 1;
      // fs->FCBS[empty_block].create_time = gtime++;
      // fs->FCBS[empty_block].modified_time = 0;
      if (gtime == 65535) {
        gtime = sort_by_time(fs->FCBS);
      }
      // empty not anymore
      SET_VALID(fs->FCBS[empty_block].address);
      copy_str(s, fs->FCBS[empty_block].name);
      set_create_time(&fs->FCBS[empty_block], gtime);
      set_modified_time(&fs->FCBS[empty_block], gtime++);
      return empty_block;
    }
  }
  else {
    printf("Please input correct op!!!\n");
    return 0;
  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	for (int i = 0; i < size; i++) {
    output[i] = fs->FILES[get_address(fs->FCBS[fp])][i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{


  if (gtime == 65535) {
    gtime = sort_by_time(fs->FCBS);
    printf("gtime:%d\n", gtime);
  }

  set_modified_time(&fs->FCBS[fp], gtime++);
  // printf("name%s time: %d\n", fs->FCBS[fp].name, gtime - 1);

  fs->FCBS[fp].size = size;
  for (int i =0; i < 1024; i++) {
    fs->FILES[get_address(fs->FCBS[fp])][i] = '\0';
  }
  for (int i = 0; i < size; i++) {
    fs->FILES[get_address(fs->FCBS[fp])][i] = input[i];
  }
  return 0;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  if (op == LS_D) {
    FCB valid_fcbs[1024];
    int offset = 0;
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        valid_fcbs[offset++] = fs->FCBS[i];
      }
    }
    sort_by_date(valid_fcbs, offset);
    print_array_by_date(valid_fcbs, offset);
  }
  else if (op == LS_S) {
    FCB valid_fcbs[1024];
    int offset = 0;
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        valid_fcbs[offset++] = fs->FCBS[i];
      }
    }
    sort_by_size(valid_fcbs, offset);
    print_array_by_size(valid_fcbs, offset);


  }
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	if (op == RM) {
    int file = -1;
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (cmp_str(fs->FCBS[i].name, s)) {
          file = i;
        }
      }
    }

    if (file == - 1) {
      printf("no such file to delete");
    }
    else {
      for (int i = 0; i < 1024; i++) {
        fs->FILES[get_address(fs->FCBS[file])][i] = '\0';
      }
      RESET_VALID(fs->FCBS[file].address);

    }

  }
}