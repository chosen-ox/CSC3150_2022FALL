﻿#include "file_system.h"
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
  // for (int i = 0; i < 1024; i++) {
  //   printf("%x\n", get_address(fs->FCBS[i]));
  // }
  

}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  int empty_block = -1;
  for (int i = 0; i < 1024; i++) {
    if (fs->SUPERBLOCK[i] == 1) {
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
      fs->SUPERBLOCK[empty_block] = 1;
      copy_str(s, fs->FCBS[empty_block].name);
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
	/* Implement read operation here */
  for (int i = 0; i < size; i++) {
    // output[i] = fs->volume[fp+i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{

  // for (int i =0; i < 1024; i++) {
  //   printf("%c\n", ptr[i]);
  // }
  for (int i = 0; i < size+1; i++) {
    // fs->volume[fp + i] = input[i];
  }
	/* Implement write operation here */
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  if (op == LS_D) {

  }
  
  else if (op == LS_S) {

  }
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}