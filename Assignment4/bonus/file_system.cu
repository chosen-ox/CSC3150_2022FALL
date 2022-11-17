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
    RESET(fs->SUPERBLOCK[i]);
  }

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
  int WD[2];
  get_WD(fs, WD);
  int empty_block = -1;
  if (WD[0] == -1) {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (ROOT(fs->SUPERBLOCK[i])) {
          if (cmp_str(fs->FCBS[i].name, s)) 
          return i;
        }
      }
      else {
        empty_block = i;
      }
    }
  }
  else {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (!ROOT(fs->SUPERBLOCK[i]) && PARENT(fs->SUPERBLOCK[i]) == WD[0]) {
          if (cmp_str(fs->FCBS[i].name, s)) 
            return i;
        }
      }
      else {
        empty_block = i;
      }
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
      RESET(fs->SUPERBLOCK[empty_block]);
      SET_VALID(fs->SUPERBLOCK[empty_block]);
      copy_str(s, fs->FCBS[empty_block].name);
      fs->FCBS[empty_block].create_time = gtime++;
      fs->FCBS[empty_block].modified_time = 0;
      if (WD[0] != -1) {
      SET_PARENT(fs->SUPERBLOCK[empty_block], WD[0]);
      set_size(&fs->FCBS[WD[0]], get_size(fs->FCBS[WD[0]]) + get_len(s));
      // printf("size:%d\n", get_size(fs->FCBS[WD[0]]) + get_len(s));
      }
      else {
        SET_ROOT(fs->SUPERBLOCK[empty_block]);
      }
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


  fs->FCBS[fp].modified_time = gtime++;
  set_size(&fs->FCBS[fp], size);
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
    int WD[2];
    get_WD(fs, WD);
    if (WD[0] != - 1) {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->SUPERBLOCK[i])) {
          if (PARENT(fs->SUPERBLOCK[i]) == WD[0] && !ROOT(fs->SUPERBLOCK[i])) {
            valid_fcbs[offset++] = fs->FCBS[i];
          }
        }
      }
    }
    else {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->SUPERBLOCK[i])) {
          if (ROOT(fs->SUPERBLOCK[i]))
            valid_fcbs[offset++] = fs->FCBS[i];
        }
      }
    }
    sort_by_date(valid_fcbs, offset);
    print_array_by_date(fs, valid_fcbs, offset);
  }
  else if (op == LS_S) {
    FCB valid_fcbs[1024];
    int offset = 0;
    int WD[2];
    get_WD(fs, WD);
    if (WD[0] != - 1) {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->SUPERBLOCK[i])) {
          if (PARENT(fs->SUPERBLOCK[i]) == WD[0] && !ROOT(fs->SUPERBLOCK[i])) {
            valid_fcbs[offset++] = fs->FCBS[i];
          }
        }
      }
    }
    else {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->SUPERBLOCK[i])) {
          if (ROOT(fs->SUPERBLOCK[i]))
            valid_fcbs[offset++] = fs->FCBS[i];
        }
      }
    }
    
    sort_by_size(valid_fcbs, offset);
    print_array_by_size(fs, valid_fcbs, offset);
  }
  else if (op == CD_P) {
    int WD[2];
    get_WD(fs, WD); 
    if (WD[0] == -1) {
      printf("Already in root directory!\n");
    }
    else {
      if (WD[1] != -1){
        SET_WD(fs->SUPERBLOCK[WD[1]]);
      }
      RESET_WD(fs->SUPERBLOCK[WD[0]]);
    }
  }
  else if (op == PWD) {
    int WD[2];
    get_WD(fs, WD);
    printf("/");
    if (WD[1] != - 1) {
      printf("%s", fs->FCBS[WD[1]].name);
      printf("/%s", fs->FCBS[WD[0]].name);
    }
    else if (WD[0] != -1) {
      printf("%s", fs->FCBS[WD[0]].name);
    }
    printf("\n");
  }
  	
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  
	if (op == RM) {
    int WD[2];
    get_WD(fs, WD);
    int file = -1;
    if (WD[0] == -1) {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (ROOT(fs->SUPERBLOCK[i])) {
          if (cmp_str(fs->FCBS[i].name, s)) 
          file =i;
        }
      }
    }
  }
  else {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (!ROOT(fs->SUPERBLOCK[i]) && PARENT(fs->SUPERBLOCK[i]) == WD[0]) {
          if (cmp_str(fs->FCBS[i].name, s)) 
            file = i;
        }
      }
    }
  }
    

    if (file == - 1) {
      printf("no such file to delete\n");
    }
    else if (DIR(fs->SUPERBLOCK[file])) {
      printf("It is a directory!\n");
    }
    else {
      for (int i = 0; i < 1024; i++) {
        fs->FILES[get_address(fs->FCBS[file])][i] = '\0';
      }

      if (!ROOT(fs->SUPERBLOCK[file])) 
      set_size(&fs->FCBS[WD[0]], get_size(fs->FCBS[WD[0]]) - get_len(s));
      RESET_VALID(fs->SUPERBLOCK[file]);
    }

  }
  else if (op == MKDIR) {
    int WD[2];  
    int empty_block;
    get_WD(fs, WD);
    for (int i = 0; i < 1024; i++) {
      if (!VALID(fs->SUPERBLOCK[i])) {
        empty_block = i;
        break;
      }
    }
    RESET(fs->SUPERBLOCK[empty_block]);
    SET_VALID(fs->SUPERBLOCK[empty_block]);
    copy_str(s, fs->FCBS[empty_block].name);
    SET_DIR(fs->SUPERBLOCK[empty_block]);
    fs->FCBS[empty_block].create_time = gtime++;
    fs->FCBS[empty_block].modified_time = gtime++;


    if (WD[0] != -1) {
      SET_PARENT(fs->SUPERBLOCK[empty_block], WD[0]);
      set_size(&fs->FCBS[WD[0]], get_size(fs->FCBS[WD[0]]) + get_len(s));
      // printf("size:%d\n", get_size(fs->FCBS[WD[0]]) + get_len(s));
    }
    else {
      SET_ROOT(fs->SUPERBLOCK[empty_block]);
    }
  }
  else if (op == CD) {
    int WD[2];  
    int block = -1;
    get_WD(fs, WD);

    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (cmp_str(fs->FCBS[i].name, s)) {
          block = i;
          break;
        }
      }
    }
    if (block == -1) {
      printf("No such directory!\n");
    }
    else {
      if (!DIR(fs->SUPERBLOCK[block])) {
        printf("Not a directory!\n");
      }
      else if (WD[0] != -1 && WD[0] != PARENT(fs->SUPERBLOCK[block])) {
        printf("No such directory in current directory!\n");
      }
      else {
        if (WD[0] != -1) {
          RESET_WD(fs->SUPERBLOCK[WD[0]]);
        }
        SET_WD(fs->SUPERBLOCK[block]);
      }
    } 
  }
  else if (op == RM_RF) {
    int WD[2];
    get_WD(fs, WD);
    int file = -1;
  if (WD[0] == -1) {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (ROOT(fs->SUPERBLOCK[i])) {
          if (cmp_str(fs->FCBS[i].name, s)) 
          file = i;
        }
      }
    }
  }
  else {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (!ROOT(fs->SUPERBLOCK[i]) && PARENT(fs->SUPERBLOCK[i]) == WD[0]) {
          if (cmp_str(fs->FCBS[i].name, s)) 
            file = i;
        }
      }
    }
  }
 
    if (file == - 1) {
      printf("no such directory to delete");
    }
    else if (DIR(fs->SUPERBLOCK[file])) {
      rm_DIR(fs, file);
    }
    else {
      fs_gsys(fs, RM, s);
    }

  }
}