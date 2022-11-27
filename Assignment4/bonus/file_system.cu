#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, int SUPERBLOCK_SIZE,
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
    set_address(&fs->FCBS[i], 0);  
  }

}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  int WD[2];
  get_WD(fs, WD);
  int empty_block = -1;
  if (WD[0] == -1) {
  for (int i = 0; i < 1024; i++) {
    if (VALID(fs->FCBS[i].address)) {
      if(ROOT(fs->FCBS[i].address)) {
        if (!DIR(fs->FCBS[i].address)) {
        if (cmp_str(fs->FCBS[i].name, s)) {
          if (op == G_READ) {
            SET_READ(i);
          }
          else {
            SET_WRITE(i);
          }
          return i;
        }
      }
      }
    }
    else {
      empty_block = i;
    }
  }
  }
  else {
    for (int i = 0; i < 1024; i++) {
    if (VALID(fs->FCBS[i].address)) {
      if (!ROOT(fs->FCBS[i].address) && PARENT(fs->FCBS[i].address) == WD[0]) {
        if (!DIR(fs->FCBS[i].address)) {
        if (cmp_str(fs->FCBS[i].name, s)) {
          if (op == G_READ) {
            SET_READ(i);
          }
          else {
            SET_WRITE(i);
          }
          return i;
          }
        }
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
      printf("The file number reaches the limit!!!\n");
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
      RESET(fs->FCBS[empty_block].address);
      SET_VALID(fs->FCBS[empty_block].address);
      copy_str(s, fs->FCBS[empty_block].name);
      set_create_time(&fs->FCBS[empty_block], gtime);
      set_modified_time(&fs->FCBS[empty_block], gtime);
      fs->FCBS[empty_block].size = 0;
      if (WD[0] != -1) {
        SET_PARENT(fs->FCBS[empty_block].address, WD[0]);
        fs->FCBS[WD[0]].size += get_len(s);
      }
      else {
        SET_ROOT(fs->FCBS[empty_block].address);
      }

      if (WD[0] != -1) 
      set_modified_time(&fs->FCBS[WD[0]], gtime);
      if (WD[1] != -1)
      set_modified_time(&fs->FCBS[WD[1]], gtime);
      gtime++;
      SET_WRITE(empty_block);
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
  if (!READ(fp)) {
    printf("No read permission!\n");
    return ; 
  }
  fp = fp & 0x0000ffff;
  if (fs->FCBS[fp].size < size) {
    printf("access size larger than the actul size of the file!!!\n");
    return ;
  }
  read_blocks(fs, get_address(fs->FCBS[fp]), size, output);
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{

  if (!WRITE(fp)) {
    printf("No write permission!\n");
    return ; 
  }
  fp = fp & 0x0000ffff;
  int WD[2];
  get_WD(fs, WD);

  // printf("name%s time: %d\n", fs->FCBS[fp].name, gtime - 1);
  
  int block_need = ceil(size, 32);
  int block_need_old = ceil(fs->FCBS[fp].size, 32);
  if (block_need <= block_need_old) {
    refill_blocks(fs, get_address(fs->FCBS[fp]), fs->FCBS[fp].size, size, input); 
    fs->FCBS[fp].size = size;
  }
  else {
    int address = find_hole(fs->SUPERBLOCK, size);
    if (address != -1) {
      flush_blocks(fs, get_address(fs->FCBS[fp]), block_need_old);
      fill_blocks(fs, address, size, input);
      set_address(&fs->FCBS[fp], address);
      fs->FCBS[fp].size = size;
    }
    else {

      RESET_VALID(fs->FCBS[fp].address);
      compact_blocks(fs);
      SET_VALID(fs->FCBS[fp].address);
      int address = find_hole(fs->SUPERBLOCK, size);
      if (address != -1) {
        flush_blocks(fs, get_address(fs->FCBS[fp]), block_need_old);
        fill_blocks(fs, address, size, input);
        set_address(&fs->FCBS[fp], address);
        fs->FCBS[fp].size = size;
      }
      else {
        printf("no big enough continous space!!!\n");
      }
      return 0;
    }
  }

  if (gtime == 65535) {
    gtime = sort_by_time(fs->FCBS);
  }

  set_modified_time(&fs->FCBS[fp], gtime);
  if (WD[0] != -1) 
  set_modified_time(&fs->FCBS[WD[0]], gtime);
  if (WD[1] != -1)
  set_modified_time(&fs->FCBS[WD[1]], gtime);
  gtime++;
  return 0;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
  if (op == LS_D) {
    FCB valid_fcbs[1024];
    int WD[2];
    get_WD(fs, WD);
    int offset = 0;
    
    if (WD[0] != - 1) {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->FCBS[i].address)) {
          if (PARENT(fs->FCBS[i].address) == WD[0] && !ROOT(fs->FCBS[i].address)) {
            valid_fcbs[offset++] = fs->FCBS[i];
          }
        }
      }
    }
    else {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->FCBS[i].address)) {
          if (ROOT(fs->FCBS[i].address))
            valid_fcbs[offset++] = fs->FCBS[i];
        }
      }
    }

    sort_by_date(valid_fcbs, offset);
    print_array_by_date(valid_fcbs, offset);
  }
  else if (op == LS_S) {
    FCB valid_fcbs[1024];
    int offset = 0;
    int WD[2];
    get_WD(fs, WD);
    if (WD[0] != - 1) {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->FCBS[i].address)) {
          if (PARENT(fs->FCBS[i].address) == WD[0] && !ROOT(fs->FCBS[i].address)) {
            valid_fcbs[offset++] = fs->FCBS[i];
          }
        }
      }
    }
    else {
      for (int i = 0; i < 1024; i++) {
        if (VALID(fs->FCBS[i].address)) {
          if (ROOT(fs->FCBS[i].address))
            valid_fcbs[offset++] = fs->FCBS[i];
        }
      }
    }

    sort_by_size(valid_fcbs, offset);
    print_array_by_size(valid_fcbs, offset);

  }
	else if (op == CD_P) {
    int WD[2];
    get_WD(fs, WD);
    if (WD[0] == -1) {
      printf("Already in root directory!\n");
    }
    else {
      if (WD[1] != -1){
        SET_WD(fs->FCBS[WD[1]].address);
      }
      RESET_WD(fs->FCBS[WD[0]].address);
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
      if (VALID(fs->FCBS[i].address)) {
        if (ROOT(fs->FCBS[i].address)) {
          if (cmp_str(fs->FCBS[i].name, s)) 
          file =i;
        }
      }
    }
  }
  else {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (!ROOT(fs->FCBS[i].address) && PARENT(fs->FCBS[i].address) == WD[0]) {
          if (cmp_str(fs->FCBS[i].name, s)) 
            file = i;
        }
      }
    }
  }


    if (file == - 1) {
      printf("No such file to delete!!\n");
    }
    else if (DIR(fs->SUPERBLOCK[file])) {
      printf("It is a directory!\n");
    }
    else {
      if (!ROOT(fs->FCBS[file].address)) 
      fs->FCBS[WD[0]].size -= get_len(s);
      if (gtime == 65535) {
        gtime = sort_by_time(fs->FCBS);
      }
      set_modified_time(&fs->FCBS[file], gtime);
      if (WD[0] != -1) 
      set_modified_time(&fs->FCBS[WD[0]], gtime);
      if (WD[1] != -1)
      set_modified_time(&fs->FCBS[WD[1]], gtime);
      gtime++;
      flush_blocks(fs, get_address(fs->FCBS[file]), ceil(fs->FCBS[file].size, 32));
      RESET_VALID(fs->FCBS[file].address);
    }
  }
  else if (op == MKDIR) {
    int WD[2];
    get_WD(fs, WD);
    int empty_block = -1;
    if (WD[0] == -1) {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (ROOT(fs->FCBS[i].address)) {
        if (DIR(fs->FCBS[i].address)) {
          if (cmp_str(fs->FCBS[i].name, s)) {
            printf("The directory already exists!!!\n");
            return ;
          }
        }
      }
      }
      else {
        empty_block = i;
      }
    }
  }
  else {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (!ROOT(fs->FCBS[i].address) && PARENT(fs->FCBS[i].address) == WD[0]) {
          if (DIR(fs->FCBS[i].address)) {
          if (cmp_str(fs->FCBS[i].name, s)) {
            printf("The directory already exists!!!\n");
            return ;
          }
        }
      }
      }
      else {
        empty_block = i;
      }
    }
  }

    if  (empty_block == -1) {
      printf("The file number reaches the limit!!!\n");
      return ;
    }
      if (gtime == 65535) {
        gtime = sort_by_time(fs->FCBS);
      }

      // empty not anymore
      RESET(fs->FCBS[empty_block].address);
      SET_VALID(fs->FCBS[empty_block].address);
      copy_str(s, fs->FCBS[empty_block].name);
      SET_DIR(fs->FCBS[empty_block].address);
      set_create_time(&fs->FCBS[empty_block], gtime);
      set_modified_time(&fs->FCBS[empty_block], gtime);
      fs->FCBS[empty_block].size = 0;
      if (WD[0] != -1) {
        SET_PARENT(fs->FCBS[empty_block].address, WD[0]);
        fs->FCBS[WD[0]].size += get_len(s);
      }
      else {
        SET_ROOT(fs->FCBS[empty_block].address);
      }
      
      if (WD[0] != -1) 
        set_modified_time(&fs->FCBS[WD[0]], gtime);
      if (WD[1] != -1)
        set_modified_time(&fs->FCBS[WD[1]], gtime);
      gtime++;
  }
  else if (op == CD) {
    int WD[2];
    int block = -1;
    get_WD(fs, WD);
    if (WD[0] == -1) {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (ROOT(fs->FCBS[i].address)) {
        if (DIR(fs->FCBS[i].address)) {
          if (cmp_str(fs->FCBS[i].name, s)) {
            block = i;
          }
        }
      }
      }
    }
  }
  else {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (!ROOT(fs->FCBS[i].address) && PARENT(fs->FCBS[i].address) == WD[0]) {
          if (DIR(fs->FCBS[i].address)) {
          if (cmp_str(fs->FCBS[i].name, s)) {
            block = i;
          }
        }
      }
      }
    }
  }


  //  for (int i = 0; i < 1024; i++) {
  //     if (VALID(fs->FCBS[i].address)) {
  //       if (cmp_str(fs->FCBS[i].name, s)) {
  //         block = i;
  //         break;
  //       }
  //     }
  //   }
 
  
  if (block == -1) {
      printf("No such directory!\n");
    }
    else {
        if (WD[0] != -1) {
          RESET_WD(fs->FCBS[WD[0]].address);
        }
        SET_WD(fs->FCBS[block].address);
    } 
  }
  else if (op == RM_RF) {
    int WD[2];
    get_WD(fs, WD);
    int file = -1;
    if (WD[0] == -1) {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (ROOT(fs->FCBS[i].address)) {  
          if (DIR(fs->FCBS[i].address)) {
          if (cmp_str(fs->FCBS[i].name, s)) 
          file = i;
        }
        }
      }
    }
  }
  else {
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->FCBS[i].address)) {
        if (!ROOT(fs->FCBS[i].address) && PARENT(fs->FCBS[i].address) == WD[0]) {
          if (DIR(fs->FCBS[i].address)) {
          if (cmp_str(fs->FCBS[i].name, s)) 
            file = i;
        }
        }
      }
    }
  }
 
    if (file == - 1) {
      printf("No such directory to delete\n");
    }
    else {
      if (WD[0] != -1)
      set_modified_time(&fs->FCBS[WD[0]], gtime);
      if (WD[1] != -1)
      set_modified_time(&fs->FCBS[WD[1]], gtime);
      rm_DIR(fs, file);
      gtime++;
    }
       
  }

}