#ifndef UTILS
#define UTILS

__device__ int get_len(char * s) {
  for (int i = 0; i < 20; i++) {
    if (s[i] == '\0') {
      return i + 1;
    }
  }
  return 20;
}
__device__ void get_WD(FileSystem * fs, int *WD) {
  for (int i =0; i < 1024; i++) {
    if (VALID(fs->SUPERBLOCK[i])) {
      if (WD(fs->SUPERBLOCK[i])) {
        WD[0] = i;
        if (!ROOT(fs->SUPERBLOCK[i])) {
          WD[1] = PARENT(fs->SUPERBLOCK[i]);
        }
        else {
          WD[1] = -1;
        }
        return ; 
      }
    }
  }
  WD[0] = -1;
  WD[1] = -1;
  return ;
}
__device__ void swap(FCB* xp, FCB* yp)
{
    FCB temp = *xp;
    *xp = *yp;
    *yp = temp;
}
 
__device__ int get_address(FCB fcb) {
  return fcb.address_size >> 16;
}

__device__ int get_size(FCB fcb) {
  return fcb.address_size & 0xffff;
}

__device__ void set_address(FCB *fcb, int address) { 
  fcb->address_size &= 0x0000ffff; 
  fcb->address_size |= (address << 16);
}

__device__ void set_size(FCB *fcb, int size) {
  fcb->address_size &= 0xffff0000;
  fcb->address_size |= size;
}

__device__ bool cmp_str(char* a, char* b) {

  int i = 0;
  do {
    if (a[i] != b[i]) 
      return 0;
  }
  while (a[i++] != '\0');
  return 1;
}

__device__ void copy_str(char* ori, char* dst) {
  int i;
  for (i = 0; ori[i] != '\0'; i++) {
    dst[i] = ori[i];
  }
  dst[i] = '\0';
}

__device__ void print_array_by_date(FileSystem *fs, FCB *fcbs, int len) {
  printf("===sort by modified time===\n");
  for (int i = 0; i < len; i++) {
    if (DIR(fs->SUPERBLOCK[get_address(fcbs[i])])) {
      printf("%s d\n", fcbs[i].name);
    }
    else {
      printf("%s\n", fcbs[i].name);
    } 
  }
}


__device__ void print_array_by_size(FileSystem *fs, FCB *fcbs, int len) {
  printf("===sort by file size===\n");
  for (int i = 0; i < len; i++) {
    if (DIR(fs->SUPERBLOCK[get_address(fcbs[i])])) {
      printf("%s %d d\n", fcbs[i].name, get_size(fcbs[i]));
    }
    else {
      printf("%s %d\n", fcbs[i].name, get_size(fcbs[i]));
    }
  }
}

__device__ void sort_by_size(FCB* fcbs, int n)
{
    int i, j, max_idx;
 
    // One by one move boundary of unsorted subarray
    for (i = 0; i < n - 1; i++) {
 
        // Find the minimum element in unsorted array
        max_idx = i;
        for (j = i + 1; j < n; j++)
            if (get_size(fcbs[j]) > get_size(fcbs[max_idx]) || (get_size(fcbs[j]) == get_size(fcbs[max_idx]) && fcbs[j].create_time < fcbs[max_idx].create_time))
                max_idx = j;
 
        // Swap the found minimum element
        // with the first element
        swap(&fcbs[max_idx], &fcbs[i]);
    }
}

 __device__ void sort_by_date(FCB* fcbs, int n)
{
    int i, j, max_idx;
 
    // One by one move boundary of unsorted subarray
    for (i = 0; i < n - 1; i++) {
 
        // Find the minimum element in unsorted array
        max_idx = i;
        for (j = i + 1; j < n; j++)
            if (fcbs[j].modified_time > fcbs[max_idx].modified_time)
                max_idx = j;
 
        // Swap the found minimum element
        // with the first element
        swap(&fcbs[max_idx], &fcbs[i]);
    }
}
__device__ void rm_DIR(FileSystem *fs, int block) {
  int queue[1024];
  for (int i = 0; i < 1024; i++) {
    queue[i] = -1;
  }
  int ptr = 0;
  int cur_ptr = 0;
  int cur_idx = -1;
  queue[ptr++] = block; 
  while (queue[cur_ptr] != -1) {
    cur_idx = queue[cur_ptr++];
    for (int i = 0; i < 1024; i++) {
      if (VALID(fs->SUPERBLOCK[i])) {
        if (!ROOT(fs->SUPERBLOCK[i])) {
          if (PARENT(fs->SUPERBLOCK[i]) == cur_idx) {
            if (DIR(fs->SUPERBLOCK[i])) {
              queue[ptr++] = i;
            }
            else {
              for (int j = 0; j < 1024; j++) {
                fs->FILES[get_address(fs->FCBS[i])][j] = '\0';
              }
              RESET_VALID(fs->SUPERBLOCK[i]);
            }
          }
        }
      }
    }
    for (int j = 0; j < 1024; j++) {
      fs->FILES[get_address(fs->FCBS[cur_idx])][j] = '\0';
    }
    RESET_VALID(fs->SUPERBLOCK[cur_idx]);
    if (!ROOT(fs->SUPERBLOCK[cur_idx])) 
    set_size(&fs->FCBS[PARENT(fs->SUPERBLOCK[cur_idx])], get_size(fs->FCBS[PARENT(fs->SUPERBLOCK[cur_idx])]) - get_len(fs->FCBS[cur_idx].name));

  }
}



#endif