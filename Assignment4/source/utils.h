#ifndef UTILS
#define UTILS
__device__ int is_filled(int * superblock, int size) {
  int x = size / 32;
  int y = size % 32;
  return superblock[x] & (1<<y); 
}

__device__ int ceil(int x, int y) {
  if (x % y) {
    return x / y + 1;
  }
  return x / y;
}

__device__ void swap(FCB* xp, FCB* yp)
{
    FCB temp = *xp;
    *xp = *yp;
    *yp = temp;
}

__device__ int get_size(FCB fcb) {
  return fcb.size;
}

__device__ int get_address(FCB fcb) {
  return fcb.address & 0xffff;
}
 
__device__ int get_create_time(FCB fcb) {
  return fcb.create_modified >> 16;
}

__device__ int get_modified_time(FCB fcb) {
  return fcb.create_modified & 0xffff;
}

__device__ int set_address(FCB *fcb, int address) {
  fcb->address &= 0xffff0000;
  fcb->address |= address;
}

__device__ void set_create_time(FCB *fcb, int create_time) { 
  fcb->create_modified &= 0x0000ffff; 
  fcb->create_modified |= (create_time << 16);
}

__device__ void set_modified_time(FCB *fcb, int modified_time) {
  fcb->create_modified &= 0xffff0000;
  fcb->create_modified |= modified_time;
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

__device__ int find_hole(int *superblock, int size) {
  int block_num = ceil(size, 32);
  int j;
  for (int i = 0; i < 32768; i++) {
    if (!is_filled(superblock, i)) {
      if (32768 - i < size) {
        break;
      }
      for (j = 0; j < size; j++) {
        if (is_filled(superblock ,j)) {
          break;
        }
      }
      if (j == size) {
        return i;
      }
      else {
        i = MIN(32767 ,j + 1);
      }
    }
  }
} 

__device__ int sort_by_time(FCB *fcbs) {
  int i, j, min_idx;
  FCB valid_fcbs[1024];
  int n = 0;
  for (i = 0; i < 1024; i++) {
    if (VALID(fcbs[i].address)) {
      valid_fcbs[n++] = fcbs[i];
    }
  }
    // One by one move boundary of unsorted subarray
    for (i = 0; i < n - 1; i++) {
 
        // Find the minimum element in unsorted array
        min_idx = i;
        for (j = i + 1; j < n; j++)
            if (get_modified_time(valid_fcbs[j]) < get_modified_time(valid_fcbs[min_idx]))
                min_idx = j;
 
        // Swap the found minimum element
        // with the first element
        swap(&valid_fcbs[min_idx], &valid_fcbs[i]);
    }

    for (i = 0; i < n; i++) {
      for (j = 0; j < 1024; j++) {
        if (VALID(fcbs[i].address) && cmp_str(valid_fcbs[i].name, fcbs[j].name)) {
          set_modified_time(&fcbs[j], i);
        }
      }
    }

    for (i = 0; i < n - 1; i++) {
 
        // Find the minimum element in unsorted array
        min_idx = i;
        for (j = i + 1; j < n; j++)
            if (get_create_time(valid_fcbs[j]) < get_create_time(valid_fcbs[min_idx]))
                min_idx = j;
 
        // Swap the found minimum element
        // with the first element
        swap(&valid_fcbs[min_idx], &valid_fcbs[i]);
    }

    for (i = 0; i < n; i++) {
      for (j = 0; j < 1024; j++) {
        if (VALID(fcbs[i].address) && cmp_str(valid_fcbs[i].name, fcbs[j].name)) {
          set_create_time(&fcbs[i], i);
        }
      }
    }

    return n;
}


__device__ void print_array_by_date(FCB *fcbs, int len) {
  printf("===sort by modified time===\n");
  for (int i = 0; i < len; i++) {
    printf("%s\n", fcbs[i].name);
  }
}


__device__ void print_array_by_size(FCB *fcbs, int len) {
  printf("===sort by file size===\n");
  for (int i = 0; i < len; i++) {
    printf("%s %d\n", fcbs[i].name, get_size(fcbs[i]));
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
            if (get_size(fcbs[j]) > get_size(fcbs[max_idx]) || (get_size(fcbs[j]) == get_size(fcbs[max_idx]) && get_create_time(fcbs[j]) < get_create_time(fcbs[max_idx])))
            {
                max_idx = j;
            }
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
            if (get_modified_time(fcbs[j]) > get_modified_time(fcbs[max_idx]))
                max_idx = j;
 
        // Swap the found minimum element
        // with the first element
        swap(&fcbs[max_idx], &fcbs[i]);
    }
}


#endif