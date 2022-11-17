#ifndef UTILS
#define UTILS

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
  if (create_time >= 1024) {
    printf("overflow!");
  }
  fcb->create_modified &= 0x0000ffff; 
  fcb->create_modified |= (create_time << 16);
}

__device__ void set_modified_time(FCB *fcb, int modified_time) {

  if (modified_time >= 1024) {
    printf("overflow!!!");
  }
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