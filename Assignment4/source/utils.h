#ifndef UTILS
#define UTILS

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
#endif