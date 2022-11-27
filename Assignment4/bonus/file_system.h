#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2
#define MKDIR 3
#define CD 4
#define CD_P 5
#define RM_RF 6
#define PWD 7

#define RESET(x) ((x) = (0))

#define VALID(x) (x & 0x10000)
#define SET_VALID(x) ((x) |= (0x10000))
#define RESET_VALID(x) ((x) &= (0xfffeffff))

#define DIR(x) ((x & 0x20000))
#define SET_DIR(x) ((x) |= (0x20000))
#define RESET_DIR(x) ((x) &= (0xfffdffff))

#define	WD(x) (x & 0x40000) 
#define SET_WD(x) ((x) |= (0x40000))
#define RESET_WD(x) ((x) &= (0xfffbffff))

#define ROOT(x) (x & 0x80000)
#define SET_ROOT(x) ((x) |= (0x80000))
#define RESET_ROOT(x) ((x) &= (0xfff7ffff))

#define WRITE(x) ((x) & (0x10000000))
#define SET_WRITE(x) ((x) |= (0x10000000))
#define RESET_WRITE(x) ((x) &= (0xefffffff))

#define READ(x) ((x) & (0x20000000))
#define SET_READ(x) ((x) |= (0x20000000))
#define RESET_READ(x) ((x) &= (0xdfffffff))

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define PARENT(x) ((unsigned int)x >>22)
#define SET_PARENT(x, p) {\
	(x) &= 0x003fffff;\
	(x) |= (p) << 22;\
}

typedef struct FCB {
	char name[20];
	int size;//0-15 size, 16-31 addr
	int create_modified;
	int address;	
} FCB;

struct FileSystem {
	int SUPERBLOCK[1024];
	FCB FCBS[1024];
	uchar BLOCKS[32768][32];

	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
};


__device__ void fs_init(FileSystem *fs, int SUPERBLOCK_SIZE,
	int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
	int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
	int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);


#endif