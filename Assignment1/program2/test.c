#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

int main(int argc,char* argv[]){
	int i=0;
	FILE * fPtr;
	printf("--------USER PROGRAM--------\n");
	fPtr = fopen("test.txt", "w");
	fclose(fPtr);
//	alarm(2);
	raise(SIGBUS);
	sleep(5);
	printf("user process success!!\n");
	printf("--------USER PROGRAM--------\n");
	return 100;
}
