#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>  
#include <string.h>

int create_and_write_txt(char data[])
{
    //下面是写数据，将写入到data.txt文件中  
    //char data[] = "世界如此多娇！";
    FILE* fpWrite = fopen("data.txt", "w");
    if (fpWrite == NULL)
    {
        return 0;
    }
    fprintf(fpWrite, "%s", data);
    fclose(fpWrite);
    return 0;
}
