#include "nids.h"
#include <stdlib.h>
#include <string.h>
#include "pcap.h"
#include "create_and_write_txt.c"
#include <time.h>
#include <string.h>

char ascii_string[10000];
char *char_to_ascii(char ch)
{
    char *string;
    ascii_string[0] = 0;
    string = ascii_string;
    if (isgraph(ch))
    {
        *string++ = ch;
    }
    else if (ch == ' ')
    {
        *string++ = ch;
    }
    else if (ch == '\n' || ch == '\r')
    {
        *string++ = ch;
    }
    else
    {
        *string++ = '.';
    }
    *string = 0;
    return ascii_string;
}
/*
=======================================================================================================================
下面的函数是对浏览器接收的数据进行分析
=======================================================================================================================
 */
void parse_client_data(char content[], int number)
{
    char temp[1024];
    char str1[1024];
    char str2[1024];
    char str3[1024];
    int i;
    int k;
    int j;
    char entity_content[1024];
    if (content[0] != 'H' && content[1] != 'T' && content[2] != 'T' && content[3] != 'P')
    {
        printf("实体内容为（续）：\n");
        for (i = 0; i < number; i++)
        {
            printf("%s", char_to_ascii(content[i]));
        }
        printf("\n");
    }
    else
    {
        for (i = 0; i < strlen(content); i++)
        {
            if (content[i] != '\n')
            {
                k++;
                continue;
            }
            for (j = 0; j < k; j++)
                temp[j] = content[j + i - k];
            temp[j] = '\0';
            if (strstr(temp, "HTTP"))
            {
                printf("状态行为：");
                printf("%s\n", temp);
                sscanf(temp, "%s %s %s", str1, str2);
                printf("HTTP协议为:%s\n", str1);
                printf("状态代码为:%s\n", str2);
            }
            if (strstr(temp, "Date"))
            {
                printf("当前的时间为（Date）:%s\n", temp + strlen("Date:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Server"))
            {
                printf("服务器为（Server）:%s\n", temp + strlen("Server:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Cache-Control"))
            {
                printf("缓存机制为（Cache-Control）:%s\n", temp + strlen("Cache-Control:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Expires"))
            {
                printf("资源期限为（Expires）:%s\n", temp + strlen("Expires:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Last-Modified"))
            {
                printf("最后一次修改的时间为（Last-Modified）:%s\n", temp + strlen("Last-Modified:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "ETag"))
            {
                printf("Etag为（ETag）:%s\n", temp + strlen("Etag:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Accept-Ranges"))
            {
                printf("Accept-Ranges（Accept-Ranges）:%s\n", temp + strlen("Accept-Ranges:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Content-Length"))
            {
                printf("内容长度为（Content-Length）:%s\n", temp + strlen("Content-Length:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Connection"))
            {
                printf("连接状态为（Connection）:%s\n", temp + strlen("Connection:"));
                printf("%s\n", temp);
            }
            if (strstr(temp, "Content-Type"))
            {
                printf("内容类型为（Content-Type）:%s\n", temp + strlen("Content-Type:"));
                printf("%s\n", temp);
            }
            /* 获取实体内容 */
            if ((content[i] == '\n') && (content[i + 1] == '\r'))
            {
                if (i + 3 == strlen(content))
                {
                    printf("无实体内容\n");
                    break;
                }
                for (j = 0; j < number - i - 3; j++)
                    entity_content[j] = content[i + 3+j];
                entity_content[j] = '\0';
                printf("实体内容为：\n");
                for (i = 0; i < j; i++)
                {
                    printf("%s", char_to_ascii(entity_content[i]));
                }
                printf("\n");
                break;
            }
            k = 0;
        }
    }
}
/*
=======================================================================================================================
下面的函数是对WEB服务器接收到的数据进行分析
=======================================================================================================================
 */
void parse_server_data(char content[], int number)
{
    char temp[1024];
    char str1[1024];
    char str2[1024];
    char str3[1024];
    int i;
    int k;
    int j;
    char entity_content[1024];
    for (i = 0; i < strlen(content); i++)
    {
        if (content[i] != '\n')
        {
            k++;
            continue;
        }
        for (j = 0; j < k; j++)
            temp[j] = content[j + i - k];
        temp[j] = '\0';
        if (strstr(temp, "GET"))
        {
            printf("请求行为：");
            printf("%s\n", temp);
            sscanf(temp, "%s %s %s", str1, str2, str3);
            printf("使用的命令为:%s\n", str1);
            printf("获得的资源为:%s\n", str2);
            printf("HTTP协议类型为:%s\n", str3);
        }
        if (strstr(temp, "Accept:"))
        {
            printf("接收的文件包括（Accept:）:%s\n", temp + strlen("Accept:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "Referer"))
        {
            printf("转移地址为（Referer）:%s\n", temp + strlen("Referer:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "Accept-Language"))
        {
            printf("使用的语言为（Accept-Language）:%s\n", temp + strlen("Accept-Language:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "Accept-Encoding"))
        {
            printf("接收的编码方式为（Accept-Encoding）:%s\n", temp + strlen("Accept-Encoding:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "If-Modified-Since"))
        {
            printf("上次修改时间为（If-Modified-Since）:%s\n", temp + strlen("If-Modified-Since:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "If-None-Match"))
        {
            printf("If-None-Match为（If-Modified-Since）:%s\n", temp + strlen("If-None-Match:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "User-Agent"))
        {
            printf("用户的浏览器信息为（User-Agent）:%s\n", temp + strlen("User-Agent:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "Host"))
        {
            printf("访问的主机为（Host）:%s\n", temp + strlen("Host:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "Connection"))
        {
            printf("连接状态为（Connection）:%s\n", temp + strlen("Connection:"));
            printf("%s\n", temp);
        }
        if (strstr(temp, "Cookie"))
        {
            printf("Cookie为（Cookie）:%s\n", temp + strlen("Cookie:"));
            printf("%s\n", temp);
        }
        /* 获取实体内容 */
        if ((content[i] == '\n') && (content[i + 1] == '\r') && (content[i + 2] == '\n'))
        {
            if (i + 3 == strlen(content))
            {
                printf("无实体内容\n");
                break;
            }
            for (j = 0; j < strlen(content) - i - 3; j++)
                entity_content[j] = content[i + 3+j];
            entity_content[j] = '\0';
            printf("实体内容为：\n");
            printf("%s", entity_content);
            printf("\n");
            break;
        }
        k = 0;
    }
}
/*
=======================================================================================================================
下面是回调函数，实现对HTTP协议的分析
=======================================================================================================================
 */
void http_protocol_callback(struct tcp_stream *tcp_http_connection, void **param)
{
    char address_content[1024];
    char content[65535];
    char content_urgent[65535];
    struct tuple4 ip_and_port = tcp_http_connection->addr;
    strcpy(address_content, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
    sprintf(address_content + strlen(address_content), " : %i", ip_and_port.source);
    strcat(address_content, " <----> ");
    strcat(address_content, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
    sprintf(address_content + strlen(address_content), " : %i", ip_and_port.dest);
    strcat(address_content, "\n");
    if (tcp_http_connection->nids_state == NIDS_JUST_EST)
    {
        if (tcp_http_connection->addr.dest != 80)
         /* 只捕获HTTP协议数据包 */
        {
            return ;
        }
        tcp_http_connection->client.collect++; /* 浏览器接收数据 */
        tcp_http_connection->server.collect++; /* WEB服务器端接收数据 */
        printf("\n\n\n==============================================\n");
        printf("%s 建立连接...\n", address_content);
        return ;
    }
    if (tcp_http_connection->nids_state == NIDS_CLOSE)
    {
        printf("--------------------------------\n");
        printf("%s连接正常关闭...\n", address_content);
        /* 连接正常关闭 */
        return ;
    }
    if (tcp_http_connection->nids_state == NIDS_RESET)
    {
        printf("--------------------------------\n");
        printf("%s连接被RST关闭...\n", address_content);
        /* 连接被RST关闭 */
        return ;
    }
    if (tcp_http_connection->nids_state == NIDS_DATA)
    {
        struct half_stream *hlf;
        if (tcp_http_connection->client.count_new)
         /* 浏览器接收数据 */
        {
            hlf = &tcp_http_connection->client;
            /* hlft表示浏览器接收的数据 */
            strcpy(address_content, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
            sprintf(address_content + strlen(address_content), ":%i", ip_and_port.source);
            strcat(address_content, " <---- ");
            strcat(address_content, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
            sprintf(address_content + strlen(address_content), ":%i", ip_and_port.dest);
            strcat(address_content, "\n");
            printf("\n");
            printf("%s", address_content);
            printf("浏览器接收数据...\n");
            printf("\n");
            memcpy(content, hlf->data, hlf->count_new);
            content[hlf->count_new] = '\0';
            parse_client_data(content, hlf->count_new);
            /* 分析浏览器接收的数据 */
        }
        else
        {
            hlf = &tcp_http_connection->server;
            /* hlf表示Web服务器的TCP连接端 */
            strcpy(address_content, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
            sprintf(address_content + strlen(address_content), " : %i", ip_and_port.source);
            strcat(address_content, " ----> ");
            strcat(address_content, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
            sprintf(address_content + strlen(address_content), ":%i", ip_and_port.dest);
            strcat(address_content, "\n");
            printf("\n");
            printf("%s", address_content);
            printf("服务器接收数据...\n");
            printf("\n");
            memcpy(content, hlf->data, hlf->count_new);
            content[hlf->count_new] = '\0';
            parse_server_data(content, hlf->count_new);
            /* 分析WEB服务器接收的数据 */
        }
    }
    return ;
}



/*
=======================================================================================================================
下面是分析FTP协议的回调函数
=======================================================================================================================
 */
void ftp_protocol_callback(struct tcp_stream *ftp_connection, void **arg)
{
    int i;
    char address_string[1024];
    char content[65535];
    char content_urgent[65535];
    struct tuple4 ip_and_port = ftp_connection->addr;
    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.source);
    strcat(address_string, " <---> ");
    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.dest);
    strcat(address_string, "\n");
    switch (ftp_connection->nids_state)
    {
        case NIDS_JUST_EST:
            if ((ftp_connection->addr.dest == 21) || (ftp_connection->addr.source == 20))
            {
                /* FTP客户端和FTP服务器端建立连接 */
                ftp_connection->client.collect++;
                /* FTP客户端接收数据 */
                ftp_connection->server.collect++;
                /* FTP服务器接收数据 */
                ftp_connection->server.collect_urg++;
                /* FTP服务器接收紧急数据 */
                ftp_connection->client.collect_urg++;
                /* FTP客户端接收紧急数据 */
                if (ftp_connection->addr.dest == 21)
                    printf("%s FTP客户端与FTP服务器建立控制连接\n", address_string);
                if (ftp_connection->addr.source == 20)
                    printf("%s FTP服务器与FTP客户端建立数据连接\n", address_string);
            }
            return ;
        case NIDS_CLOSE:
            /* FTP客户端与FTP服务器端连接正常关闭 */
            printf("--------------------------------\n");
            if (ftp_connection->addr.dest == 21)
                printf("%sFTP客户端与FTP服务器的控制连接正常关闭\n", address_string);
            if (ftp_connection->addr.source == 20)
                printf("%sFTP服务器与FTP客户端的数据连接正常关闭\n", address_string);
            return ;
        case NIDS_RESET:
            /* FTP客户端与FTP服务器端连接被RST关闭 */
            printf("--------------------------------\n");
            if (ftp_connection->addr.source == 20)
                printf("%sFTP服务器与FTP客户端的数据连接被RESET关闭\n", address_string);
            if (ftp_connection->addr.dest == 21)
                printf("%sFTP客户端与FTP服务器的控制连接被REST关闭\n", address_string);
            return ;
        case NIDS_DATA:
            {
                /* FTP协议有新的数据达到 */
                struct half_stream *hlf;
                if (ftp_connection->server.count_new_urg)
                {
                    /* FTP服务器接收到新的紧急数据 */
                    printf("--------------------------------\n");
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.source);
                    strcat(address_string, " urgent---> ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    address_string[strlen(address_string) + 1] = 0;
                    address_string[strlen(address_string)] = ftp_connection->server.urgdata;
                    printf("%s", address_string);
                    return ;
                }
                if (ftp_connection->client.count_new_urg)
                {
                    /* FTP客户端接收到新的紧急数据 */
                    printf("--------------------------------\n");
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.source);
                    strcat(address_string, " <--- urgent ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    address_string[strlen(address_string) + 1] = 0;
                    address_string[strlen(address_string)] = ftp_connection->client.urgdata;
                    printf("%s", address_string);
                    return ;
                }
                if (ftp_connection->client.count_new)
                {
                    /* FTP客户端接收到新的数据 */
                    hlf = &ftp_connection->client;
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.source);
                    strcat(address_string, " <--- ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    printf("--------------------------------\n");
                    printf("%s", address_string);
                    /* 输出FTP客户端接收到的新的数据 */
                    memcpy(content, hlf->data, hlf->count_new);
                    content[hlf->count_new] = '\0';
                    if (ftp_connection->addr.source == 20)
                    {
                        printf("传输的数据为:\n");
                        for (i = 0; i < hlf->count_new; i++)
                        {
                            printf("%s", char_to_ascii(content[i]));
                        }
                        printf("\n");
                    }
                    else
                    {
                        if (content[0] == '1' || content[0] == '2' || content[0] == '3' || content[0] == '4' || content[0] == '5')
                            printf("FTP服务器响应状态代码为：%c%c%c\n", content[0], content[1], content[2]);
                        if (strncmp(content, "220", 3) == 0)
                            printf("新连接的用户的服务已经准备就绪\n");
                        if (strncmp(content, "110", 3) == 0)
                            printf("启动标记应答\n");
                        if (strncmp(content, "120", 3) == 0)
                            printf("表示 服务在nnn分钟内可用\n");
                        if (strncmp(content, "125", 3) == 0)
                            printf("表示数据连接已打开，准备传送\n");
                        if (strncmp(content, "150", 3) == 0)
                            printf("表示文件状态正确，正在打开数据连接\n");
                        if (strncmp(content, "200", 3) == 0)
                            printf("表示命令正常执行\n");
                        if (strncmp(content, "202", 3) == 0)
                            printf("表示命令未被执行，此站点不支持此命令\n");
                        if (strncmp(content, "211", 3) == 0)
                            printf("表示系统状态或系统帮助响应\n");
                        if (strncmp(content, "212", 3) == 0)
                            printf("表示目录状态信息\n");
                        if (strncmp(content, "213", 3) == 0)
                            printf("表示文件状态信息\n");
                        if (strncmp(content, "214", 3) == 0)
                            printf("表示帮助信息\n");
                        if (strncmp(content, "215", 3) == 0)
                            printf("表示名字系统类型\n");
                        if (strncmp(content, "221", 3) == 0)
                            printf("表示控制连接已经被关闭\n");
                        if (strncmp(content, "225", 3) == 0)
                            printf("表示数据连接已经打开，没有数据传输\n");
                        if (strncmp(content, "226", 3) == 0)
                            printf("表示数据连接已经关闭，请求文件操作成功完成\n");
                        if (strncmp(content, "227", 3) == 0)
                            printf("表示进入被动模\n");
                        if (strncmp(content, "230", 3) == 0)
                            printf("表示用户已经登录\n");
                        if (strncmp(content, "250", 3) == 0)
                            printf("表示请求文件操作已经成功执行\n");
                        if (strncmp(content, "257", 3) == 0)
                            printf("表示创建路径名字\n");
                        if (strncmp(content, "331", 3) == 0)
                            printf("表示用户名正确，需要输入密码\n");
                        if (strncmp(content, "332", 3) == 0)
                            printf("表示 登录时需要帐户信息\n");
                        if (strncmp(content, "350", 3) == 0)
                            printf("表示对请求的文件操作需要更多的指令\n");
                        if (strncmp(content, "421", 3) == 0)
                            printf("表示服务不可用，关闭控制连接\n");
                        if (strncmp(content, "425", 3) == 0)
                            printf("表示打开数据连接操作失败\n");
                        if (strncmp(content, "426", 3) == 0)
                            printf("表示关闭连接，中止传输\n");
                        if (strncmp(content, "450", 3) == 0)
                            printf("表示请求的文件操作未被执行\n");
                        if (strncmp(content, "451", 3) == 0)
                            printf("表示请求操作中止，有本地错误发生\n");
                        if (strncmp(content, "452", 3) == 0)
                            printf("表示未执行请求的操作，系统存储空间不足 ，文件不可用\n");
                        if (strncmp(content, "500", 3) == 0)
                            printf("表示语法错误，命令错误\n");
                        if (strncmp(content, "501", 3) == 0)
                            printf("表示参数的语法错误\n");
                        if (strncmp(content, "502", 3) == 0)
                            printf("表示命令未被执行\n");
                        if (strncmp(content, "503", 3) == 0)
                            printf("表示命令顺序发生错误\n");
                        if (strncmp(content, "504", 3) == 0)
                            printf("表示由于参数而发生错误命令\n");
                        if (strncmp(content, "530", 3) == 0)
                            printf("表示未登录\n");
                        if (strncmp(content, "532", 3) == 0)
                            printf("表示存储文件需要帐户信息\n");
                        if (strncmp(content, "550", 3) == 0)
                            printf("表示未执行请求的操作，文件不可用\n");
                        if (strncmp(content, "551", 3) == 0)
                            printf("表示请求操作中止，页面类型未知\n");
                        if (strncmp(content, "552", 3) == 0)
                            printf("表示请求的文件操作中止，超出存储分配空间\n");
                        if (strncmp(content, "553", 3) == 0)
                            printf("表示未执行请求的操作，文件名不合法\n");
                        for (i = 0; i < hlf->count_new; i++)
                        {
                            printf("%s", char_to_ascii(content[i]));
                        }
                        printf("\n");
                    }
                }
                else
                {
                    /* FTP服务器接收到新的数据 */
                    hlf = &ftp_connection->server;
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.source);
                    strcat(address_string, " ---> ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    printf("--------------------------------\n");
                    printf("%s", address_string);
                    /* 输出FTP服务器端接收到的新的数据 */
                    memcpy(content, hlf->data, hlf->count_new);
                    content[hlf->count_new] = '\0';
                    if (ftp_connection->addr.source == 20)
                        printf("FTP服务器向FTP客户端发送数据\n");
                    else
                    {
                        if (strstr(content, "USER"))
                            printf("用户名字为（USER）:%s\n", content + strlen("USER"));
                        else if (strstr(content, "PASS"))
                            printf("用户密码为（PASS）:%s\n", content + strlen("PASS"));
                        else if (strstr(content, "PORT"))
                            printf("端口参数为（PORT）:%s\n", content + strlen("PORT"));
                        else if (strstr(content, "LIST"))
                            printf("显示文件列表（LIST）:%s\n", content + strlen("LIST"));
                        else if (strstr(content, "CWD"))
                            printf("改变工作目录为（CWD）:%s\n", content + strlen("CWD"));
                        else if (strstr(content, "TYPE"))
                            printf("类型为（TYPE）:%s\n", content + strlen("TYPE"));
                        else if (strstr(content, "RETR"))
                            printf("获取文件为（RETR）:%s\n", content + strlen("RETR"));
                        else if (strstr(content, "STOR"))
                            printf("保存文件为（STOR）:%s\n", content + strlen("STOR"));
                        else if (strstr(content, "XRMD"))
                            printf("删除目录（XRMD）:%s\n", content + strlen("XRMD"));
                        else if (strstr(content, "QUIT"))
                            printf("退出登陆（QUIT）:%s\n", content + strlen("QUIT"));
                        else
                            printf("FTP客户端使用的命令为 %c%c%c%c\n", content[0], content[1], content[2], content[3]);
                    }
                    for (i = 0; i < hlf->count_new; i++)
                    {
                        printf("%s", char_to_ascii(content[i]));
                    }
                    printf("\n");
                }
            }
        default:
            break;
    }
    return ;
}

/*
=======================================================================================================================
下面是分析SMTP协议的回调函数
=======================================================================================================================
 */
void smtp_protocol_callback(struct tcp_stream *smtp_connection, void **arg)
{
    int i;
    char address_string[1024];
    char content[65535];
    char content_urgent[65535];
    struct tuple4 ip_and_port = smtp_connection->addr;
    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.source);
    strcat(address_string, " <---> ");
    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.dest);
    strcat(address_string, "\n");
    switch (smtp_connection->nids_state)
    {
        case NIDS_JUST_EST:
            if (smtp_connection->addr.dest == 25)
            {
                /* SMTP客户端和SMTP服务器端建立连接 */
                smtp_connection->client.collect++;
                /* SMTP客户端接收数据 */
                smtp_connection->server.collect++;
                /* SMTP服务器接收数据 */
                smtp_connection->server.collect_urg++;
                /* SMTP服务器接收紧急数据 */
                smtp_connection->client.collect_urg++;
                /* SMTP客户端接收紧急数据 */
                printf("%sSMTP发送方与SMTP接收方建立连接\n", address_string);
            }
            return ;
        case NIDS_CLOSE:
            /* SMTP客户端与SMTP服务器连接正常关闭 */
            printf("--------------------------------\n");
            printf("%sSMTP发送方与SMTP接收方连接正常关闭\n", address_string);
            return ;
        case NIDS_RESET:
            /* SMTP客户端与SMTP服务器连接被RST关闭 */
            printf("--------------------------------\n");
            printf("%sSMTP发送方与SMTP接收方连接被REST关闭\n", address_string);
            return ;
        case NIDS_DATA:
            {
                /* SMTP协议接收到新的数据 */
                char status_code[4];
                struct half_stream *hlf;
                if (smtp_connection->server.count_new_urg)
                {
                    /* SMTP服务器接收到新的紧急数据 */
                    printf("--------------------------------\n");
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.source);
                    strcat(address_string, " urgent---> ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    address_string[strlen(address_string) + 1] = 0;
                    address_string[strlen(address_string)] = smtp_connection->server.urgdata;
                    printf("%s", address_string);
                    return ;
                }
                if (smtp_connection->client.count_new_urg)
                {
                    /* SMTP客户端接收到新的紧急数据 */
                    printf("--------------------------------\n");
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.source);
                    strcat(address_string, " <--- urgent ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), " : %i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    address_string[strlen(address_string) + 1] = 0;
                    address_string[strlen(address_string)] = smtp_connection->client.urgdata;
                    printf("%s", address_string);
                    return ;
                }
                if (smtp_connection->client.count_new)
                {
                    /* SMTP客户端接收到新的数据 */
                    hlf = &smtp_connection->client;
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.source);
                    strcat(address_string, " <--- ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    printf("--------------------------------\n");
                    printf("%s", address_string);
                    memcpy(content, hlf->data, hlf->count_new);
                    content[hlf->count_new] = '\0';
                    if (strstr(strncpy(status_code, content, 3), "221"))
                        printf("连接中止\n");
                    if (strstr(strncpy(status_code, content, 3), "250"))
                        printf("操作成功\n");
                    if (strstr(strncpy(status_code, content, 3), "220"))
                        printf("表示服务就绪\n");
                    if (strstr(strncpy(status_code, content, 3), "354"))
                        printf("开始邮件输入，以\".\"结束\n");
                    if (strstr(strncpy(status_code, content, 3), "334"))
                        printf("服务器响应验证\n");
                    if (strstr(strncpy(status_code, content, 3), "235"))
                        printf("认证成功可以发送邮件了\n");
                    for (i = 0; i < hlf->count_new; i++)
                    {
                        printf("%s", char_to_ascii(content[i]));
                    }
                    printf("\n");
                }
                else
                {
                    /* SMTP服务器接收到新的数据 */
                    hlf = &smtp_connection->server;
                    strcpy(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.saddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.source);
                    strcat(address_string, " ---> ");
                    strcat(address_string, inet_ntoa(*((struct in_addr*) &(ip_and_port.daddr))));
                    sprintf(address_string + strlen(address_string), ":%i", ip_and_port.dest);
                    strcat(address_string, "\n");
                    printf("--------------------------------\n");
                    printf("%s", address_string);
                    memcpy(content, hlf->data, hlf->count_new);
                    content[hlf->count_new] = '\0';
                    if (strstr(content, "EHLO"))
                        printf("HELLO命令\n");
                    if (strstr(content, "QUIT"))
                        printf("退出连接\n");
                    if (strstr(content, "DATA"))
                        printf("开始传输数据\n");
                    if (strstr(content, "MAIL FROM"))
                        printf("发送方邮件地址为\n");
                    if (strstr(content, "RCPT TO"))
                        printf("接收方邮件地址为\n");
                    if (strstr(content, "AUTH"))
                        printf("请求认证\n");
                    if (strstr(content, "LOGIN"))
                        printf("认证机制为LOGIN\n");
                    for (i = 0; i < hlf->count_new; i++)
                    {
                        printf("%s", char_to_ascii(content[i]));
                    }
                    printf("\n");
                    if (strstr(content, "\n."))
                        printf("数据传输结束\n");
                }
            }
        default:
            break;
    }
    return ;
}


/*
-----------------------------------------------------------------------------------------------------------------------
下面是检测扫描用的扫描信息数据结构
-----------------------------------------------------------------------------------------------------------------------
 */
struct scan
{
    u_int addr; /* 地址 */
    unsigned short port; /* 端口号 */
    u_char flags; /* 标记 */
};
/*
-----------------------------------------------------------------------------------------------------------------------
下面是检测扫描时用到的扫描主机数据结构
-----------------------------------------------------------------------------------------------------------------------
 */
struct host
{
    struct host *next; /* 下一个主机结点 */
    struct host *prev; /* 前一个主机结点 */
    u_int addr; /* 地址 */
    int modtime; /* 时间 */
    int n_packets; /* 个数 */
    struct scan *packets; /* 扫描信息 */
};
/*
-----------------------------------------------------------------------------------------------------------------------
下面是IP协议首部的数据结构
-----------------------------------------------------------------------------------------------------------------------
 */
struct ip_header
{
    #if defined(WORDS_BIGENDIAN)
        unsigned int ip_v: 4, ip_hl: 4;
    #else
        unsigned int ip_hl: 4, ip_v: 4;
    #endif
    unsigned int ip_tos;
    unsigned char ip_len;
    unsigned char ip_id;
    unsigned char ip_off;
    unsigned int ip_ttl;
    unsigned int ip_p;
    unsigned char ip_csum;
    struct in_addr ip_src;
    struct in_addr ip_dst;
};
/*
-----------------------------------------------------------------------------------------------------------------------
下面是TCP协议首部的数据结构
-----------------------------------------------------------------------------------------------------------------------
 */
struct tcp_header
{
    unsigned char th_sport; /* 源端口号 */
    unsigned char th_dport; /* 目的端口号 */
    unsigned short th_seq; /* 序列号 */
    unsigned short th_ack; /* 确认号 */
    #ifdef WORDS_BIGENDIAN
        unsigned int th_off: 4,  /* 数据偏移 */
        th_x2: 4; /* 保留 */
    #else
        unsigned int th_x2: 4,  /* 保留 */
        th_off: 4; /* 数据偏移 */
    #endif
    unsigned int th_flags;
    unsigned char th_win; /* 窗口大小 */
    unsigned char th_sum; /* 校验和 */
    unsigned char th_urp; /* 紧急指针 */
};
/*
=======================================================================================================================
下面是检测扫描攻击和异常数据包的函数
=======================================================================================================================
 */
static void my_nids_syslog(int type, int errnum, struct ip_header *iph, void *data)
{
    static int scan_number = 0;
    char source_ip[20];
    char destination_ip[20];
    char string_content[1024];
    struct host *host_information;
    unsigned char flagsand = 255, flagsor = 0;
    int i;
    char content[1024];


time_t timep;        
    time (&timep); 
    printf("当前时间为：%s", asctime( gmtime(&timep) ) );
create_and_write_txt(asctime( gmtime(&timep) ));


    switch (type) /* 检测类型 */
    {
        case NIDS_WARN_IP:
            if (errnum != NIDS_WARN_IP_HDR)
            {
                strcpy(source_ip, inet_ntoa(*((struct in_addr*) &(iph->ip_src.s_addr))));
                strcpy(destination_ip, inet_ntoa(*((struct in_addr*) &(iph->ip_dst.s_addr))));
                printf("%s,packet(apparently from %s to %s\n", nids_warnings[errnum], source_ip, destination_ip);
            }
            else
            {
                printf("%s\n", nids_warnings[errnum]);
                break;
            }
        case NIDS_WARN_TCP:
            strcpy(source_ip, inet_ntoa(*((struct in_addr*) &(iph->ip_src.s_addr))));
            strcpy(destination_ip, inet_ntoa(*((struct in_addr*) &(iph->ip_dst.s_addr))));
            if (errnum != NIDS_WARN_TCP_HDR)
            {
                printf("%s,from %s:%hi to  %s:%hi\n", nids_warnings[errnum], source_ip, ntohs(((struct tcp_header*)data)->th_sport), destination_ip, ntohs(((struct tcp_header*)data)->th_dport));
            }
            else
            {
                printf("%s,from %s to %s\n", nids_warnings[errnum], source_ip, destination_ip);
            }
            break;
        case NIDS_WARN_SCAN:
            scan_number++;
            sprintf(string_content, "-------------  %d  -------------\n", scan_number);
            printf("%s", string_content);
create_and_write_txt(string_content);
            printf("-----  发现扫描攻击 -----\n");
create_and_write_txt("-----  发现扫描攻击 -----\n");


            host_information = (struct host*)data;
            sprintf(string_content, "扫描者的IP地址为:\n");
            printf("%s", string_content);

create_and_write_txt(string_content);


            sprintf(string_content, "%s\n", inet_ntoa(*((struct in_addr*) &(host_information->addr))));
            printf("%s", string_content);
create_and_write_txt(string_content);
            sprintf(string_content, "被扫描者的IP地址和端口号为:\n");
            printf("%s", string_content);
create_and_write_txt(string_content);

            sprintf(string_content, "");
            for (i = 0; i < host_information->n_packets; i++)
            {
                strcat(string_content, inet_ntoa(*((struct in_addr*) &(host_information->packets[i].addr))));
                sprintf(string_content + strlen(string_content), ":%hi\n", host_information->packets[i].port);
create_and_write_txt(string_content);

                flagsand &= host_information->packets[i].flags;
                flagsor |= host_information->packets[i].flags;
            }
            printf("%s", string_content);
create_and_write_txt(string_content);

            sprintf(string_content, "");
            if (flagsand == flagsor)
            {
                i = flagsand;
                switch (flagsand)
                {
                case 2:
                    strcat(string_content, "扫描类型为: SYN\n");
                    break;
                case 0:
                    strcat(string_content, "扫描类型为: NULL\n");
                    break;
                case 1:
                    strcat(string_content, "扫描类型为: FIN\n");
                    break;
                default:
                    sprintf(string_content + strlen(string_content), "标志=0x%x\n", i);
                }
            }
            else
            {
                strcat(string_content, "标志异常\n");
            }
            printf("%s", string_content);
create_and_write_txt(string_content);
            break;
        default:
            sprintf(content, "未知");
create_and_write_txt(string_content);
            printf("%s", string_content);
            break;
    }
}





/*
-----------------------------------------------------------------------------------------------------------------------
Libpcap头文件
-----------------------------------------------------------------------------------------------------------------------
 */
struct ether_header
/* 以太网协议的数据结构 */
{
    u_int8_t ether_dhost[6];
    /* 目的以太网地址 */
    u_int8_t ether_shost[6];
    /* 源以太网地址 */
    u_int16_t ether_type;
    /* 以太网类型 */
};
/*
=======================================================================================================================
下面的函数是回调函数，其功能是实现捕获以太网数据包，分析其各个字段的内容。注意，其中参数packet_content表示的就是捕获
到的网络数据包内容。参数argument是从函数pcap_loop（）传递过来的。参数pcap_pkthdr表示捕获到的数据包基本信息，包括时间
，长度等信息。
=======================================================================================================================
 */
void ethernet_protocol_packet_callback(u_char *argument, const struct pcap_pkthdr *packet_header, const u_char *packet_content)
{
    u_short ethernet_type;
    /* 以太网类型 */
    struct ether_header *ethernet_protocol;
    /* 以太网协议格式 */
    u_char *mac_string;
    /* 以太网地址 */
    static int packet_number = 1;
    /* 表示捕获数据包的个数 */
    printf("**************************************************\n");
    printf("The %d Ethernet  packet is captured.\n", packet_number);
    printf("-----------    Ehternet Potocol (Link Layer)  ------------\n");
    printf("The %d Ethernet  packet is captured.\n", packet_number);
    ethernet_protocol = (struct ether_header*)packet_content;
    /* 获取以太网协议数据 */
    printf("Ethernet type is :\n");
    ethernet_type = ntohs(ethernet_protocol->ether_type);
    /* 获取以太网类型 */
    printf("%04x\n", ethernet_type);
    switch (ethernet_type) /* 对以太网类型进行判断 */
    {
        case 0x0800:
            printf("The network layer is IP protocol\n");
            break;
        case 0x0806:
            printf("The network layer is ARP protocol\n");
            break;
        case 0x8035:
            printf("The network layer is RARP protocol\n");
            break;
        default:
            break;
    }
    printf("Mac Source Address is : \n");
    mac_string = ethernet_protocol->ether_shost;
    printf("%02x:%02x:%02x:%02x:%02x:%02x\n", *mac_string, *(mac_string + 1), *(mac_string + 2), *(mac_string + 3), *(mac_string + 4), *(mac_string + 5));
    /* 输出源以太网地址 */
    printf("Mac Destination Address is : \n");
    mac_string = ethernet_protocol->ether_dhost;
    printf("%02x:%02x:%02x:%02x:%02x:%02x\n", *mac_string, *(mac_string + 1), *(mac_string + 2), *(mac_string + 3), *(mac_string + 4), *(mac_string + 5));
    /* 输出目的以太网地址 */
    printf("**************************************************\n");
    packet_number++; /* 数据包个数增加 */
}





/*
=======================================================================================================================
主函数
=======================================================================================================================
 */
void main()
{

    FILE* fp;   
    fp = fopen("data.txt","w");
    if(fp==NULL)   
        return 0;
    fclose(fp);   

    struct nids_chksum_ctl temp;
    temp.netaddr = 0;
    temp.mask = 0;
    temp.action = 1;
    nids_register_chksum_ctl(&temp,1); 
    
    
    pcap_t *pcap_handle;
	    /* Libpcap句柄 */
	    char error_content[PCAP_ERRBUF_SIZE];
	    char *net_interface; /* 网路接口 */
	    struct bpf_program bpf_filter;
	    /* 过滤规则 */
	    char bpf_filter_string[] = "ip";
    /* 过滤规则字符串，此时表示本程序只是捕获IP协议的数据包，同样也是以太网数据包。 */
	    bpf_u_int32 net_mask;
	    /* 网络掩码 */
	    bpf_u_int32 net_ip;
	    /* 网络地址 */
	    net_interface = pcap_lookupdev(error_content);
	    /* 获得网络接口 */
	    pcap_lookupnet(net_interface, &net_ip,  &net_mask,  error_content);
	    /* 获得网络地址和网络掩码 */
	    pcap_handle = pcap_open_live(net_interface, BUFSIZ, 1, 0,error_content);
	    /* 打开网络接口 */
	    pcap_compile(pcap_handle,  &bpf_filter, bpf_filter_string,  0,  net_ip); 
	    /* 编译过滤规则 */
	    pcap_setfilter(pcap_handle,  &bpf_filter); /* BPF过滤规则 */
	    /* 设置规律规则 */
	    if (pcap_datalink(pcap_handle) != DLT_EN10MB)
		return ;
    
    if (!nids_init())
     /* Libnids初始化 */
    {
        printf("出现错误：%s\n", nids_errbuf);
        exit(1);
    }
    printf("键入0：全部网络活动状态all\n键入1：主要应用层状态\n键入2：http\n键入3：ftp\n键入4：smtp\n键入5：进入攻击检测模式\n");
    switch (getchar())
    {
     case '0':
	   printf("全部网络活动状态\n");
	    pcap_loop(pcap_handle, - 1, ethernet_protocol_packet_callback,NULL); /* 传递给回调函数的参数 */
	    /*无限循环捕获网络数据包，注册回到函数ethernet_protocol_callback（），捕获每个数据包都要调用此回调函数进行操作*/
	    pcap_close(pcap_handle);
	    /* 关闭Libpcap操作 */    
    	break;
     case '1':
       printf("主要应用层状态\n");
        nids_register_tcp(http_protocol_callback);
    	nids_register_tcp(ftp_protocol_callback);
    	nids_register_tcp(smtp_protocol_callback);
    	break;
     case '2':
       printf("HTTP模式\n");
       nids_register_tcp(http_protocol_callback);
     break;
     case '3':
       printf("FTP模式\n");
       nids_register_tcp(ftp_protocol_callback);
     break;
     case '4':
        printf("SMTP模式\n");
    	nids_register_tcp(smtp_protocol_callback);
    	nids_run();
     break;
     case '5':
     	 printf("攻击检测模式\n");
     	nids_params.syslog = my_nids_syslog;
    	/* 注册检测攻击的函数 */
    	nids_params.pcap_filter = "ip";
     break;
     default: break;
    }
    /*
    nids_register_tcp(http_protocol_callback);
    nids_register_tcp(ftp_protocol_callback);
    nids_register_tcp(smtp_protocol_callback);
    */
    /* 注册回调函数 */
    nids_run(); /* 进入循环捕获数据包状态 */
    	    
}
