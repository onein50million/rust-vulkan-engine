//
// Created by Daniel on 2021-10-06.
//

#include "system.h"

int print(char* string, int argument_count, long* arguments){
    int syscall_type = 1;
    __asm__("addi sp, sp, - %4\n"
            "sd %0, 0(sp)\n" //syscall type
            "sd %1, 8(sp)\n" //string pointer
            "sd %2, 16(sp)\n" //argument count
            "sd %3, 24(sp)\n" //argument pointer
            "ecall\n"
            "addi sp, sp, %4\n"
    :
    :"r"(syscall_type),"r"(string), "r"(argument_count), "r" (arguments), "i"(32));
    return 1;
}
