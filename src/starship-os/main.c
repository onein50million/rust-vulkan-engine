//__asm__("li a1, 0x40404040404040\n"
//        "addi a2, sp, -0xFF\n"
//        "sd a1, (a2)\n"
//        "sd a2, (sp)\n"
//        "ecall\n"
//        "ebreak");
#include "system.c"


long fib(long n);

__attribute__((section(".text.main"))) int main()  {
    long color = 0;
    for(int i = 0; i<128*128; i++){
//        long fib_return = fib(5);
//        print("fib: ",1, (long[]){fib_return});

        void * pMemory = (void *) (0xFFFF + 64 * 8 + i * sizeof(long));
        long * pDisplayMemory = (long *) pMemory;
        *pDisplayMemory = color;
        color = (color + 1); //Undefined behaviour I think
        long out = (long)i;
        print("hello!", 1, &out);
    }

//    for(int i = 0; i<1024*8; i++){
//        long* pMemory = (long *) 0xFFFF + 64*8 + i;
//        *pMemory = 0xDEADBEEF;
//    }

    __asm__("ebreak");
    return 1;
}

// fib.c contains the following C code and fib.bin is the build result of it:
long fib(long n) {
//    print("n: ",1, (long[]){n});
    if (n == 0 || n == 1)
        return n;
    else
        return (fib(n-1) + fib(n-2));
}




