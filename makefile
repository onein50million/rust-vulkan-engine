starship-os.bin: src/starship-os/main.c
	riscv64-unknown-elf-gcc -S src/starship-os/main.c -o starship-os.s
	riscv64-unknown-elf-gcc -Wl,-Ttext=0x0 -ffunction-sections -Xlinker -T linker.ld -nostdlib -march=rv64i -mabi=lp64 -o starship-os.elf starship-os.s
	riscv64-unknown-elf-objcopy -O binary starship-os.elf starship-os.bin
