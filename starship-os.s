	.file	"main.c"
	.option nopic
	.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	1
	.globl	print
	.type	print, @function
print:
	addi	sp,sp,-64
	sd	s0,56(sp)
	addi	s0,sp,64
	sd	a0,-40(s0)
	mv	a5,a1
	sd	a2,-56(s0)
	sw	a5,-44(s0)
	li	a5,1
	sw	a5,-20(s0)
	lw	a5,-20(s0)
	ld	a4,-40(s0)
	lw	a3,-44(s0)
	ld	a2,-56(s0)
 #APP
# 9 "src/starship-os/system.c" 1
	addi sp, sp, - 32
sd a5, 0(sp)
sd a4, 8(sp)
sd a3, 16(sp)
sd a2, 24(sp)
ecall
addi sp, sp, 32

# 0 "" 2
 #NO_APP
	li	a5,1
	mv	a0,a5
	ld	s0,56(sp)
	addi	sp,sp,64
	jr	ra
	.size	print, .-print
	.section	.rodata
	.align	3
.LC0:
	.string	"hello!"
	.section	.text.main,"ax",@progbits
	.align	1
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-64
	sd	ra,56(sp)
	sd	s0,48(sp)
	addi	s0,sp,64
	sd	zero,-24(s0)
	sw	zero,-28(s0)
	j	.L4
.L5:
	lw	a5,-28(s0)
	slli	a4,a5,3
	li	a5,65536
	addi	a5,a5,511
	add	a5,a4,a5
	sd	a5,-40(s0)
	ld	a5,-40(s0)
	sd	a5,-48(s0)
	ld	a5,-48(s0)
	ld	a4,-24(s0)
	sd	a4,0(a5)
	ld	a5,-24(s0)
	addi	a5,a5,1
	sd	a5,-24(s0)
	lw	a5,-28(s0)
	sd	a5,-56(s0)
	addi	a5,s0,-56
	mv	a2,a5
	li	a1,1
	lui	a5,%hi(.LC0)
	addi	a0,a5,%lo(.LC0)
	call	print
	lw	a5,-28(s0)
	addiw	a5,a5,1
	sw	a5,-28(s0)
.L4:
	lw	a5,-28(s0)
	sext.w	a4,a5
	li	a5,16384
	blt	a4,a5,.L5
 #APP
# 31 "src/starship-os/main.c" 1
	ebreak
# 0 "" 2
 #NO_APP
	li	a5,1
	mv	a0,a5
	ld	ra,56(sp)
	ld	s0,48(sp)
	addi	sp,sp,64
	jr	ra
	.size	main, .-main
	.text
	.align	1
	.globl	fib
	.type	fib, @function
fib:
	addi	sp,sp,-48
	sd	ra,40(sp)
	sd	s0,32(sp)
	sd	s1,24(sp)
	addi	s0,sp,48
	sd	a0,-40(s0)
	ld	a5,-40(s0)
	beq	a5,zero,.L8
	ld	a4,-40(s0)
	li	a5,1
	bne	a4,a5,.L9
.L8:
	ld	a5,-40(s0)
	j	.L10
.L9:
	ld	a5,-40(s0)
	addi	a5,a5,-1
	mv	a0,a5
	call	fib
	mv	s1,a0
	ld	a5,-40(s0)
	addi	a5,a5,-2
	mv	a0,a5
	call	fib
	mv	a5,a0
	add	a5,s1,a5
.L10:
	mv	a0,a5
	ld	ra,40(sp)
	ld	s0,32(sp)
	ld	s1,24(sp)
	addi	sp,sp,48
	jr	ra
	.size	fib, .-fib
	.ident	"GCC: (SiFive GCC-Metal 10.2.0-2020.12.8) 10.2.0"
