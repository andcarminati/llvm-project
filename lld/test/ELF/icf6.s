# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o /dev/null --icf=all --print-icf-sections | count 0

.globl _start, f1, f2
_start:
  ret

.section .init, "ax"
f1:
  mov $60, %rax
  mov $42, %rdi
  syscall

.section .fini, "ax"
f2:
  mov $60, %rax
  mov $42, %rdi
  syscall
