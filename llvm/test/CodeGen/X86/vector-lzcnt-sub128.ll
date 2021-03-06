; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1 immarg)

define <2 x i32> @illegal_ctlz(<2 x i32> %v1) {
; CHECK-LABEL: illegal_ctlz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrld $1, %xmm1
; CHECK-NEXT:    por %xmm1, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrld $2, %xmm1
; CHECK-NEXT:    por %xmm1, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrld $4, %xmm1
; CHECK-NEXT:    por %xmm1, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrld $8, %xmm1
; CHECK-NEXT:    por %xmm1, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrld $16, %xmm1
; CHECK-NEXT:    por %xmm1, %xmm0
; CHECK-NEXT:    pcmpeqd %xmm1, %xmm1
; CHECK-NEXT:    pxor %xmm1, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrlw $1, %xmm1
; CHECK-NEXT:    pand {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    psubb %xmm1, %xmm0
; CHECK-NEXT:    movdqa {{.*#+}} xmm1 = [51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51]
; CHECK-NEXT:    movdqa %xmm0, %xmm2
; CHECK-NEXT:    pand %xmm1, %xmm2
; CHECK-NEXT:    psrlw $2, %xmm0
; CHECK-NEXT:    pand %xmm1, %xmm0
; CHECK-NEXT:    paddb %xmm2, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrlw $4, %xmm1
; CHECK-NEXT:    paddb %xmm1, %xmm0
; CHECK-NEXT:    pand {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    movdqa %xmm0, %xmm2
; CHECK-NEXT:    punpckhdq {{.*#+}} xmm2 = xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; CHECK-NEXT:    psadbw %xmm1, %xmm2
; CHECK-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; CHECK-NEXT:    psadbw %xmm1, %xmm0
; CHECK-NEXT:    packuswb %xmm2, %xmm0
; CHECK-NEXT:    retq
  %v2 = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %v1, i1 true)
  ret <2 x i32> %v2
}

declare <2 x i32> @llvm.cttz.v2i32(<2 x i32>, i1 immarg)

define <2 x i32> @illegal_cttz(<2 x i32> %v1) {
; CHECK-LABEL: illegal_cttz:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pcmpeqd %xmm1, %xmm1
; CHECK-NEXT:    paddd %xmm0, %xmm1
; CHECK-NEXT:    pandn %xmm1, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrlw $1, %xmm1
; CHECK-NEXT:    pand {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    psubb %xmm1, %xmm0
; CHECK-NEXT:    movdqa {{.*#+}} xmm1 = [51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51]
; CHECK-NEXT:    movdqa %xmm0, %xmm2
; CHECK-NEXT:    pand %xmm1, %xmm2
; CHECK-NEXT:    psrlw $2, %xmm0
; CHECK-NEXT:    pand %xmm1, %xmm0
; CHECK-NEXT:    paddb %xmm2, %xmm0
; CHECK-NEXT:    movdqa %xmm0, %xmm1
; CHECK-NEXT:    psrlw $4, %xmm1
; CHECK-NEXT:    paddb %xmm1, %xmm0
; CHECK-NEXT:    pand {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm0
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    movdqa %xmm0, %xmm2
; CHECK-NEXT:    punpckhdq {{.*#+}} xmm2 = xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; CHECK-NEXT:    psadbw %xmm1, %xmm2
; CHECK-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; CHECK-NEXT:    psadbw %xmm1, %xmm0
; CHECK-NEXT:    packuswb %xmm2, %xmm0
; CHECK-NEXT:    retq
  %v2 = call <2 x i32> @llvm.cttz.v2i32(<2 x i32> %v1, i1 true)
  ret <2 x i32> %v2
}
