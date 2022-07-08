//=----- XtensaInstrGISel.td - Xtensa GISel target pseudos -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Xtensa GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===/

class XtensaGenericInstruction : GenericInstruction {
  let Namespace = "AArch64";
}


// Pseudo for a Xtensa_pcrel_wrapper instruction. Produced post-legalization from
def G_PCREL_WRAPPER : XtensaGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type1:$src);
  let hasSideEffects = 0;
}


def : GINodeEquiv<G_PCREL_WRAPPER, Xtensa_pcrel_wrapper>;