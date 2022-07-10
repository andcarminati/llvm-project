//===- XtensaSubtarget.cpp - Xtensa Subtarget Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Xtensa specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "XtensaSubtarget.h"
#include "XtensaTargetMachine.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "GISel/XtensaCallLowering.h"
#include "GISel/XtensaLegalizerInfo.h"
#include "GISel/XtensaRegisterBankInfo.h"

#define DEBUG_TYPE "xtensa-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "XtensaGenSubtargetInfo.inc"

using namespace llvm;

XtensaSubtarget &
XtensaSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  StringRef CPUName = CPU;
  if (CPUName.empty()) {
    // set default cpu name
    CPUName = "esp32";
  }

  HasDensity = false;
  HasSingleFloat = false;
  HasWindowed = false;
  HasBoolean = false;
  HasLoop = false;
  HasSEXT = false;
  HasNSA = false;
  HasMul16 = false;
  HasMul32 = false;
  HasMul32High = false;
  HasDiv32 = false;
  HasMAC16 = false;
  HasDFPAccel = false;
  HasS32C1I = false;
  HasTHREADPTR = false;
  HasExtendedL32R = false;
  HasATOMCTL = false;
  HasMEMCTL = false;
  HasDebug = false;
  HasException = false;
  HasHighPriInterrupts = false;
  HasCoprocessor = false;
  HasInterrupt = false;
  HasRelocatableVector = false;
  HasTimerInt = false;
  HasPRID = false;
  HasRegionProtection = false;
  HasMiscSR = false;
  HasESP32S2Ops = false;
  HasESP32S3Ops = false;

  // Parse features string.
  ParseSubtargetFeatures(CPUName, CPUName, FS);
  return *this;
}

XtensaSubtarget::XtensaSubtarget(const Triple &TT, const std::string &CPU,
                                 const std::string &FS, const XtensaTargetMachine &TM)
    : XtensaGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), TargetTriple(TT),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      TSInfo(), FrameLowering() {

  CallLoweringInfo.reset(new XtensaCallLowering(*getTargetLowering()));
  //InlineAsmLoweringInfo.reset(new InlineAsmLowering(getTargetLowering()));
  Legalizer.reset(new XtensaLegalizerInfo(*this));

  auto *RBI = new XtensaRegisterBankInfo(*getRegisterInfo());

  // FIXME: At this point, we can't rely on Subtarget having RBI.
  // It's awkward to mix passing RBI and the Subtarget; should we pass
  // TII/TRI as well?
  InstSelector.reset(createXtensaInstructionSelector(
      *static_cast<const XtensaTargetMachine *>(&TM), *this, *RBI));

  RegBankInfo.reset(RBI);
}


const CallLowering *XtensaSubtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

const InlineAsmLowering *XtensaSubtarget::getInlineAsmLowering() const {
  return InlineAsmLoweringInfo.get();
}

InstructionSelector *XtensaSubtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *XtensaSubtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *XtensaSubtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}