//===--- AArch64CallLowering.cpp - Call lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "XtensaCallLowering.h"
#include "XtensaISelLowering.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/MachineValueType.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>

#define DEBUG_TYPE "xtensa-call-lowering"

using namespace llvm;

XtensaCallLowering::XtensaCallLowering(const XtensaTargetLowering &TLI)
  : CallLowering(&TLI) {}

static void applyStackPassedSmallTypeDAGHack(EVT OrigVT, MVT &ValVT,
                                             MVT &LocVT) {
  // If ValVT is i1/i8/i16, we should set LocVT to i8/i8/i16. This is a legacy
  // hack because the DAG calls the assignment function with pre-legalized
  // register typed values, not the raw type.
  //
  // This hack is not applied to return values which are not passed on the
  // stack.
  if (OrigVT == MVT::i1 || OrigVT == MVT::i8)
    ValVT = LocVT = MVT::i8;
  else if (OrigVT == MVT::i16)
    ValVT = LocVT = MVT::i16;
}

// Account for i1/i8/i16 stack passed value hack
static LLT getStackValueStoreTypeHack(const CCValAssign &VA) {
  const MVT ValVT = VA.getValVT();
  return (ValVT == MVT::i8 || ValVT == MVT::i16) ? LLT(ValVT)
                                                 : LLT(VA.getLocVT());
}

namespace{
  struct XtensaOutgoingValueAssigner
    : public CallLowering::OutgoingValueAssigner {
  const XtensaSubtarget &Subtarget;

  /// Track if this is used for a return instead of function argument
  /// passing. We apply a hack to i1/i8/i16 stack passed values, but do not use
  /// stack passed returns for them and cannot apply the type adjustment.
  bool IsReturn;

  XtensaOutgoingValueAssigner(CCAssignFn *AssignFn_,
                               CCAssignFn *AssignFnVarArg_,
                               const XtensaSubtarget &Subtarget_,
                               bool IsReturn)
      : OutgoingValueAssigner(AssignFn_, AssignFnVarArg_),
        Subtarget(Subtarget_), IsReturn(IsReturn) {}

  bool assignArg(unsigned ValNo, EVT OrigVT, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, ISD::ArgFlagsTy Flags,
                 CCState &State) override {
    bool UseVarArgsCCForFixed = State.isVarArg();

    if (!State.isVarArg() && !UseVarArgsCCForFixed && !IsReturn)
      applyStackPassedSmallTypeDAGHack(OrigVT, ValVT, LocVT);

    bool Res;
    if (Info.IsFixed && !UseVarArgsCCForFixed)
      Res = AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State);
    else
      Res = AssignFnVarArg(ValNo, ValVT, LocVT, LocInfo, Flags, State);

    StackOffset = State.getNextStackOffset();
    return Res;
  }
};
struct OutgoingArgHandler : public CallLowering::OutgoingValueHandler {
  OutgoingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                     MachineInstrBuilder MIB, bool IsTailCall = false,
                     int FPDiff = 0)
      : OutgoingValueHandler(MIRBuilder, MRI), MIB(MIB), IsTailCall(IsTailCall),
        FPDiff(FPDiff),
        Subtarget(MIRBuilder.getMF().getSubtarget<XtensaSubtarget>()) {}

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO,
                           ISD::ArgFlagsTy Flags) override {
    MachineFunction &MF = MIRBuilder.getMF();
    LLT p0 = LLT::pointer(0, 32);
    LLT s32 = LLT::scalar(32);

    if (IsTailCall) {
      assert(!Flags.isByVal() && "byval unhandled with tail calls");

      Offset += FPDiff;
      int FI = MF.getFrameInfo().CreateFixedObject(Size, Offset, true);
      auto FIReg = MIRBuilder.buildFrameIndex(p0, FI);
      MPO = MachinePointerInfo::getFixedStack(MF, FI);
      return FIReg.getReg(0);
    }

    if (!SPReg)
      SPReg = MIRBuilder.buildCopy(p0, Register(Xtensa::SP)).getReg(0);

    auto OffsetReg = MIRBuilder.buildConstant(s32, Offset);

    auto AddrReg = MIRBuilder.buildPtrAdd(p0, SPReg, OffsetReg);

    MPO = MachinePointerInfo::getStack(MF, Offset);
    return AddrReg.getReg(0);
  }

  /// We need to fixup the reported store size for certain value types because
  /// we invert the interpretation of ValVT and LocVT in certain cases. This is
  /// for compatability with the DAG call lowering implementation, which we're
  /// currently building on top of.
  LLT getStackValueStoreType(const DataLayout &DL, const CCValAssign &VA,
                             ISD::ArgFlagsTy Flags) const override {
    if (Flags.isPointer())
      return CallLowering::ValueHandler::getStackValueStoreType(DL, VA, Flags);
    return getStackValueStoreTypeHack(VA);
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign VA) override {
    MIB.addUse(PhysReg, RegState::Implicit);
    Register ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, LLT MemTy,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    MachineFunction &MF = MIRBuilder.getMF();
    auto MMO = MF.getMachineMemOperand(MPO, MachineMemOperand::MOStore, MemTy,
                                       inferAlignFromPtrInfo(MF, MPO));
    MIRBuilder.buildStore(ValVReg, Addr, *MMO);
  }

  void assignValueToAddress(const CallLowering::ArgInfo &Arg, unsigned RegIndex,
                            Register Addr, LLT MemTy, MachinePointerInfo &MPO,
                            CCValAssign &VA) override {
    unsigned MaxSize = MemTy.getSizeInBytes() * 8;
    // For varargs, we always want to extend them to 8 bytes, in which case
    // we disable setting a max.
    if (!Arg.IsFixed)
      MaxSize = 0;

    Register ValVReg = Arg.Regs[RegIndex];
    if (VA.getLocInfo() != CCValAssign::LocInfo::FPExt) {
      MVT LocVT = VA.getLocVT();
      MVT ValVT = VA.getValVT();

      if (VA.getValVT() == MVT::i8 || VA.getValVT() == MVT::i16) {
        std::swap(ValVT, LocVT);
        MemTy = LLT(VA.getValVT());
      }

      ValVReg = extendRegister(ValVReg, VA, MaxSize);
    } else {
      // The store does not cover the full allocated stack slot.
      MemTy = LLT(VA.getValVT());
    }

    assignValueToAddress(ValVReg, Addr, MemTy, MPO, VA);
  }

  MachineInstrBuilder MIB;

  bool IsTailCall;

  /// For tail calls, the byte offset of the call's argument area from the
  /// callee's. Unused elsewhere.
  int FPDiff;

  // Cache the SP register vreg if we need it more than once in this call site.
  Register SPReg;

  const XtensaSubtarget &Subtarget;
};
}

bool XtensaCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                      const Value *Val,
                                      ArrayRef<Register> VRegs,
                                      FunctionLoweringInfo &FLI,
                                      Register SwiftErrorVReg) const {
  
  auto MIB = MIRBuilder.buildInstrNoInsert(Xtensa::RET);
  assert(((Val && !VRegs.empty()) || (!Val && VRegs.empty())) &&
         "Return value without a vreg");

  bool Success = true;
  if (!VRegs.empty()) {
    MachineFunction &MF = MIRBuilder.getMF();
    const Function &F = MF.getFunction();
    const XtensaSubtarget &Subtarget = MF.getSubtarget<XtensaSubtarget>();

    MachineRegisterInfo &MRI = MF.getRegInfo();
    const XtensaTargetLowering &TLI = *getTLI<XtensaTargetLowering>();
    CCAssignFn *AssignFn = TLI.CCAssignFnForReturn(F.getCallingConv());
    auto &DL = F.getParent()->getDataLayout();
    LLVMContext &Ctx = Val->getType()->getContext();

    SmallVector<EVT, 4> SplitEVTs;
    ComputeValueVTs(TLI, DL, Val->getType(), SplitEVTs);
    assert(VRegs.size() == SplitEVTs.size() &&
           "For each split Type there should be exactly one VReg.");

    SmallVector<ArgInfo, 8> SplitArgs;
    CallingConv::ID CC = F.getCallingConv();

    for (unsigned i = 0; i < SplitEVTs.size(); ++i) {
      Register CurVReg = VRegs[i];
      ArgInfo CurArgInfo = ArgInfo{CurVReg, SplitEVTs[i].getTypeForEVT(Ctx), 0};
      setArgFlags(CurArgInfo, AttributeList::ReturnIndex, DL, F);
       
      // todo... 
      if (CurVReg != CurArgInfo.Regs[0]) {
        CurArgInfo.Regs[0] = CurVReg;
        // Reset the arg flags after modifying CurVReg.
        setArgFlags(CurArgInfo, AttributeList::ReturnIndex, DL, F);
      }
      splitToValueTypes(CurArgInfo, SplitArgs, DL, CC);
    }
    XtensaOutgoingValueAssigner Assigner(AssignFn, AssignFn, Subtarget,
                                          /*IsReturn*/ true);
    OutgoingArgHandler Handler(MIRBuilder, MRI, MIB);
    Success = determineAndHandleAssignments(Handler, Assigner, SplitArgs,
                                            MIRBuilder, CC, F.isVarArg());
  }


  MIRBuilder.insertInstr(MIB);
  //MIB->dump();
  return Success;
}



bool XtensaCallLowering::fallBackToDAGISel(const MachineFunction &MF) const {
  auto &F = MF.getFunction();

  return false;
}

bool XtensaCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs, FunctionLoweringInfo &FLI) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  for (auto &Arg : F.args()) {
    Arg.dump();
  }  
  //llvm_unreachable("XtensaCallLowering::lowerFormalArguments");

  return true;
}


bool XtensaCallLowering::isEligibleForTailCallOptimization(
    MachineIRBuilder &MIRBuilder, CallLoweringInfo &Info,
    SmallVectorImpl<ArgInfo> &InArgs,
    SmallVectorImpl<ArgInfo> &OutArgs) const {

  // Must pass all target-independent checks in order to tail call optimize.
  if (!Info.IsTailCall)
    return false;


  LLVM_DEBUG(
      dbgs() << "... Call is eligible for tail call optimization.\n");
  return true;
}




bool XtensaCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                    CallLoweringInfo &Info) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();
  const XtensaTargetLowering &TLI = *getTLI<XtensaTargetLowering>();

  llvm_unreachable("XtensaCallLowering::lowerCall");
  return true;
}

bool XtensaCallLowering::isTypeIsValidForThisReturn(EVT Ty) const {
  return Ty.getSizeInBits() == 64;
}
