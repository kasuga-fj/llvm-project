#ifndef LLVM_ANALYSIS_EXECUTIONDOMAIN_H
#define LLVM_ANALYSIS_EXECUTIONDOMAIN_H

#include "llvm/Analysis/Delinearization.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionDivision.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace llvm {

struct OverflowSafeSignedAPInt {
  OverflowSafeSignedAPInt() : Value(std::nullopt) {}
  OverflowSafeSignedAPInt(const APInt &V) : Value(V) {}
  OverflowSafeSignedAPInt(const std::optional<APInt> &V) : Value(V) {}

  OverflowSafeSignedAPInt operator+(const OverflowSafeSignedAPInt &RHS) const {
    if (!Value || !RHS.Value)
      return OverflowSafeSignedAPInt();
    bool Overflow;
    APInt Result = Value->sadd_ov(*RHS.Value, Overflow);
    if (Overflow)
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(Result);
  }

  OverflowSafeSignedAPInt operator+(int RHS) const {
    if (!Value)
      return OverflowSafeSignedAPInt();
    return *this + fromInt(RHS);
  }

  OverflowSafeSignedAPInt operator-(const OverflowSafeSignedAPInt &RHS) const {
    if (!Value || !RHS.Value)
      return OverflowSafeSignedAPInt();
    bool Overflow;
    APInt Result = Value->ssub_ov(*RHS.Value, Overflow);
    if (Overflow)
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(Result);
  }

  OverflowSafeSignedAPInt operator-(int RHS) const {
    if (!Value)
      return OverflowSafeSignedAPInt();
    return *this - fromInt(RHS);
  }

  OverflowSafeSignedAPInt operator*(const OverflowSafeSignedAPInt &RHS) const {
    if (!Value || !RHS.Value)
      return OverflowSafeSignedAPInt();
    bool Overflow;
    APInt Result = Value->smul_ov(*RHS.Value, Overflow);
    if (Overflow)
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(Result);
  }

  OverflowSafeSignedAPInt operator-() const {
    if (!Value)
      return OverflowSafeSignedAPInt();
    if (Value->isMinSignedValue())
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(-*Value);
  }

  OverflowSafeSignedAPInt operator/(const OverflowSafeSignedAPInt &RHS) const {
    if (!Value || !RHS.Value)
      return OverflowSafeSignedAPInt();
    if (RHS.Value->isZero())
      return OverflowSafeSignedAPInt();
    bool Overflow = false;
    APInt Res = Value->sdiv_ov(*RHS.Value, Overflow);
    if (Overflow)
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(Res);
  }

  operator bool() const { return Value.has_value(); }

  bool operator!() const { return !Value.has_value(); }

  const APInt &operator*() const {
    assert(Value && "Value is not available.");
    return *Value;
  }

  const APInt *operator->() const {
    assert(Value && "Value is not available.");
    return &*Value;
  }

  void print(raw_ostream &OS) const {
    if (Value)
      OS << *Value;
    else
      OS << "unknown";
  }

private:
  /// Underlying value. std::nullopt means "unknown". An arithmetic operation on
  /// "unknown" always produces "unknown".
  std::optional<APInt> Value;

  OverflowSafeSignedAPInt fromInt(uint64_t V) const {
    assert(Value && "Value is not available.");
    return OverflowSafeSignedAPInt(
        APInt(Value->getBitWidth(), V, /*isSigned=*/true));
  }
};

struct InequalityType {
  CmpPredicate Pred;
  const SCEV *LHS;
  OverflowSafeSignedAPInt RHS;

  void print(raw_ostream &OS) const {
    OS << *LHS << " " << ICmpInst::getPredicateName(Pred) << " " << RHS;
  }
};

struct ExecutionContext {
  const SCEV *S;
  SmallVector<InequalityType, 4> Contexts;

private:
  uint64_t UpdatedAt = 0;
  std::optional<ConstantRange> CachedRange;
};

struct ExecutionDomain {
  ExecutionDomain(ScalarEvolution &SE) : SE(SE) {}

  void smin(const SCEV *S, const APInt &RHS);
  void smax(const SCEV *S, const APInt &RHS);
  void clamp(const APInt &Min, const SCEV *S, const APInt &Max);

  bool addWillNotOverflow(const SCEV *LHS, const SCEV *RHS);
  bool mulWillNotOverflow(const SCEV *LHS, const SCEV *RHS);

  bool isKnownNonNegative(const SCEV *S);
  bool isKnownNonPositive(const SCEV *S);

  ConstantRange getRange(const SCEV *S) { return fetch(S)->second; }

  void setRange(const SCEV *S, const ConstantRange &Range);

  ScalarEvolution &getSE() const { return SE; }

private:
  using Cache = DenseMap<const SCEV *, ConstantRange>;
  ScalarEvolution &SE;
  DenseMap<const SCEV *, ConstantRange> Ranges;

  Cache::iterator fetch(const SCEV *S) {
    return Ranges.try_emplace(S, SE.getSignedRange(S)).first;
  }
};

struct ExecutionDomainPrinterPass
    : public PassInfoMixin<ExecutionDomainPrinterPass> {
  explicit ExecutionDomainPrinterPass(raw_ostream &OS);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  raw_ostream &OS;
};

struct ExecutionDomainInterpreter
    : public SCEVVisitor<ExecutionDomainInterpreter, ConstantRange> {

  ExecutionDomainInterpreter(ExecutionDomain &ED) : ED(ED) {}

  static ConstantRange evaluate(const SCEV *S, ExecutionDomain &ED) {
    return ExecutionDomainInterpreter(ED).visit(S);
  }

  ConstantRange visitConstant(const SCEVConstant *C) {
    return ConstantRange(C->getAPInt());
  }

  ConstantRange visitAddExpr(const SCEVAddExpr *S) {
    ConstantRange Result(APInt::getZero(S->getType()->getIntegerBitWidth()));
    for (const SCEV *Op : S->operands())
      Result = Result.add(visit(Op));
    ED.setRange(S, Result);
    return Result;
  }

  ConstantRange visitMulExpr(const SCEVMulExpr *S) {
    ConstantRange Result(
        APInt(S->getType()->getIntegerBitWidth(), 1, true, false));
    for (const SCEV *Op : S->operands())
      Result = Result.smul_fast(visit(Op));
    ED.setRange(S, Result);
    return Result;
  }

  ConstantRange visitSignExtendExpr(const SCEVSignExtendExpr *S) {
    ConstantRange Result =
        visit(S->getOperand()).signExtend(S->getType()->getIntegerBitWidth());
    ED.setRange(S, Result);
    return Result;
  }

  ConstantRange visitSMinExpr(const SCEVSMinExpr *S) {
    ConstantRange Result = visit(S->getOperand(0));
    for (unsigned I = 1, E = S->getNumOperands(); I != E; ++I)
      Result = Result.smin(visit(S->getOperand(I)));
    ED.setRange(S, Result);
    return Result;
  }

  ConstantRange visitSMaxExpr(const SCEVSMaxExpr *S) {
    ConstantRange Result = visit(S->getOperand(0));
    for (unsigned I = 1, E = S->getNumOperands(); I != E; ++I)
      Result = Result.smax(visit(S->getOperand(I)));
    ED.setRange(S, Result);
    return Result;
  }

  ConstantRange visitVScale(const SCEVVScale *S) { return unknownRange(S); }
  ConstantRange visitPtrToAddrExpr(const SCEVPtrToAddrExpr *S) {
    return unknownRange(S);
  }
  ConstantRange visitPtrToIntExpr(const SCEVPtrToIntExpr *S) {
    return unknownRange(S);
  }
  ConstantRange visitTruncateExpr(const SCEVTruncateExpr *S) {
    return unknownRange(S);
  }
  ConstantRange visitZeroExtendExpr(const SCEVZeroExtendExpr *S) {
    return unknownRange(S);
  }
  ConstantRange visitUDivExpr(const SCEVUDivExpr *S) { return unknownRange(S); }
  ConstantRange visitAddRecExpr(const SCEVAddRecExpr *S) {
    return unknownRange(S);
  }
  ConstantRange visitUMaxExpr(const SCEVUMaxExpr *S) { return unknownRange(S); }
  ConstantRange visitUMinExpr(const SCEVUMinExpr *S) { return unknownRange(S); }
  ConstantRange visitUnknown(const SCEVUnknown *S) { return unknownRange(S); }
  ConstantRange visitSequentialUMinExpr(const SCEVSequentialUMinExpr *S) {
    return unknownRange(S);
  }

private:
  ExecutionDomain &ED;

  ConstantRange unknownRange(const SCEV *S) {
    IntegerType *Ty = cast<IntegerType>(S->getType());
    return ConstantRange::getFull(Ty->getBitWidth());
  }
};

}  // namespace llvm

#endif
