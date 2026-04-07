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

struct ExecutionDomain {
  ExecutionDomain(ScalarEvolution &SE) : SE(SE) {}

  void smin(const SCEV *S, const APInt &RHS);
  void smax(const SCEV *S, const APInt &RHS);
  void clamp(const APInt &Min, const SCEV *S, const APInt &Max);

  bool isKnownNonNegative(const SCEV *S);
  bool isKnownNonPositive(const SCEV *S);

private:
  using Cache = DenseMap<const SCEV *, ConstantRange>;
  ScalarEvolution &SE;
  DenseMap<const SCEV *, ConstantRange> Ranges;

  Cache::iterator fetch(const SCEV *S) {
    return Ranges.try_emplace(S, SE.getSignedRange(S)).first;
  }
};

struct DelinearizationPrinterPass
    : public PassInfoMixin<DelinearizationPrinterPass> {
  explicit DelinearizationPrinterPass(raw_ostream &OS);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  raw_ostream &OS;
};

}  // namespace llvm

#endif
