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

struct InequalityType {
  ICmpInst::Predicate Pred;
  const SCEV *LHS;
  APInt RHS;

  InequalityType(CmpPredicate Pred, const SCEV *LHS, APInt RHS)
      : Pred(Pred), LHS(LHS), RHS(RHS) {}

  void print(raw_ostream &OS) const {
    OS << *LHS << " " << ICmpInst::getPredicateName(Pred) << " " << RHS;
  }

  raw_ostream &operator<<(raw_ostream &OS) const {
    print(OS);
    return OS;
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const InequalityType &I) {
  I.print(OS);
  return OS;
}

struct ExecutionDomain {
  ExecutionDomain(ScalarEvolution &SE) : SE(SE) {}

  void addInequality(const InequalityType &Inequality);

  bool hasContext(const SCEV *S) const { return Contexts.contains(S); }

  bool isKnownNonNegative(const SCEV *S);

  bool isKnownNonPositive(const SCEV *S);

  ScalarEvolution &getSE() const { return SE; }

  ConstantRange withContext(const SCEV *S, ConstantRange Range);

  void print(raw_ostream &OS);

private:
  ScalarEvolution &SE;
  DenseMap<const SCEV *, DenseMap<ICmpInst::Predicate, InequalityType>>
      Contexts;
};

struct ExecutionDomainPrinterPass
    : public PassInfoMixin<ExecutionDomainPrinterPass> {
  explicit ExecutionDomainPrinterPass(raw_ostream &OS);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  raw_ostream &OS;
};

}  // namespace llvm

#endif
