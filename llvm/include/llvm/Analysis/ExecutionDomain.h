#ifndef LLVM_ANALYSIS_EXECUTIONDOMAIN_H
#define LLVM_ANALYSIS_EXECUTIONDOMAIN_H

#include "llvm/ADT/SetVector.h"
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
  CmpPredicate Pred;
  const SCEV *LHS;
  APInt RHS;

  InequalityType(CmpPredicate Pred, const SCEV *LHS, APInt RHS)
      : Pred(Pred), LHS(LHS), RHS(RHS) {}

  void print(raw_ostream &OS) const {
    OS << *LHS << " " << ICmpInst::getPredicateName(Pred) << " " << RHS;
  }
};

struct ExecutionContext {
  const SCEV *S;
  SmallVector<InequalityType, 4> Inequalities;

  ExecutionContext(const SCEV *S) : S(S), Inequalities() {}
};

struct ExecutionDomain {
  ExecutionDomain(ScalarEvolution &SE) : SE(SE) {}

  void addInequality(const InequalityType &Inequality);

  bool hasContext(const SCEV *S) const { return Contexts.contains(S); }

  bool isKnownNonNegative(const SCEV *S);

  bool isKnownNonPositive(const SCEV *S);

  std::optional<ArrayRef<InequalityType>> getInequalities(const SCEV *S) {
    auto Ite = Contexts.find(S);
    if (Ite == Contexts.end())
      return std::nullopt;
    return Ite->second->Inequalities;
  }

  ScalarEvolution &getSE() const { return SE; }

  ConstantRange withContext(const SCEV *S, ConstantRange Range);

  void print(raw_ostream &OS);

private:
  ScalarEvolution &SE;
  DenseMap<const SCEV *, std::unique_ptr<ExecutionContext>> Contexts;

  void addDependencies(const SCEV *Entry);
  void tryAddDependency(const SCEV *From, const SCEV *FromSub, const SCEV *To);
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
