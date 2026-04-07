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

  void print(raw_ostream &OS) const {
    OS << *LHS << " " << ICmpInst::getPredicateName(Pred) << " " << RHS;
  }
};

struct ExecutionContext {
  const SCEV *S;
  SmallVector<InequalityType, 4> Contexts;

private:
  uint64_t CachedAt = 0;
  uint64_t UpdatedAt = 0;
  std::optional<ConstantRange> CachedRange;

  void setCache(const ConstantRange &Range, uint64_t Version) {
    CachedRange = Range;
    UpdatedAt = Version;
  }

  void addContext(const InequalityType &Inequality, uint64_t Version) {
    Contexts.push_back(Inequality);
    UpdatedAt = Version;
  }

  friend struct ExecutionDomain;
};

struct ExecutionDomain {
  ExecutionDomain(ScalarEvolution &SE) : SE(SE), Version(1) {}

  void addInequality(const InequalityType &Inequality);

  uint64_t getVersion() const { return Version; }

  ConstantRange getRange(const SCEV *S);

private:
  ScalarEvolution &SE;
  uint64_t Version = 0;

  DenseMap<const SCEV *, std::unique_ptr<ExecutionContext>> Contexts;
  DenseMap<const SCEV *, SetVector<const SCEV *>> Preds;
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
