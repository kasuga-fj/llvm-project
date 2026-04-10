#ifndef LLVM_ANALYSIS_EXECUTIONDOMAIN_H
#define LLVM_ANALYSIS_EXECUTIONDOMAIN_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionDivision.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
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

struct LoopNestInspector {
  LoopNestInspector(const DominatorTree &DT, const LoopInfo &LI,
                    ScalarEvolution &SE)
      : DT(DT), LI(LI), SE(SE) {}

  /// Assume that the \p Ptr is surrounded by N nested loops, Returns true if we
  /// can prove that the \p Ptr is used by certain memory access for every
  /// combination of (i_1, ..., i_N) where:
  ///
  ///   - i_k denotes the iteration number of the k-th loop surrounding the \p
  ///   Ptr.
  ///   - 0 <= i_k < BTC_k where BTC_k is the exact backedge taken count of the
  ///   k-th loop.
  ///
  /// If either of the loops does not have an exact BTC, then returns false.
  bool isSafeToEstimateMaxOffsetValue(Value *Ptr, const Loop *Outermost);

private:
  const DominatorTree &DT;
  const LoopInfo &LI;
  ScalarEvolution &SE;
  DenseMap<const Loop *, bool> Checked;

  /// Returns true if \p Ptr is used by certain memory access at every iteration
  /// of \p L.
  bool isPtrUsedAtEveryIteration(Value *Ptr, const Loop *L);

  /// Returns true if the outer loop of \p Inner has an exact BTC and \p Inner
  /// is executed at every iteration of the outer loop. \p Inner must have a
  /// parent loop.
  bool isInnerLoopExecutedAtEveryIteration(const Loop *Inner);
  bool isSafeToEstimateMaxOffsetValueRec(const SCEV *S, const Loop *Inner,
                                         const Loop *Outermost);
};

struct ExecutionDomain {
  ExecutionDomain(ScalarEvolution &SE) : SE(SE) {}

  void run(Function &F, const LoopInfo &LI, const DominatorTree &DT);

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
