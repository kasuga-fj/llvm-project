#include "llvm/Analysis/ExecutionDomain.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"

#define DEBUG_TYPE "execution-domain"

using namespace llvm;

static bool isExecutedAtEveryIteration(const SCEV *S) {
  // TODO: Impl.
  return true;
}

static SmallPtrSet<const SCEV *, 4> computeComplexity(const SCEV *S) {
  struct FindUnknown {
    SmallPtrSet<const SCEV *, 4> Result;

    bool follow(const SCEV *S) {
      if (isa<SCEVUnknown>(S)) {
        Result.insert(S);
        return false;
      }

      return 1 < S->operands().size();
    }

    bool isDone() const { return false; }
  };

  FindUnknown Finder;
  SCEVTraversal<FindUnknown>(Finder).visitAll(S);
  return std::move(Finder.Result);
}

namespace {

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

raw_ostream &operator<<(raw_ostream &OS, const OverflowSafeSignedAPInt &V) {
  V.print(OS);
  return OS;
}

struct InequalityType {
  CmpPredicate Pred;
  const SCEV *LHS;
  OverflowSafeSignedAPInt RHS;

  void print(raw_ostream &OS) const {
    OS << *LHS << " " << ICmpInst::getPredicateName(Pred) << " " << RHS;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const InequalityType &I) {
  I.print(OS);
  return OS;
}

using Inequalities = SmallVector<InequalityType, 4>;

Inequalities collectInequalities(Function &F, ScalarEvolution &SE) {
  SmallVector<std::pair<InequalityType, unsigned>, 4> Worklist;
  for (Instruction &Inst : instructions(F)) {
    if (!isa<LoadInst, StoreInst>(&Inst))
      continue;

    const SCEV *AccessFn = SE.getSCEV(getPointerOperand(&Inst));
    const SCEVUnknown *BasePointer =
        dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFn));
    if (!BasePointer)
      continue;
    AccessFn = SE.getMinusSCEV(AccessFn, BasePointer);

    Value *Ptr = BasePointer->getValue();
    const DataLayout &DL = F.getDataLayout();
    bool CheckForNonNull, CheckForFreed;
    // TODO: We want the upper bound, not the lower bound.
    uint64_t DerefBytes =
        Ptr->getPointerDereferenceableBytes(DL, CheckForNonNull, CheckForFreed);
    if (DerefBytes && !CheckForNonNull && !CheckForFreed) {
      APInt RHS = APInt(DL.getIndexSize(0) * 8, DerefBytes, false, false);
      unsigned Complexity = computeComplexity(AccessFn).size();
      InequalityType Inequality{ICmpInst::ICMP_SLT, AccessFn, RHS};
      Worklist.emplace_back(Inequality, Complexity);
    }
  }

  llvm::stable_sort(Worklist,
                    [](const std::pair<InequalityType, unsigned> &LHS,
                       const std::pair<InequalityType, unsigned> &RHS) {
                      return LHS.second < RHS.second;
                    });
  Inequalities Result;
  for (const auto &KV : Worklist)
    Result.push_back(KV.first);
  return Result;
}

struct InequalitySimpliler : public SCEVVisitor<InequalitySimpliler, void> {
  InequalitySimpliler(const InequalityType &I, ScalarEvolution &SE)
      : Inequality(I), SE(SE) {}

  static InequalityType simplify(const InequalityType &Inequality,
                                 ScalarEvolution &SE) {
    InequalitySimpliler Simplifier(Inequality, SE);
    Simplifier.visit(Inequality.LHS);
    return Simplifier.Inequality;
  }

  void visitAddExpr(const SCEVAddExpr *S) {
    if (!S->hasNoSignedWrap())
      return;

    SmallVector<SCEVUse, 4> NewOps;
    bool Update = false;
    for (const SCEV *Op : S->operands()) {
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Op)) {
        Update = true;
        Inequality.RHS =
            Inequality.RHS - OverflowSafeSignedAPInt(C->getAPInt());
      } else {
        NewOps.push_back(Op);
      }
    }

    if (!Update)
      return;
    if (NewOps.size() == 1) {
      Inequality.LHS = NewOps[0];
      visit(Inequality.LHS);
    } else {
      Inequality.LHS = SE.getAddExpr(NewOps, S->getNoWrapFlags(), 0);
    }
  }

  void visitMulExpr(const SCEVMulExpr *S) {
    if (!S->hasNoSignedWrap())
      return;

    SmallVector<SCEVUse, 4> NewOps;
    bool Update = false;
    for (const SCEV *Op : S->operands()) {
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Op)) {
        Update = true;
        // TODO: Sign?
        Inequality.RHS =
            Inequality.RHS / OverflowSafeSignedAPInt(C->getAPInt());
      } else {
        NewOps.push_back(Op);
      }
    }

    if (!Update)
      return;
    if (NewOps.size() == 1) {
      Inequality.LHS = NewOps[0];
      visit(Inequality.LHS);
    } else {
      Inequality.LHS = SE.getMulExpr(NewOps, S->getNoWrapFlags(), 0);
    }
  }

  void visitConstant(const SCEVConstant *) {}
  void visitVScale(const SCEVVScale *) {}
  void visitSignExtendExpr(const SCEVSignExtendExpr *S) {}
  void visitSMinExpr(const SCEVSMinExpr *S) {}
  void visitSMaxExpr(const SCEVSMaxExpr *S) {}
  void visitPtrToAddrExpr(const SCEVPtrToAddrExpr *) {}
  void visitPtrToIntExpr(const SCEVPtrToIntExpr *) {}
  void visitTruncateExpr(const SCEVTruncateExpr *) {}
  void visitZeroExtendExpr(const SCEVZeroExtendExpr *) {}
  void visitUDivExpr(const SCEVUDivExpr *) {}
  void visitAddRecExpr(const SCEVAddRecExpr *) {}
  void visitUMaxExpr(const SCEVUMaxExpr *) {}
  void visitUMinExpr(const SCEVUMinExpr *) {}
  void visitUnknown(const SCEVUnknown *) {}
  void visitSequentialUMinExpr(const SCEVSequentialUMinExpr *) {}

private:
  InequalityType Inequality;
  ScalarEvolution &SE;
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

} // anonymous namespace

static const SCEV *findMaxValueAux(const SCEV *S, ExecutionDomain &ED) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (!AR)
    return S;
  if (!AR->isAffine())
    return nullptr;

  ScalarEvolution &SE = ED.getSE();
  const SCEV *Step = AR->getStepRecurrence(SE);
  const SCEV *Start = findMaxValueAux(AR->getStart(), ED);
  if (!Start)
    return nullptr;
  const SCEV *BTC = SE.getBackedgeTakenCount(AR->getLoop());
  if (!BTC)
    return nullptr;
  if (ED.isKnownNonNegative(Step))
    return SE.getAddExpr(Start, SE.getMulExpr(BTC, Step));
  if (ED.isKnownNonPositive(Step))
    return SE.getAddExpr(Start, SE.getMulExpr(BTC, Step));
  return nullptr;
}

static const SCEV *findMaxValue(const SCEV *S, ExecutionDomain &ED) {
  if (!isExecutedAtEveryIteration(S))
    return nullptr;
  return findMaxValueAux(S, ED);
}

void ExecutionDomain::smin(const SCEV *S, const APInt &RHS) {
  auto Ite = fetch(S);
  Ite->second = Ite->second.intersectWith(
      ConstantRange::makeExactICmpRegion(ICmpInst::ICMP_SLT, RHS));
}

void ExecutionDomain::smax(const SCEV *S, const APInt &RHS) {
  auto Ite = fetch(S);
  Ite->second = Ite->second.intersectWith(
      ConstantRange::makeExactICmpRegion(ICmpInst::ICMP_SGE, RHS));
}

void ExecutionDomain::clamp(const APInt &Min, const SCEV *S, const APInt &Max) {
  auto Ite = fetch(S);
  Ite->second = Ite->second.intersectWith(ConstantRange(Min, Max + 1));
}

bool ExecutionDomain::isKnownNonNegative(const SCEV *S) {
  auto Ite = Ranges.find(S);
  if (Ite != Ranges.end() && Ite->second.getSignedMin().isNonNegative())
    return true;
  return SE.isKnownNonNegative(S);
}

bool ExecutionDomain::isKnownNonPositive(const SCEV *S) {
  auto Ite = Ranges.find(S);
  if (Ite != Ranges.end() && Ite->second.getSignedMin().isNonPositive())
    return true;
  return SE.isKnownNonPositive(S);
}

void ExecutionDomain::setRange(const SCEV *S, const ConstantRange &Range) {
  auto Ite = fetch(S);
  Ite->second = Ite->second.intersectWith(Range);
}

static void printExecutionDomain(raw_ostream &OS, Function &F,
                                 ScalarEvolution &SE) {
  ExecutionDomain ED(SE);
  Inequalities Inequalities = collectInequalities(F, SE);
  for (const auto &Inequality : Inequalities) {
    OS << "Adding inequality: " << Inequality << "\n";
    const SCEV *MaxValue = findMaxValue(Inequality.LHS, ED);
    if (!MaxValue)
      continue;
    OS << "Max value: " << *MaxValue << "\n";
    InequalityType ToSolve = Inequality;
    ToSolve.LHS = MaxValue;
    ToSolve = InequalitySimpliler::simplify(ToSolve, SE);
    OS << "Solved: " << ToSolve << "\n";
    // ED.setRange(Inequality.LHS,
    // ConstantRange::makeExactICmpRegion(Inequality.Pred, *Inequality.RHS));
  }
}

ExecutionDomainPrinterPass::ExecutionDomainPrinterPass(raw_ostream &OS)
    : OS(OS) {}

PreservedAnalyses ExecutionDomainPrinterPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  printExecutionDomain(OS, F, AM.getResult<ScalarEvolutionAnalysis>(F));
  return PreservedAnalyses::all();
}
