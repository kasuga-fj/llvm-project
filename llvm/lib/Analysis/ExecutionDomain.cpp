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
  InequalitySimpliler(const InequalityType &Inequality, ScalarEvolution &SE)
      : SE(SE), Pred(Inequality.Pred), LHS(Inequality.LHS),
        RHS(Inequality.RHS) {}

  static std::optional<InequalityType>
  simplify(const InequalityType &Inequality, ScalarEvolution &SE) {
    InequalitySimpliler Simplifier(Inequality, SE);
    Simplifier.visit(Inequality.LHS);
    if (Simplifier.RHS)
      return InequalityType{Simplifier.Pred, Simplifier.LHS, *Simplifier.RHS};
    return std::nullopt;
  }

  void visitAddExpr(const SCEVAddExpr *S) {
    // if (!S->hasNoSignedWrap())
    //   return;

    SmallVector<SCEVUse, 4> NewOps;
    bool Update = false;
    for (const SCEV *Op : S->operands()) {
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Op)) {
        Update = true;
        RHS = RHS - OverflowSafeSignedAPInt(C->getAPInt());
      } else {
        NewOps.push_back(Op);
      }
    }

    if (!Update)
      return;
    if (NewOps.size() == 1) {
      LHS = NewOps[0];
      visit(LHS);
    } else {
      LHS = SE.getAddExpr(NewOps, S->getNoWrapFlags(), 0);
    }
  }

  void visitMulExpr(const SCEVMulExpr *S) {
    // if (!S->hasNoSignedWrap())
    //   return;

    SmallVector<SCEVUse, 4> NewOps;
    bool Update = false;
    for (const SCEV *Op : S->operands()) {
      if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Op)) {
        Update = true;
        // TODO: Sign?
        RHS = RHS / OverflowSafeSignedAPInt(C->getAPInt());
      } else {
        NewOps.push_back(Op);
      }
    }

    if (!Update)
      return;
    if (NewOps.size() == 1) {
      LHS = NewOps[0];
      visit(LHS);
    } else {
      LHS = SE.getMulExpr(NewOps, S->getNoWrapFlags(), 0);
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
  ScalarEvolution &SE;
  CmpPredicate Pred;
  const SCEV *LHS;
  OverflowSafeSignedAPInt RHS;
};

struct ExecutionDomainInterpreter
    : public SCEVVisitor<ExecutionDomainInterpreter, ConstantRange> {
  using Base = SCEVVisitor<ExecutionDomainInterpreter, ConstantRange>;

  ExecutionDomainInterpreter(ExecutionDomain &ED) : ED(ED) {}

  static ConstantRange evaluate(const SCEV *S, ExecutionDomain &ED) {
    dbgs() << "Evaluating " << *S << "\n";
    return ExecutionDomainInterpreter(ED).visit(S);
  }

  ConstantRange visit(const SCEV *S) {
    ConstantRange Res = Base::visit(S);
    return ED.withContext(S, Res);
  }

  ConstantRange visitConstant(const SCEVConstant *C) {
    return ConstantRange(C->getAPInt());
  }

  ConstantRange visitAddExpr(const SCEVAddExpr *S) {
    ConstantRange Result(APInt::getZero(S->getType()->getIntegerBitWidth()));
    for (const SCEV *Op : S->operands())
      Result = Result.add(visit(Op));
    return Result;
  }

  ConstantRange visitMulExpr(const SCEVMulExpr *S) {
    ConstantRange Result(
        APInt(S->getType()->getIntegerBitWidth(), 1, true, false));
    for (const SCEV *Op : S->operands())
      Result = Result.smul_fast(visit(Op));
    return Result;
  }

  ConstantRange visitSignExtendExpr(const SCEVSignExtendExpr *S) {
    ConstantRange Result =
        visit(S->getOperand()).signExtend(S->getType()->getIntegerBitWidth());
    return Result;
  }

  ConstantRange visitSMinExpr(const SCEVSMinExpr *S) {
    ConstantRange Result = visit(S->getOperand(0));
    for (unsigned I = 1, E = S->getNumOperands(); I != E; ++I)
      Result = Result.smin(visit(S->getOperand(I)));
    return Result;
  }

  ConstantRange visitSMaxExpr(const SCEVSMaxExpr *S) {
    ConstantRange Result = visit(S->getOperand(0));
    for (unsigned I = 1, E = S->getNumOperands(); I != E; ++I)
      Result = Result.smax(visit(S->getOperand(I)));
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
    return ED.getSE().getSignedRange(S);
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
  dbgs() << "Could not determine the sign of the step: " << *Step << "\n";
  return nullptr;
}

static const SCEV *findMaxValue(const SCEV *S, ExecutionDomain &ED) {
  if (!isExecutedAtEveryIteration(S))
    return nullptr;
  return findMaxValueAux(S, ED);
}

static SmallVector<InequalityType, 2>
canonicalizeInequality(InequalityType Inequality, ScalarEvolution &SE) {
  SmallVector<InequalityType, 2> Result;
  switch (Inequality.Pred) {
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_SLE:
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_SGE:
    Result.push_back(Inequality);
    break;
  case ICmpInst::ICMP_ULT:
    // X <u C
    // TODO?: When C = 0.
    Inequality.Pred = ICmpInst::ICMP_ULE;
    Inequality.RHS = Inequality.RHS - 1;
    [[fallthrough]];
  case ICmpInst::ICMP_ULE: {
    APInt SignedMax = APInt::getSignedMaxValue(Inequality.RHS.getBitWidth());
    if (SignedMax.ult(Inequality.RHS))
      break;
    Inequality.Pred = ICmpInst::ICMP_SLE;
    Result.push_back(Inequality);
    Result.push_back({ICmpInst::ICMP_SGE, Inequality.LHS,
                      APInt::getZero(Inequality.RHS.getBitWidth())});
    break;
  }
  default:
    break;
  }
  for (InequalityType &I : Result) {
    I = InequalitySimpliler::simplify(I, SE).value_or(I);
    // dbgs() << "Canonicalized inequality: " << I << "\n";
  }
  return Result;
}

void ExecutionDomain::addInequality(const InequalityType &Inequality) {
  SmallVector<InequalityType, 2> Canonicalized =
      canonicalizeInequality(Inequality, SE);
  for (const InequalityType &I : Canonicalized) {
    if (!Contexts.contains(I.LHS)) {
      Contexts[I.LHS] = std::make_unique<ExecutionContext>(I.LHS);
    }
    Contexts[I.LHS]->Inequalities.push_back(I);
  }
}

ConstantRange ExecutionDomain::withContext(const SCEV *S, ConstantRange Range) {
  auto Ite = Contexts.find(S);
  if (Ite == Contexts.end())
    return Range;
  for (const InequalityType &Inequality : Ite->second->Inequalities) {
    assert(Inequality.LHS == S);
    switch (Inequality.Pred) {
    case ICmpInst::ICMP_SLT:
      Range = Range.smin(Inequality.RHS - 1);
      break;
    case ICmpInst::ICMP_SLE:
      Range = Range.smin(Inequality.RHS);
      break;
    case ICmpInst::ICMP_SGT:
      Range = Range.smax(Inequality.RHS + 1);
      break;
    case ICmpInst::ICMP_SGE:
      Range = Range.smax(Inequality.RHS);
      break;
    default:
      llvm_unreachable("Unexpected predicate");
    }
  }
  return Range;
}

bool ExecutionDomain::isKnownNonNegative(const SCEV *S) {
  ConstantRange Range = ExecutionDomainInterpreter::evaluate(S, *this);
  if (!Range.isFullSet() && !Range.isSignWrappedSet() &&
      Range.getSignedMin().isNonNegative())
    return true;
  return SE.isKnownNonNegative(S);
}

bool ExecutionDomain::isKnownNonPositive(const SCEV *S) {
  ConstantRange Range = ExecutionDomainInterpreter::evaluate(S, *this);
  if (!Range.isFullSet() && !Range.isSignWrappedSet() &&
      Range.getSignedMax().isNonPositive())
    return true;
  return SE.isKnownNonPositive(S);
}

void ExecutionDomain::print(raw_ostream &OS) {
  for (const auto &[S, Context] : Contexts) {
    OS << "Context for " << *S << ":\n";
    for (const InequalityType &Inequality : Context->Inequalities) {
      OS << "  ";
      Inequality.print(OS);
      OS << "\n";
    }
    ConstantRange R = ExecutionDomainInterpreter::evaluate(S, *this);
    OS << "  Range: " << R << "\n";
  }
}

static void traverseLoop(const Loop *L, ExecutionDomain &ED) {
  ScalarEvolution &SE = ED.getSE();
  if (SE.hasLoopInvariantBackedgeTakenCount(L)) {
    const SCEV *BTC = SE.getBackedgeTakenCount(L);
    const SCEVConstant *Max =
        dyn_cast<SCEVConstant>(SE.getConstantMaxBackedgeTakenCount(L));
    InequalityType Inequality(ICmpInst::ICMP_ULE, BTC, Max->getAPInt());
    ED.addInequality(Inequality);
  }
  for (const Loop *Sub : *L)
    traverseLoop(Sub, ED);
}

static void printExecutionDomain(raw_ostream &OS, Function &F,
                                 ScalarEvolution &SE, LoopInfo &LI) {
  ExecutionDomain ED(SE);

  for (const Loop *L : LI)
    traverseLoop(L, ED);

  ED.print(OS);

  auto Inequalities = collectInequalities(F, SE);
  for (InequalityType &Inequality : Inequalities) {
    const SCEV *Max = findMaxValue(Inequality.LHS, ED);
    if (Max) {
      dbgs() << "Found max value for " << *Inequality.LHS << ": " << *Max
             << "\n";
      Inequality.LHS = Max;
    } else {
      dbgs() << "Could not find max value for " << *Inequality.LHS << "\n";
    }
    ED.addInequality(Inequality);
  }
  ED.print(OS);
}

ExecutionDomainPrinterPass::ExecutionDomainPrinterPass(raw_ostream &OS)
    : OS(OS) {}

PreservedAnalyses ExecutionDomainPrinterPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  printExecutionDomain(OS, F, AM.getResult<ScalarEvolutionAnalysis>(F),
                       AM.getResult<LoopAnalysis>(F));
  return PreservedAnalyses::all();
}
