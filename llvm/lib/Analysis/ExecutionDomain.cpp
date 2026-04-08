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
      return true;
    }

    bool isDone() const { return false; }
  };

  FindUnknown Finder;
  SCEVTraversal<FindUnknown>(Finder).visitAll(S);
  return std::move(Finder.Result);
}

namespace {

class OverflowSafeSignedAPInt {
  using Self = OverflowSafeSignedAPInt;

  /// Underlying value. std::nullopt means "unknown". An arithmetic operation on
  /// "unknown" always produces "unknown".
  std::optional<APInt> Value;

  OverflowSafeSignedAPInt fromInt(uint64_t V) const {
    assert(Value && "Value is not available.");
    return OverflowSafeSignedAPInt(
        APInt(Value->getBitWidth(), V, /*isSigned=*/true));
  }

public:
  OverflowSafeSignedAPInt() : Value(std::nullopt) {}
  OverflowSafeSignedAPInt(const APInt &V) : Value(V) {}
  OverflowSafeSignedAPInt(const std::optional<APInt> &V) : Value(V) {}

  Self &operator+=(const Self &RHS) {
    if (!Value || !RHS.Value)
      return *this = Self();
    bool Overflow;
    APInt Result = Value->sadd_ov(*RHS.Value, Overflow);
    if (Overflow)
      return *this = Self();
    Value = Result;
    return *this;
  }

  Self operator+(const Self &RHS) const {
    Self LHS = *this;
    LHS += RHS;
    return LHS;
  }

  Self operator+(int RHS) const { return *this + fromInt(RHS); }

  Self &operator+=(int RHS) { return *this += fromInt(RHS); }

  Self &operator-=(const Self &RHS) {
    if (!Value || !RHS.Value)
      return *this = Self();
    bool Overflow;
    APInt Result = Value->ssub_ov(*RHS.Value, Overflow);
    if (Overflow)
      return *this = Self();
    Value = Result;
    return *this;
  }

  Self operator-(const Self &RHS) const {
    Self LHS = *this;
    LHS -= RHS;
    return LHS;
  }

  Self &operator-=(int RHS) { return *this -= fromInt(RHS); }

  Self operator-(int RHS) const { return *this - fromInt(RHS); }

  Self operator*(const Self &RHS) const {
    if (!Value || !RHS.Value)
      return Self();
    bool Overflow;
    APInt Result = Value->smul_ov(*RHS.Value, Overflow);
    if (Overflow)
      return Self();
    return Self(Result);
  }

  Self operator-() const {
    if (!Value)
      return Self();
    if (Value->isMinSignedValue())
      return Self();
    return Self(-*Value);
  }

  Self operator/(const Self &RHS) const {
    if (!Value || !RHS.Value)
      return Self();
    if (RHS.Value->isZero())
      return Self();
    bool Overflow = false;
    APInt Res = Value->sdiv_ov(*RHS.Value, Overflow);
    if (Overflow)
      return Self();
    return Self(Res);
  }

  std::pair<Self, Self> sdivrem(const Self &RHS) const {
    Self Q = (*this) / RHS;
    Self R = *this - Q * RHS;
    return std::make_pair(Q, R);
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

  bool isPositive() const { return Value && Value->isStrictlyPositive(); }
  bool isNonNegative() const { return Value && Value->isNonNegative(); }
  bool isZero() const { return Value && Value->isZero(); }
  bool isNonZero() const { return Value && !Value->isZero(); }
};

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

  stable_sort(Worklist, [](const std::pair<InequalityType, unsigned> &LHS,
                           const std::pair<InequalityType, unsigned> &RHS) {
    return LHS.second < RHS.second;
  });
  Inequalities Result;
  for (const auto &KV : Worklist)
    Result.push_back(KV.first);
  return Result;
}

struct ExecutionDomainInterpreter
    : public SCEVVisitor<ExecutionDomainInterpreter, ConstantRange> {
  using Base = SCEVVisitor<ExecutionDomainInterpreter, ConstantRange>;

  ExecutionDomainInterpreter(ExecutionDomain &ED) : ED(ED) {}

  static ConstantRange evaluate(const SCEV *S, ExecutionDomain &ED) {
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

struct InequalitySimpliler : public SCEVVisitor<InequalitySimpliler, void> {
  using Base = SCEVVisitor<InequalitySimpliler, void>;

  InequalitySimpliler(const InequalityType &Inequality, ExecutionDomain &ED)
      : ED(ED), Pred(Inequality.Pred), LHS(Inequality.LHS),
        RHS(Inequality.RHS) {}

  static std::optional<InequalityType>
  simplify(const InequalityType &Inequality, ExecutionDomain &ED) {
    InequalitySimpliler Simplifier(Inequality, ED);
    Simplifier.visit(Inequality.LHS);
    if (Simplifier.RHS)
      return InequalityType(Simplifier.Pred, Simplifier.LHS, *Simplifier.RHS);
    return std::nullopt;
  }

  void visit(const SCEV *S) {
    if (!RHS) {
      LLVM_DEBUG(dbgs() << "  Failed to simplify...\n");
      return;
    }
    InequalityType Cur(Pred, S, *RHS);
    Base::visit(S);
  }

  void visitAddExpr(const SCEVAddExpr *S) {
    // https://alive2.llvm.org/ce/z/rEsNYQ
    // (X0 + ... + Xn) + C0 cmp C1 -->  X0 + ... + Xn cmp C1 - C0
    SmallVector<SCEVUse, 1> NewOps;
    OverflowSafeSignedAPInt C;
    for (const SCEV *Op : S->operands()) {
      if (const SCEVConstant *COp = dyn_cast<SCEVConstant>(Op)) {
        C = COp->getAPInt();
      } else {
        NewOps.push_back(Op);
      }
    }

    if (!C)
      return;

    bool NoWrap = [&] {
      if (S->hasNoSignedWrap())
        return true;
      ConstantRange Range =
          ConstantRange(APInt::getZero(S->getType()->getIntegerBitWidth()));
      for (const SCEV *Op : NewOps) {
        ConstantRange Other = ExecutionDomainInterpreter::evaluate(Op, ED);
        if (Range.signedAddMayOverflow(Other) !=
            ConstantRange::OverflowResult::NeverOverflows)
          return false;
        Range = Range.add(ExecutionDomainInterpreter::evaluate(Op, ED));
      }
      return true;
    }();

    if (!NoWrap)
      return;

    RHS -= C;
    if (NewOps.size() == 1) {
      LHS = NewOps[0];
      visit(LHS);
    } else {
      LHS = ED.getSE().getAddExpr(NewOps, S->getNoWrapFlags(), 0);
    }
  }

  void visitMulExpr(const SCEVMulExpr *S) {
    SmallVector<SCEVUse, 1> NewOps;
    OverflowSafeSignedAPInt C;
    for (const SCEV *Op : S->operands()) {
      if (const SCEVConstant *COp = dyn_cast<SCEVConstant>(Op)) {
        C = COp->getAPInt();
      } else {
        NewOps.push_back(Op);
      }
    }

    if (!C)
      return;
    bool NoWrap = [&] {
      if (S->hasNoSignedWrap())
        return true;
      ConstantRange Range = ConstantRange(
          APInt(S->getType()->getIntegerBitWidth(), 1, true, false));
      for (const SCEV *Op : NewOps) {
        Range = Range.smul_fast(ExecutionDomainInterpreter::evaluate(Op, ED));
        if (Range.isFullSet())
          return false;
      }
      return true;
    }();

    if (!NoWrap)
      return;

    if (!RHS.isPositive())
      return;
    auto [Q, R] = RHS.sdivrem(C);
    bool Update = false;
    if (R.isZero()) {
      // https://alive2.llvm.org/ce/z/fPnXoS
      RHS = Q;
      Update = true;
    } else if (C.isPositive()) {
      assert(Pred == ICmpInst::ICMP_SLT || Pred == ICmpInst::ICMP_SGT);
      if (Pred == ICmpInst::ICMP_SLT) {
        if (C.isPositive() && RHS.isNonNegative() && Q.isNonZero()) {
          // https://alive2.llvm.org/ce/z/JyeHkD
          RHS = Q + 1;
          Update = true;
        }
      } else {
        // https://alive2.llvm.org/ce/z/osWHWw
        RHS = Q;
        Update = true;
      }
    }

    if (!Update) {
      RHS = OverflowSafeSignedAPInt();
      return;
    }
    if (NewOps.size() == 1) {
      LHS = NewOps[0];
      visit(LHS);
    } else {
      LHS = ED.getSE().getMulExpr(NewOps, S->getNoWrapFlags(), 0);
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
  ExecutionDomain &ED;
  CmpPredicate Pred;
  const SCEV *LHS;
  OverflowSafeSignedAPInt RHS;
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

static SmallVector<InequalityType, 2>
canonicalizeInequality(InequalityType Inequality, ExecutionDomain &ED) {
  SmallVector<InequalityType, 2> Result;
  auto Push = [&Result](InequalityType I) {
    switch (I.Pred) {
    case ICmpInst::ICMP_SLE: {
      OverflowSafeSignedAPInt Tmp(I.RHS);
      Tmp += 1;
      if (!Tmp)
        break;
      I.Pred = ICmpInst::ICMP_SLT;
      I.RHS = *Tmp;
      [[fallthrough]];
    }
    case ICmpInst::ICMP_SLT:
      Result.push_back(I);
      break;
    case ICmpInst::ICMP_SGE: {
      OverflowSafeSignedAPInt Tmp(I.RHS);
      Tmp -= 1;
      if (!Tmp)
        break;
      I.Pred = ICmpInst::ICMP_SGT;
      I.RHS = *Tmp;
      [[fallthrough]];
    }
    case ICmpInst::ICMP_SGT:
      Result.push_back(I);
      break;
    default:
      llvm_unreachable("Unexpected predicate");
    }
  };

  switch (Inequality.Pred) {
  case ICmpInst::ICMP_SLT:
  case ICmpInst::ICMP_SLE:
  case ICmpInst::ICMP_SGT:
  case ICmpInst::ICMP_SGE:
    Push(Inequality);
    break;
  case ICmpInst::ICMP_ULT: {
    // https://alive2.llvm.org/ce/z/TworUD
    // X <u C --> X <=u C - 1
    if (Inequality.RHS.isZero())
      break;
    Inequality.Pred = ICmpInst::ICMP_ULE;
    OverflowSafeSignedAPInt Tmp(Inequality.RHS);
    Tmp -= 1;
    if (!Tmp)
      break;
    Inequality.RHS = *Tmp;
    [[fallthrough]];
  }
  case ICmpInst::ICMP_ULE: {
    // https://alive2.llvm.org/ce/z/ZaXVPV
    // X <=u C --> 0 <=s X && X <=s C
    APInt SignedMax = APInt::getSignedMaxValue(Inequality.RHS.getBitWidth());
    if (!Inequality.RHS.ult(SignedMax))
      break;
    Inequality.Pred = ICmpInst::ICMP_SLE;
    Push(Inequality);
    Push(InequalityType(ICmpInst::ICMP_SGE, Inequality.LHS,
                        APInt::getZero(Inequality.RHS.getBitWidth())));
    break;
  }
  default:
    break;
  }
  for (InequalityType &I : Result)
    I = InequalitySimpliler::simplify(I, ED).value_or(I);
  return Result;
}

void ExecutionDomain::addInequality(const InequalityType &Inequality) {
  SmallVector<InequalityType, 2> Canonicalized =
      canonicalizeInequality(Inequality, *this);
  for (const InequalityType &I : Canonicalized) {
    auto &Inequalities = Contexts[I.LHS];
    auto [Ite, Inserted] = Inequalities.try_emplace(I.Pred, I);
    if (Inserted)
      continue;
    APInt &RHS = Ite->second.RHS;
    switch (I.Pred) {
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE:
      RHS = APIntOps::smin(RHS, I.RHS);
      break;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE:
      RHS = APIntOps::smax(RHS, I.RHS);
      break;
    default:
      llvm_unreachable("Unexpected predicate");
    }
  }
}

ConstantRange ExecutionDomain::withContext(const SCEV *S, ConstantRange Range) {
  auto Ite = Contexts.find(S);
  if (Ite == Contexts.end())
    return Range;
  for (const auto &[Pred, Inequality] : Ite->second) {
    assert(Inequality.LHS == S);
    assert(Inequality.Pred == Pred);
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
  SmallVector<const SCEV *, 4> SortedSCEVs;
  for (const auto &[S, Inequalities] : Contexts)
    SortedSCEVs.push_back(S);
  stable_sort(SortedSCEVs,
              [](const SCEV *LHS, const SCEV *RHS) { return LHS < RHS; });

  for (const SCEV *S : SortedSCEVs) {
    OS << "Context for " << *S << ":\n";
    SmallVector<InequalityType, 4> SortedInequalities;
    for (const auto &[Pred, Inequality] : Contexts[S])
      SortedInequalities.push_back(Inequality);
    stable_sort(SortedInequalities,
                [](const InequalityType &LHS, const InequalityType &RHS) {
                  return LHS.Pred < RHS.Pred;
                });
    for (const InequalityType &Inequality : SortedInequalities)
      OS << "  " << Inequality << "\n";
  }

  for (const SCEV *S : SortedSCEVs) {
    ConstantRange Range = ExecutionDomainInterpreter::evaluate(S, *this);
    OS << "Range for " << *S << ": " << Range << "\n";
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

  auto Inequalities = collectInequalities(F, SE);
  for (InequalityType &Inequality : Inequalities) {
    const SCEV *Max = findMaxValue(Inequality.LHS, ED);
    if (Max) {
      Inequality.LHS = Max;
      ED.addInequality(Inequality);
    }
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
