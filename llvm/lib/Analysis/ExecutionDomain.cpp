#include "llvm/Analysis/ExecutionDomain.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "execution-domain"

using namespace llvm;

namespace {

enum class TestType {
  ExecutionDomain,
  LoopInspect,
  InequalitySimplification,
};

} // anonymous namespace

static cl::opt<TestType> LitTestType(
    "execution-domain-test", cl::MiscFlags::CommaSeparated, cl::Hidden,
    cl::desc("Component that runs in the printer pass"),
    cl::init(TestType::ExecutionDomain),
    cl::values(clEnumValN(TestType::ExecutionDomain, "execution-domain",
                          "ExecutionDomain"),
               clEnumValN(TestType::LoopInspect, "loop-inspect",
                          "LoopNestInspector"),
               clEnumValN(TestType::InequalitySimplification,
                          "inequality-simplification", "InequalitySimpliler")));

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

struct MemAccessConstraint {
  Value *Ptr;
  APInt DerefBytes;
  unsigned Complexity;
  const Loop *Outermost;
};

using MemAccessConstraints = SmallVector<MemAccessConstraint, 4>;

MemAccessConstraints collectConstraints(Function &F, ScalarEvolution &SE,
                                        const LoopInfo &LI) {
  MemAccessConstraints Result;
  for (Instruction &Inst : instructions(F)) {
    if (!isa<LoadInst, StoreInst>(&Inst))
      continue;

    const BasicBlock *BB = Inst.getParent();
    const Loop *L = LI.getLoopFor(BB);
    if (!L)
      continue;
    const Loop *Outermost = L->getOutermostLoop();
    Value *Ptr = getPointerOperand(&Inst);
    const SCEV *AccessFn = SE.getSCEV(getPointerOperand(&Inst));
    const SCEVUnknown *BasePointer =
        dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFn));
    if (!BasePointer)
      continue;
    AccessFn = SE.getMinusSCEV(AccessFn, BasePointer);

    Value *BasePtr = BasePointer->getValue();
    const DataLayout &DL = F.getDataLayout();
    bool CheckForNonNull, CheckForFreed;
    // TODO: We want the upper bound, not the lower bound.
    uint64_t DerefBytes = BasePtr->getPointerDereferenceableBytes(
        DL, CheckForNonNull, CheckForFreed);
    if (DerefBytes && !CheckForNonNull && !CheckForFreed) {
      APInt RHS = APInt(DL.getIndexSize(0) * 8, DerefBytes, false, false);
      unsigned Complexity = computeComplexity(AccessFn).size();
      Result.push_back(MemAccessConstraint{Ptr, RHS, Complexity, Outermost});
    }
  }

  stable_sort(Result, [](const MemAccessConstraint &LHS,
                         const MemAccessConstraint &RHS) {
    return LHS.Complexity < RHS.Complexity;
  });
  return Result;
}

struct InequalitySimpliler : public SCEVVisitor<InequalitySimpliler, void> {
  using Base = SCEVVisitor<InequalitySimpliler, void>;

  InequalitySimpliler(const InequalityType &Inequality, ExecutionDomain &ED)
      : ED(ED), Inequality(Inequality) {}

  static InequalityType simplify(const InequalityType &Inequality, ExecutionDomain &ED) {
    InequalitySimpliler Simplifier(Inequality, ED);
    Simplifier.visit(Inequality.LHS);
    return Simplifier.Inequality;
  }

  void visitAddExpr(const SCEVAddExpr *S) {
    // https://alive2.llvm.org/ce/z/a3aJ8z
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
      ScalarEvolution &SE = ED.getSE();
      const SCEV *Acc = SE.getZero(S->getType());
      for (const SCEV *Op : S->operands()) {
        const SCEV *RewrittenOp = ED.rewrite(Op);
        if (!SE.willNotOverflow(Instruction::Add, /*Signed=*/true, ED.rewrite(Acc), RewrittenOp))
          return false;
        Acc = SE.getAddExpr(Acc, Op);
      }
      return true;
    }();

    if (!NoWrap)
      return;

    OverflowSafeSignedAPInt RHS(Inequality.RHS);
    RHS -= C;
    if (!RHS)
      return;
    Inequality.RHS = *RHS;
    if (NewOps.size() == 1) {
      Inequality.LHS = NewOps[0];
      visit(Inequality.LHS);
    } else {
      Inequality.LHS = ED.getSE().getAddExpr(NewOps, S->getNoWrapFlags(), 0);
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
      ScalarEvolution &SE = ED.getSE();
      ConstantRange Acc = ConstantRange(
          APInt(S->getType()->getIntegerBitWidth(), 1, true, true));
      for (const SCEV *Op : S->operands()) {
        const SCEV *RewrittenOp = ED.rewrite(Op);
        ConstantRange Val = SE.getSignedRange(RewrittenOp);
        Acc = Acc.smul_fast(Val);
        if (Acc.isFullSet())
          return false;
      }
      return true;
    }();

    if (!NoWrap)
      return;

    if (!Inequality.RHS.isStrictlyPositive())
      return;
    APInt Q, R;
    APInt::sdivrem(Inequality.RHS, *C, Q, R);
    bool Update = false;
    if (R.isZero()) {
      // https://alive2.llvm.org/ce/z/fPnXoS
      Inequality.RHS = Q;
      Update = true;
    } else if (C.isPositive()) {
      assert(Inequality.Pred == ICmpInst::ICMP_SLE || Inequality.Pred == ICmpInst::ICMP_SGE);
      if (Inequality.Pred == ICmpInst::ICMP_SLE) {
        if (C.isPositive() && Inequality.RHS.isNonNegative() && !Q.isZero()) {
          // https://alive2.llvm.org/ce/z/f9prP4
          Inequality.RHS = Q;
          Update = true;
        }
      } else {
        // https://alive2.llvm.org/ce/z/osWHWw
        Inequality.RHS = Q;
        Update = true;
      }
    }

    if (!Update)
      return;

    if (NewOps.size() == 1) {
      Inequality.LHS = NewOps[0];
      visit(Inequality.LHS);
    } else {
      Inequality.LHS = ED.getSE().getMulExpr(NewOps, S->getNoWrapFlags(), 0);
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
  InequalityType Inequality;
};

struct MulExpander : public SCEVVisitor<MulExpander, SmallVector<const SCEV *, 4>> {
  using ResultType = SmallVector<const SCEV *, 4>;

  MulExpander(ScalarEvolution &SE) : SE(SE) {}

  static ResultType expand(const SCEV *S, ScalarEvolution &SE) {
    return MulExpander(SE).visit(S);
  }

  ResultType visitAddExpr(const SCEVAddExpr *S) {
    ResultType Result;
    for (const SCEV *Op : S->operands()) {
      ResultType Res = visit(Op);
      assert(!Res.empty() && "visit must return at least one candidate.");
      Result.append(Res.begin(), Res.end());
    }
    return Result;
  }

  ResultType visitMulExpr(const SCEVMulExpr *S) {
    SmallVector<ResultType, 4> Cands;
    for (const SCEV *Op : S->operands()) {
      ResultType Res = visit(Op);
      assert(!Res.empty() && "visit must return at least one candidate.");
      Cands.push_back(std::move(Res));
    }
    ResultType Result;
    mulDfs(Cands, 0, SE.getOne(S->getType()), Result);
    return Result;
  }

  ResultType visitConstant(const SCEVConstant *S) { return {S}; }
  ResultType visitVScale(const SCEVVScale *S) { return {S}; }
  ResultType visitSignExtendExpr(const SCEVSignExtendExpr *S) { return {S}; }
  ResultType visitSMinExpr(const SCEVSMinExpr *S) { return {S}; }
  ResultType visitSMaxExpr(const SCEVSMaxExpr *S) { return {S}; }
  ResultType visitPtrToAddrExpr(const SCEVPtrToAddrExpr *S) { return {S}; }
  ResultType visitPtrToIntExpr(const SCEVPtrToIntExpr *S) { return {S}; }
  ResultType visitTruncateExpr(const SCEVTruncateExpr *S) { return {S}; }
  ResultType visitZeroExtendExpr(const SCEVZeroExtendExpr *S) { return {S}; }
  ResultType visitUDivExpr(const SCEVUDivExpr *S) { return {S}; }
  ResultType visitAddRecExpr(const SCEVAddRecExpr *S) { return {S}; }
  ResultType visitUMaxExpr(const SCEVUMaxExpr *S) { return {S}; }
  ResultType visitUMinExpr(const SCEVUMinExpr *S) { return {S}; }
  ResultType visitUnknown(const SCEVUnknown *S) { return {S}; }
  ResultType visitSequentialUMinExpr(const SCEVSequentialUMinExpr *S) { return {S}; }

private:
  ScalarEvolution &SE;

  void mulDfs(ArrayRef<ResultType> Cands, unsigned Idx, const SCEV *Acc, ResultType &Result) {
    if (Idx == Cands.size()) {
      Result.push_back(Acc);
      return;
    }
    for (const SCEV *Op : Cands[Idx])
      mulDfs(Cands, Idx + 1, SE.getMulExpr(Op, Acc), Result);
  }
};

} // anonymous namespace

bool LoopNestInspector::isPtrUsedAtEveryIteration(Value *Ptr, const Loop *L) {
  auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP)
    return false;

  if (L->getHeader() == L->getLoopLatch())
    return true;

  for (User *U : GEP->users()) {
    if (getLoadStorePointerOperand(U) != GEP)
      continue;
    const BasicBlock *UserBB = cast<Instruction>(U)->getParent();
    if (!L->contains(UserBB))
      continue;
    const BasicBlock *Latch = L->getLoopLatch();
    if (DT.dominates(UserBB, Latch))
      return true;
  }

  return false;
}

bool LoopNestInspector::isInnerLoopExecutedAtEveryIteration(const Loop *Inner) {
  auto Ite = Checked.find(Inner);
  if (Ite != Checked.end())
    return Ite->second;

  auto Res = [&] {
    const Loop *Outer = Inner->getParentLoop();
    assert(Outer && "Inner loop must have a parent loop.");
    if (!SE.hasLoopInvariantBackedgeTakenCount(Outer))
      return false;

    const BasicBlock *OuterLatch = Outer->getLoopLatch();
    if (!OuterLatch)
      return false;
    const BasicBlock *InnerHeader = Inner->getHeader();
    if (!DT.dominates(InnerHeader, OuterLatch))
      return false;
    return true;
  }();
  return Checked[Inner] = Res;
}

bool LoopNestInspector::isSafeToEstimateMaxOffsetValueRec(
    const SCEV *S, const Loop *Inner, const Loop *Outermost) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (!AR)
    return SE.isLoopInvariant(S, Outermost);

  if (!AR->isAffine())
    return false;

  const Loop *Outer = Inner->getParentLoop();
  assert(Outer && "Inner loop must have a parent loop.");
  if (!isInnerLoopExecutedAtEveryIteration(Inner))
    return false;

  const SCEV *Next = AR->getLoop() == Outer ? AR->getStart() : AR;
  return isSafeToEstimateMaxOffsetValueRec(Next, Outer, Outermost);
}

bool LoopNestInspector::isSafeToEstimateMaxOffsetValue(Value *Ptr,
                                                       const Loop *Outermost) {
  const SCEV *S = SE.removePointerBase(SE.getSCEV(Ptr));
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (!AR)
    return SE.isLoopInvariant(S, Outermost);

  if (!AR->isAffine())
    return false;

  if (!isPtrUsedAtEveryIteration(Ptr, AR->getLoop()))
    return false;
  return isSafeToEstimateMaxOffsetValueRec(AR->getStart(), AR->getLoop(),
                                           Outermost);
}

static const SCEV *estimateMaxOffsetValueAux(const SCEV *S,
                                             ExecutionDomain &ED) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (!AR)
    return S;
  if (!AR->isAffine())
    return nullptr;

  ScalarEvolution &SE = ED.getSE();
  const SCEV *Step = AR->getStepRecurrence(SE);
  const SCEV *Start = estimateMaxOffsetValueAux(AR->getStart(), ED);
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

static const SCEV *estimateMaxOffsetValue(Value *Ptr, const Loop *OutermostLoop,
                                          ExecutionDomain &ED,
                                          LoopNestInspector &LNI) {
  ScalarEvolution &SE = ED.getSE();
  if (!LNI.isSafeToEstimateMaxOffsetValue(Ptr, OutermostLoop))
    return nullptr;
  const SCEV *S = SE.removePointerBase(SE.getSCEV(Ptr));
  return estimateMaxOffsetValueAux(S, ED);
}

static std::optional<InequalityType> tryIntoSE(InequalityType I) {
  switch (I.Pred) {
    case ICmpInst::ICMP_SLT: {
                               OverflowSafeSignedAPInt Tmp(I.RHS);
                               Tmp -= 1;
                               if (!Tmp)
                                 break;
                               I.Pred = ICmpInst::ICMP_SLE;
                               I.RHS = *Tmp;
                               [[fallthrough]];
                             }
    case ICmpInst::ICMP_SLE:
                             return I;
    case ICmpInst::ICMP_SGT: {
                               OverflowSafeSignedAPInt Tmp(I.RHS);
                               Tmp += 1;
                               if (!Tmp)
                                 break;
                               I.Pred = ICmpInst::ICMP_SGE;
                               I.RHS = *Tmp;
                               [[fallthrough]];
                             }
    case ICmpInst::ICMP_SGE:
                             return I;
    default:
                             break;
  }
  return std::nullopt;
}

static SmallVector<InequalityType, 2>
canonicalizeInequality(InequalityType Inequality, ExecutionDomain &ED) {
  SmallVector<InequalityType, 2> Result;
  auto Push = [&Result](InequalityType I) {
    if (isa<SCEVConstant>(I.LHS))
      return;
    if (std::optional<InequalityType> SEI = tryIntoSE(I))
      Result.push_back(*SEI);
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
    I = InequalitySimpliler::simplify(I, ED);
  return Result;
}

ExecutionDomainRewriter::ExecutionDomainRewriter(const ExecutionDomain &ED) : Base(ED.getSE()), ED(ED) {}

static std::tuple<SCEVTypes, const SCEV *, const SCEV *> decomposeSCEV(const SCEV *S, ScalarEvolution &SE) {
  SCEVTypes Type = S->getSCEVType();
  const SCEVNAryExpr *NAry = nullptr;
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    NAry = Add;
  } else if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(S)) {
    NAry = Mul;
  }
  if (!NAry)
    return std::make_tuple(Type, S, nullptr);

  SmallVector<SCEVUse, 2> Ops;
  const SCEV *Const = nullptr;
  for (const SCEV *Op : NAry->operands()) {
    if (const SCEVConstant *C = dyn_cast<SCEVConstant>(Op)) {
      assert(!Const && "Multiple constants in an Add/Mul expression?");
      Const = C;
    } else {
      Ops.push_back(Op);
    }
  }

  const SCEV *Base = [&] {
    switch (Type) {
      case scAddExpr:
        return SE.getAddExpr(Ops, NAry->getNoWrapFlags());
      case scMulExpr:
        return SE.getMulExpr(Ops, NAry->getNoWrapFlags());
      default:
        llvm_unreachable("Unexpected SCEV type");
    }
  }();
  return std::make_tuple(Type, Base, Const);
}

const SCEV *ExecutionDomainRewriter::visit(const SCEV *S) {
  auto [Type, Base, Const] = decomposeSCEV(S, SE);
  const SCEV *Res = Base::visit(Base);
  auto Ite = ED.Contexts.find(Base);
  if (Ite == ED.Contexts.end())
    return Res;
  for (const auto &[Pred, Inequality] : Ite->second) {
    assert(Inequality.LHS == Base);
    assert(Inequality.Pred == Pred);
    switch (Inequality.Pred) {
      case ICmpInst::ICMP_SLE:
        Res = SE.getSMinExpr(Res, SE.getConstant(Inequality.RHS));
        break;
      case ICmpInst::ICMP_SGE:
        Res = SE.getSMaxExpr(Res, SE.getConstant(Inequality.RHS));
        break;
      default:
        llvm_unreachable("Unexpected predicate");
    }
  }
  switch (Type) {
    case scAddExpr:
      if (Const)
        Res = SE.getAddExpr(Res, Const);
      break;
    case scMulExpr:
      if (Const)
        Res = SE.getMulExpr(Res, Const);
      break;
    default:
      break;
  }
  return Res;
}

ExecutionDomain::ExecutionDomain(ScalarEvolution &SE) : SE(SE) {}

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
    const SCEV *Rewritten = rewrite(S);
    ConstantRange Range = SE.getSignedRange(Rewritten);
    OS << "Rewritten SCEV for " << *S << "  -->  " << *Rewritten << ": "
       << Range << "\n";
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

const SCEV *ExecutionDomain::rewrite(const SCEV *S) {
  SmallVector<const SCEV *, 4> Terms = MulExpander::expand(S, SE);
  ExecutionDomainRewriter Rewriter(*this);
  SmallVector<SCEVUse, 4> NewTerms;
  for (const SCEV *Term : Terms)
    NewTerms.push_back(Rewriter.visit(Term));
  return SE.getAddExpr(NewTerms);
}

void ExecutionDomain::run(Function &F, const LoopInfo &LI,
                          const DominatorTree &DT) {
  LoopNestInspector LNI(DT, LI, SE);

  for (const Loop *L : LI)
    traverseLoop(L, *this);

  auto Inequalities = collectConstraints(F, SE, LI);

  for (auto &[Ptr, DerefBytes, Complexity, Outermost] : Inequalities) {
    const SCEV *Max = estimateMaxOffsetValue(Ptr, Outermost, *this, LNI);
    if (Max) {
      InequalityType Inequality(ICmpInst::ICMP_SLE, Max, DerefBytes);
      addInequality(Inequality);
    }
  }
}

bool ExecutionDomain::isAddRecNoSignedWrap(const SCEVAddRecExpr *AR) {
  if (AR->hasNoSignedWrap())
    return true;
  if (const SCEVAddRecExpr *Rewritten = dyn_cast<SCEVAddRecExpr>(rewrite(AR)))
    if (Rewritten->hasNoSignedWrap())
      return true;

  if (!AR->isAffine())
    return false;
  if (!SE.hasLoopInvariantBackedgeTakenCount(AR->getLoop()))
    return false;

  const SCEV *Start = rewrite(AR->getStart());
  const SCEV *Step = rewrite(AR->getStepRecurrence(SE));
  const SCEV *BTC = rewrite(SE.getBackedgeTakenCount(AR->getLoop()));
  ConstantRange StartRange = SE.getSignedRange(Start);
  ConstantRange StepRange = SE.getSignedRange(Step);
  ConstantRange BTCRange = SE.getSignedRange(BTC);
  ConstantRange Mul = StepRange.smul_fast(BTCRange);
  if (Mul.isFullSet() || Mul.isSignWrappedSet())
    return false;
  return Mul.signedAddMayOverflow(StartRange) == ConstantRange::OverflowResult::NeverOverflows;
}

bool ExecutionDomain::isKnownAddNoSignedWrap(const SCEV *LHS, const SCEV *RHS) {
  ConstantRange LHSRange = SE.getSignedRange(rewrite(LHS));
  ConstantRange RHSRange = SE.getSignedRange(rewrite(RHS));
  if (LHSRange.isFullSet() || LHSRange.isSignWrappedSet() ||
      RHSRange.isFullSet() || RHSRange.isSignWrappedSet())
    return false;
  if (LHSRange.signedAddMayOverflow(RHSRange) ==
      ConstantRange::OverflowResult::NeverOverflows)
    return true;
  return SE.willNotOverflow(Instruction::Add, /*Signed=*/true, rewrite(LHS),
                            rewrite(RHS));
}

bool ExecutionDomain::isKnownSubNoSignedWrap(const SCEV *LHS, const SCEV *RHS) {
  ConstantRange LHSRange = SE.getSignedRange(rewrite(LHS));
  ConstantRange RHSRange = SE.getSignedRange(rewrite(RHS));
  if (LHSRange.isFullSet() || LHSRange.isSignWrappedSet() ||
      RHSRange.isFullSet() || RHSRange.isSignWrappedSet())
    return false;
  if (LHSRange.signedSubMayOverflow(RHSRange) ==
      ConstantRange::OverflowResult::NeverOverflows)
    return true;
  return SE.willNotOverflow(Instruction::Sub, /*Signed=*/true, rewrite(LHS),
                            rewrite(RHS));
}

bool ExecutionDomain::isKnownMulNoSignedWrap(const SCEV *LHS, const SCEV *RHS) {
  ConstantRange LHSRange = SE.getSignedRange(rewrite(LHS));
  ConstantRange RHSRange = SE.getSignedRange(rewrite(RHS));
  if (LHSRange.isFullSet() || LHSRange.isSignWrappedSet() ||
      RHSRange.isFullSet() || RHSRange.isSignWrappedSet())
    return false;
  ConstantRange Mul = LHSRange.smul_fast(RHSRange);
  if (!Mul.isFullSet() && !Mul.isSignWrappedSet())
    return true;
  return SE.willNotOverflow(Instruction::Mul, /*Signed=*/true, rewrite(LHS),
                            rewrite(RHS));
}

bool ExecutionDomain::isKnownPositive(const SCEV *S) {
  return SE.isKnownPositive(rewrite(S));
}

bool ExecutionDomain::isKnownNonNegative(const SCEV *S) {
  return SE.isKnownNonNegative(rewrite(S));
}

bool ExecutionDomain::isKnownNonPositive(const SCEV *S) {
  return SE.isKnownNonPositive(rewrite(S));
}

bool ExecutionDomain::isKnownPredicate(ICmpInst::Predicate Pred, const SCEV *LHS, const SCEV *RHS) {
  if (isa<SCEVConstant>(LHS)) {
    std::swap(LHS, RHS);
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }
  if (isa<SCEVConstant>(RHS)) {
    InequalityType Inequality(Pred, LHS, cast<SCEVConstant>(RHS)->getAPInt());
    if (std::optional<InequalityType> SEI = tryIntoSE(Inequality)) {
      Inequality = InequalitySimpliler::simplify(*SEI, *this);
      LHS = Inequality.LHS;
      RHS = SE.getConstant(Inequality.RHS);
      Pred = Inequality.Pred;
    }
  }

  return SE.isKnownPredicate(Pred, rewrite(LHS), rewrite(RHS));
}

static void printLoopInspect(raw_ostream &OS, Function &F, ScalarEvolution &SE,
                             const LoopInfo &LI, const DominatorTree &DT) {
  LoopNestInspector LNI(DT, LI, SE);
  auto Inequalities = collectConstraints(F, SE, LI);

  OS << "Is safe to estimate max offset value for each pointer...\n";
  for (auto &[Ptr, DerefBytes, Complexity, Outermost] : Inequalities) {
    bool Safe = LNI.isSafeToEstimateMaxOffsetValue(Ptr, Outermost);
    OS.indent(2) << (Safe ? "Safe" : "Unsafe") << "\n";
    OS.indent(4) << *Ptr << "\n";
  }
}

static void printInequalitySimplification(raw_ostream &OS, Function &F,
                                          ScalarEvolution &SE,
                                          const LoopInfo &LI,
                                          const DominatorTree &DT) {
  ExecutionDomain ED(SE);
  for (Instruction &Inst : instructions(F)) {
    ICmpInst *ICmp = dyn_cast<ICmpInst>(&Inst);
    if (!ICmp)
      continue;
    const SCEV *LHS = SE.getSCEV(ICmp->getOperand(0));
    const SCEV *RHS = SE.getSCEV(ICmp->getOperand(1));
    ICmpInst::Predicate Pred = ICmp->getPredicate();
    if (!isa<SCEVConstant>(RHS)) {
      Pred = ICmpInst::getSwappedPredicate(Pred);
      std::swap(LHS, RHS);
    }
    if (!isa<SCEVConstant>(RHS))
      continue;

    InequalityType Original(Pred, LHS, cast<SCEVConstant>(RHS)->getAPInt());
    std::optional<InequalityType> Simplified =
        InequalitySimpliler::simplify(Original, ED);
    OS << "Original:\n";
    OS.indent(2) << Original << "\n";
    OS << "Simplified:\n";
    OS.indent(2);
    if (Simplified)
      OS << "Simplified: " << *Simplified << "\n";
    else
      OS << "Failed to simplify.\n";
  }
}

static void printExecutionDomain(raw_ostream &OS, Function &F,
                                 ScalarEvolution &SE, const LoopInfo &LI,
                                 const DominatorTree &DT) {
  ExecutionDomain ED(SE);
  ED.run(F, LI, DT);
  ED.print(OS);
}

static void printEntry(raw_ostream &OS, Function &F, ScalarEvolution &SE,
                       LoopInfo &LI, const DominatorTree &DT) {
  OS << "Printing analysis 'Execution Domain' for function '" << F.getName()
     << "':\n";

  switch (LitTestType) {
  case TestType::ExecutionDomain:
    printExecutionDomain(OS, F, SE, LI, DT);
    break;
  case TestType::LoopInspect:
    printLoopInspect(OS, F, SE, LI, DT);
    break;
  case TestType::InequalitySimplification:
    printInequalitySimplification(OS, F, SE, LI, DT);
    break;
  }
}

ExecutionDomainPrinterPass::ExecutionDomainPrinterPass(raw_ostream &OS)
    : OS(OS) {}

PreservedAnalyses ExecutionDomainPrinterPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  printEntry(OS, F, SE, LI, DT);
  return PreservedAnalyses::all();
}
