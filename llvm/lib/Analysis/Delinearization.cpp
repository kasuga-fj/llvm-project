//===---- Delinearization.cpp - MultiDimensional Index Delinearization ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements an analysis pass that tries to delinearize all GEP
// instructions in all loops using the SCEV analysis functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Delinearization.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionDivision.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DL_NAME "delinearize"
#define DEBUG_TYPE DL_NAME

static cl::opt<bool> UseFixedSizeArrayHeuristic(
    "delinearize-use-fixed-size-array-heuristic", cl::init(true), cl::Hidden,
    cl::desc("When printing analysis, use the heuristic for fixed-size arrays "
             "if the default delinearizetion fails."));

// Return true when S contains at least an undef value.
static inline bool containsUndefs(const SCEV *S) {
  return SCEVExprContains(S, [](const SCEV *S) {
    if (const auto *SU = dyn_cast<SCEVUnknown>(S))
      return isa<UndefValue>(SU->getValue());
    return false;
  });
}

namespace {

// Collect all steps of SCEV expressions.
struct SCEVCollectStrides {
  ScalarEvolution &SE;
  SmallVectorImpl<const SCEV *> &Strides;

  SCEVCollectStrides(ScalarEvolution &SE, SmallVectorImpl<const SCEV *> &S)
      : SE(SE), Strides(S) {}

  bool follow(const SCEV *S) {
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
      Strides.push_back(AR->getStepRecurrence(SE));
    return true;
  }

  bool isDone() const { return false; }
};

// Collect all SCEVUnknown and SCEVMulExpr expressions.
struct SCEVCollectTerms {
  SmallVectorImpl<const SCEV *> &Terms;

  SCEVCollectTerms(SmallVectorImpl<const SCEV *> &T) : Terms(T) {}

  bool follow(const SCEV *S) {
    if (isa<SCEVUnknown>(S) || isa<SCEVMulExpr>(S) ||
        isa<SCEVSignExtendExpr>(S)) {
      if (!containsUndefs(S))
        Terms.push_back(S);

      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

// Check if a SCEV contains an AddRecExpr.
struct SCEVHasAddRec {
  bool &ContainsAddRec;

  SCEVHasAddRec(bool &ContainsAddRec) : ContainsAddRec(ContainsAddRec) {
    ContainsAddRec = false;
  }

  bool follow(const SCEV *S) {
    if (isa<SCEVAddRecExpr>(S)) {
      ContainsAddRec = true;

      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

// Find factors that are multiplied with an expression that (possibly as a
// subexpression) contains an AddRecExpr. In the expression:
//
//  8 * (100 +  %p * %q * (%a + {0, +, 1}_loop))
//
// "%p * %q" are factors multiplied by the expression "(%a + {0, +, 1}_loop)"
// that contains the AddRec {0, +, 1}_loop. %p * %q are likely to be array size
// parameters as they form a product with an induction variable.
//
// This collector expects all array size parameters to be in the same MulExpr.
// It might be necessary to later add support for collecting parameters that are
// spread over different nested MulExpr.
struct SCEVCollectAddRecMultiplies {
  SmallVectorImpl<const SCEV *> &Terms;
  ScalarEvolution &SE;

  SCEVCollectAddRecMultiplies(SmallVectorImpl<const SCEV *> &T,
                              ScalarEvolution &SE)
      : Terms(T), SE(SE) {}

  bool follow(const SCEV *S) {
    if (auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
      bool HasAddRec = false;
      SmallVector<SCEVUse, 0> Operands;
      for (const SCEV *Op : Mul->operands()) {
        const SCEVUnknown *Unknown = dyn_cast<SCEVUnknown>(Op);
        if (Unknown && !isa<CallInst>(Unknown->getValue())) {
          Operands.push_back(Op);
        } else if (Unknown) {
          HasAddRec = true;
        } else {
          bool ContainsAddRec = false;
          SCEVHasAddRec ContiansAddRec(ContainsAddRec);
          visitAll(Op, ContiansAddRec);
          HasAddRec |= ContainsAddRec;
        }
      }
      if (Operands.size() == 0)
        return true;

      if (!HasAddRec)
        return false;

      Terms.push_back(SE.getMulExpr(Operands));
      // Stop recursion: once we collected a term, do not walk its operands.
      return false;
    }

    // Keep looking.
    return true;
  }

  bool isDone() const { return false; }
};

} // end anonymous namespace

/// Find parametric terms in this SCEVAddRecExpr. We first for parameters in
/// two places:
///   1) The strides of AddRec expressions.
///   2) Unknowns that are multiplied with AddRec expressions.
void llvm::collectParametricTerms(ScalarEvolution &SE, const SCEV *Expr,
                                  SmallVectorImpl<const SCEV *> &Terms) {
  SmallVector<const SCEV *, 4> Strides;
  SCEVCollectStrides StrideCollector(SE, Strides);
  visitAll(Expr, StrideCollector);

  LLVM_DEBUG({
    dbgs() << "Strides:\n";
    for (const SCEV *S : Strides)
      dbgs().indent(2) << *S << "\n";
  });

  for (const SCEV *S : Strides) {
    SCEVCollectTerms TermCollector(Terms);
    visitAll(S, TermCollector);
  }

  LLVM_DEBUG({
    dbgs() << "Terms:\n";
    for (const SCEV *T : Terms)
      dbgs().indent(2) << *T << "\n";
  });

  SCEVCollectAddRecMultiplies MulCollector(Terms, SE);
  visitAll(Expr, MulCollector);
}

static bool findArrayDimensionsRec(ScalarEvolution &SE,
                                   SmallVectorImpl<const SCEV *> &Terms,
                                   SmallVectorImpl<const SCEV *> &Sizes) {
  int Last = Terms.size() - 1;
  const SCEV *Step = Terms[Last];

  // End of recursion.
  if (Last == 0) {
    if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(Step)) {
      SmallVector<SCEVUse, 2> Qs;
      for (const SCEV *Op : M->operands())
        if (!isa<SCEVConstant>(Op))
          Qs.push_back(Op);

      Step = SE.getMulExpr(Qs);
    }

    Sizes.push_back(Step);
    return true;
  }

  for (const SCEV *&Term : Terms) {
    // Normalize the terms before the next call to findArrayDimensionsRec.
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, Step, &Q, &R);

    // Bail out when GCD does not evenly divide one of the terms.
    if (!R->isZero())
      return false;

    Term = Q;
  }

  // Remove all SCEVConstants.
  erase_if(Terms, [](const SCEV *E) { return isa<SCEVConstant>(E); });

  if (Terms.size() > 0)
    if (!findArrayDimensionsRec(SE, Terms, Sizes))
      return false;

  Sizes.push_back(Step);
  return true;
}

// Returns true when one of the SCEVs of Terms contains a SCEVUnknown parameter.
static inline bool containsParameters(SmallVectorImpl<const SCEV *> &Terms) {
  for (const SCEV *T : Terms)
    if (SCEVExprContains(T, [](const SCEV *S) { return isa<SCEVUnknown>(S); }))
      return true;

  return false;
}

// Return the number of product terms in S.
static inline int numberOfTerms(const SCEV *S) {
  if (const SCEVMulExpr *Expr = dyn_cast<SCEVMulExpr>(S))
    return Expr->getNumOperands();
  return 1;
}

static const SCEV *removeConstantFactors(ScalarEvolution &SE, const SCEV *T) {
  if (isa<SCEVConstant>(T))
    return nullptr;

  if (isa<SCEVUnknown>(T))
    return T;

  if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(T)) {
    SmallVector<SCEVUse, 2> Factors;
    for (const SCEV *Op : M->operands())
      if (!isa<SCEVConstant>(Op))
        Factors.push_back(Op);

    return SE.getMulExpr(Factors);
  }

  return T;
}

void llvm::findArrayDimensions(ScalarEvolution &SE,
                               SmallVectorImpl<const SCEV *> &Terms,
                               SmallVectorImpl<const SCEV *> &Sizes,
                               const SCEV *ElementSize) {
  if (Terms.size() < 1 || !ElementSize)
    return;

  // Early return when Terms do not contain parameters: we do not delinearize
  // non parametric SCEVs.
  if (!containsParameters(Terms))
    return;

  LLVM_DEBUG({
    dbgs() << "Terms:\n";
    for (const SCEV *T : Terms)
      dbgs().indent(2) << *T << "\n";
  });

  // Remove duplicates.
  array_pod_sort(Terms.begin(), Terms.end());
  Terms.erase(llvm::unique(Terms), Terms.end());

  // Put larger terms first.
  llvm::sort(Terms, [](const SCEV *LHS, const SCEV *RHS) {
    return numberOfTerms(LHS) > numberOfTerms(RHS);
  });

  // Try to divide all terms by the element size. If term is not divisible by
  // element size, proceed with the original term.
  for (const SCEV *&Term : Terms) {
    const SCEV *Q, *R;
    SCEVDivision::divide(SE, Term, ElementSize, &Q, &R);
    if (!Q->isZero())
      Term = Q;
  }

  SmallVector<const SCEV *, 4> NewTerms;

  // Remove constant factors.
  for (const SCEV *T : Terms)
    if (const SCEV *NewT = removeConstantFactors(SE, T))
      NewTerms.push_back(NewT);

  LLVM_DEBUG({
    dbgs() << "Terms after sorting:\n";
    for (const SCEV *T : NewTerms)
      dbgs().indent(2) << *T << "\n";
  });

  if (NewTerms.empty() || !findArrayDimensionsRec(SE, NewTerms, Sizes)) {
    Sizes.clear();
    return;
  }

  // The last element to be pushed into Sizes is the size of an element.
  Sizes.push_back(ElementSize);

  LLVM_DEBUG({
    dbgs() << "Sizes:\n";
    for (const SCEV *S : Sizes)
      dbgs().indent(2) << *S << "\n";
  });
}

void llvm::computeAccessFunctions(ScalarEvolution &SE, const SCEV *Expr,
                                  SmallVectorImpl<const SCEV *> &Subscripts,
                                  SmallVectorImpl<const SCEV *> &Sizes) {
  // Early exit in case this SCEV is not an affine multivariate function.
  if (Sizes.empty())
    return;

  if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr))
    if (!AR->isAffine())
      return;

  // Clear output vector.
  Subscripts.clear();

  LLVM_DEBUG(dbgs() << "\ncomputeAccessFunctions\n"
                    << "Memory Access Function: " << *Expr << "\n");

  const SCEV *Res = Expr;
  int Last = Sizes.size() - 1;

  for (int i = Last; i >= 0; i--) {
    const SCEV *Size = Sizes[i];
    const SCEV *Q, *R;

    SCEVDivision::divide(SE, Res, Size, &Q, &R);

    LLVM_DEBUG({
      dbgs() << "Computing 'MemAccFn / Sizes[" << i << "]':\n";
      dbgs() << "  MemAccFn: " << *Res << "\n";
      dbgs() << "  Sizes[" << i << "]: " << *Size << "\n";
      dbgs() << "  Quotient (Leftover): " << *Q << "\n";
      dbgs() << "  Remainder (Subscript Access Function): " << *R << "\n";
    });

    Res = Q;

    // Do not record the last subscript corresponding to the size of elements in
    // the array.
    if (i == Last) {

      // Bail out if the byte offset is non-zero.
      if (!R->isZero()) {
        Subscripts.clear();
        Sizes.clear();
        return;
      }

      continue;
    }

    // Record the access function for the current subscript.
    Subscripts.push_back(R);
  }

  // Also push in last position the remainder of the last division: it will be
  // the access function of the innermost dimension.
  Subscripts.push_back(Res);

  std::reverse(Subscripts.begin(), Subscripts.end());

  LLVM_DEBUG({
    dbgs() << "Subscripts:\n";
    for (const SCEV *S : Subscripts)
      dbgs().indent(2) << *S << "\n";
    dbgs() << "\n";
  });
}

/// Splits the SCEV into two vectors of SCEVs representing the subscripts and
/// sizes of an array access. Returns the remainder of the delinearization that
/// is the offset start of the array.  The SCEV->delinearize algorithm computes
/// the multiples of SCEV coefficients: that is a pattern matching of sub
/// expressions in the stride and base of a SCEV corresponding to the
/// computation of a GCD (greatest common divisor) of base and stride.  When
/// SCEV->delinearize fails, it returns the SCEV unchanged.
///
/// For example: when analyzing the memory access A[i][j][k] in this loop nest
///
///  void foo(long n, long m, long o, double A[n][m][o]) {
///
///    for (long i = 0; i < n; i++)
///      for (long j = 0; j < m; j++)
///        for (long k = 0; k < o; k++)
///          A[i][j][k] = 1.0;
///  }
///
/// the delinearization input is the following AddRec SCEV:
///
///  AddRec: {{{%A,+,(8 * %m * %o)}<%for.i>,+,(8 * %o)}<%for.j>,+,8}<%for.k>
///
/// From this SCEV, we are able to say that the base offset of the access is %A
/// because it appears as an offset that does not divide any of the strides in
/// the loops:
///
///  CHECK: Base offset: %A
///
/// and then SCEV->delinearize determines the size of some of the dimensions of
/// the array as these are the multiples by which the strides are happening:
///
///  CHECK: ArrayDecl[UnknownSize][%m][%o] with elements of sizeof(double)
///  bytes.
///
/// Note that the outermost dimension remains of UnknownSize because there are
/// no strides that would help identifying the size of the last dimension: when
/// the array has been statically allocated, one could compute the size of that
/// dimension by dividing the overall size of the array by the size of the known
/// dimensions: %m * %o * 8.
///
/// Finally delinearize provides the access functions for the array reference
/// that does correspond to A[i][j][k] of the above C testcase:
///
///  CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]
///
/// The testcases are checking the output of a function pass:
/// DelinearizationPass that walks through all loads and stores of a function
/// asking for the SCEV of the memory access with respect to all enclosing
/// loops, calling SCEV->delinearize on that and printing the results.
void llvm::delinearize(ScalarEvolution &SE, const SCEV *Expr,
                       SmallVectorImpl<const SCEV *> &Subscripts,
                       SmallVectorImpl<const SCEV *> &Sizes,
                       const SCEV *ElementSize) {
  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  // First step: collect parametric terms.
  SmallVector<const SCEV *, 4> Terms;
  collectParametricTerms(SE, Expr, Terms);

  if (Terms.empty())
    return;

  // Second step: find subscript sizes.
  findArrayDimensions(SE, Terms, Sizes, ElementSize);

  if (Sizes.empty())
    return;

  // Third step: compute the access functions for each subscript.
  computeAccessFunctions(SE, Expr, Subscripts, Sizes);
}

static std::optional<APInt> tryIntoAPInt(const SCEV *S) {
  if (const auto *Const = dyn_cast<SCEVConstant>(S))
    return Const->getAPInt();
  return std::nullopt;
}

/// Collects the absolute values of constant steps for all induction variables.
/// Returns true if we can prove that all step recurrences are constants and \p
/// Expr is divisible by \p ElementSize. Each step recurrence is stored in \p
/// Steps after divided by \p ElementSize.
static bool collectConstantAbsSteps(ScalarEvolution &SE, const SCEV *Expr,
                                    SmallVectorImpl<uint64_t> &Steps,
                                    uint64_t ElementSize) {
  // End of recursion. The constant value also must be a multiple of
  // ElementSize.
  if (const auto *Const = dyn_cast<SCEVConstant>(Expr)) {
    const uint64_t Mod = Const->getAPInt().urem(ElementSize);
    return Mod == 0;
  }

  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Expr);
  if (!AR || !AR->isAffine())
    return false;

  const SCEV *Step = AR->getStepRecurrence(SE);
  std::optional<APInt> StepAPInt = tryIntoAPInt(Step);
  if (!StepAPInt)
    return false;

  APInt Q;
  uint64_t R;
  APInt::udivrem(StepAPInt->abs(), ElementSize, Q, R);
  if (R != 0)
    return false;

  // Bail out when the step is too large.
  std::optional<uint64_t> StepVal = Q.tryZExtValue();
  if (!StepVal)
    return false;

  Steps.push_back(*StepVal);
  return collectConstantAbsSteps(SE, AR->getStart(), Steps, ElementSize);
}

bool llvm::findFixedSizeArrayDimensions(ScalarEvolution &SE, const SCEV *Expr,
                                        SmallVectorImpl<uint64_t> &Sizes,
                                        const SCEV *ElementSize) {
  if (!ElementSize)
    return false;

  std::optional<APInt> ElementSizeAPInt = tryIntoAPInt(ElementSize);
  if (!ElementSizeAPInt || *ElementSizeAPInt == 0)
    return false;

  std::optional<uint64_t> ElementSizeConst = ElementSizeAPInt->tryZExtValue();

  // Early exit when ElementSize is not a positive constant.
  if (!ElementSizeConst)
    return false;

  if (!collectConstantAbsSteps(SE, Expr, Sizes, *ElementSizeConst) ||
      Sizes.empty()) {
    Sizes.clear();
    return false;
  }

  // At this point, Sizes contains the absolute step recurrences for all
  // induction variables. Each step recurrence must be a multiple of the size of
  // the array element. Assuming that the each value represents the size of an
  // array for each dimension, attempts to restore the length of each dimension
  // by dividing the step recurrence by the next smaller value. For example, if
  // we have the following AddRec SCEV:
  //
  //   AddRec: {{{0,+,2048}<%for.i>,+,256}<%for.j>,+,8}<%for.k> (ElementSize=8)
  //
  // Then Sizes will become [256, 32, 1] after sorted. We don't know the size of
  // the outermost dimension, the next dimension will be computed as 256 / 32 =
  // 8, and the last dimension will be computed as 32 / 1 = 32. Thus it results
  // in like Arr[UnknownSize][8][32] with elements of size 8 bytes, where Arr is
  // a base pointer.
  //
  // TODO: Catch more cases, e.g., when a step recurrence is not divisible by
  // the next smaller one, like A[i][3*j].
  llvm::sort(Sizes.rbegin(), Sizes.rend());
  Sizes.erase(llvm::unique(Sizes), Sizes.end());

  // The last element in Sizes should be ElementSize. At this point, all values
  // in Sizes are assumed to be divided by ElementSize, so replace it with 1.
  assert(Sizes.back() != 0 && "Unexpected zero size in Sizes.");
  Sizes.back() = 1;

  for (unsigned I = 0; I + 1 < Sizes.size(); I++) {
    uint64_t PrevSize = Sizes[I + 1];
    if (Sizes[I] % PrevSize) {
      Sizes.clear();
      return false;
    }
    Sizes[I] /= PrevSize;
  }

  // Finally, the last element in Sizes should be ElementSize.
  Sizes.back() = *ElementSizeConst;
  return true;
}

/// Splits the SCEV into two vectors of SCEVs representing the subscripts and
/// sizes of an array access, assuming that the array is a fixed size array.
///
/// E.g., if we have the code like as follows:
///
///  double A[42][8][32];
///  for i
///    for j
///      for k
///        use A[i][j][k]
///
/// The access function will be represented as an AddRec SCEV like:
///
///  AddRec: {{{0,+,2048}<%for.i>,+,256}<%for.j>,+,8}<%for.k> (ElementSize=8)
///
/// Then findFixedSizeArrayDimensions infers the size of each dimension of the
/// array based on the fact that the value of the step recurrence is a multiple
/// of the size of the corresponding array element. In the above example, it
/// results in the following:
///
///  CHECK: ArrayDecl[UnknownSize][8][32] with elements of 8 bytes.
///
/// Finally each subscript will be computed as follows:
///
///  CHECK: ArrayRef[{0,+,1}<%for.i>][{0,+,1}<%for.j>][{0,+,1}<%for.k>]
///
/// Note that this function doesn't check the range of possible values for each
/// subscript, so the caller should perform additional boundary checks if
/// necessary.
///
/// Also note that this function doesn't guarantee that the original array size
/// is restored "correctly". For example, in the following case:
///
///  double A[42][4][64];
///  double B[42][8][32];
///  for i
///    for j
///      for k
///        use A[i][j][k]
///        use B[i][2*j][k]
///
/// The access function for both accesses will be the same:
///
///  AddRec: {{{0,+,2048}<%for.i>,+,512}<%for.j>,+,8}<%for.k> (ElementSize=8)
///
/// The array sizes for both A and B will be computed as
/// ArrayDecl[UnknownSize][4][64], which matches for A, but not for B.
///
/// TODO: At the moment, this function can handle only simple cases. For
/// example, we cannot handle a case where a step recurrence is not divisible
/// by the next smaller step recurrence, e.g., A[i][3*j].
bool llvm::delinearizeFixedSizeArray(ScalarEvolution &SE, const SCEV *Expr,
                                     SmallVectorImpl<const SCEV *> &Subscripts,
                                     SmallVectorImpl<const SCEV *> &Sizes,
                                     const SCEV *ElementSize) {
  // Clear output vectors.
  Subscripts.clear();
  Sizes.clear();

  // First step: find the fixed array size.
  SmallVector<uint64_t, 4> ConstSizes;
  if (!findFixedSizeArrayDimensions(SE, Expr, ConstSizes, ElementSize)) {
    Sizes.clear();
    return false;
  }

  // Convert the constant size to SCEV.
  for (uint64_t Size : ConstSizes)
    Sizes.push_back(SE.getConstant(Expr->getType(), Size));

  // Second step: compute the access functions for each subscript.
  computeAccessFunctions(SE, Expr, Subscripts, Sizes);

  return !Subscripts.empty();
}

bool llvm::validateDelinearizationResult(ScalarEvolution &SE,
                                         ArrayRef<const SCEV *> Sizes,
                                         ArrayRef<const SCEV *> Subscripts) {
  // Sizes and Subscripts are as follows:
  //
  //   Sizes:      [UNK][S_2]...[S_n]
  //   Subscripts: [I_1][I_2]...[I_n]
  //
  // where the size of the outermost dimension is unknown (UNK).

  auto AddOverflow = [&](const SCEV *A, const SCEV *B) -> const SCEV * {
    if (!SE.willNotOverflow(Instruction::Add, /*IsSigned=*/true, A, B))
      return nullptr;
    return SE.getAddExpr(A, B);
  };

  auto MulOverflow = [&](const SCEV *A, const SCEV *B) -> const SCEV * {
    if (!SE.willNotOverflow(Instruction::Mul, /*IsSigned=*/true, A, B))
      return nullptr;
    return SE.getMulExpr(A, B);
  };

  // Range check: 0 <= I_k < S_k for k = 2..n.
  for (size_t I = 1; I < Sizes.size(); ++I) {
    const SCEV *Size = Sizes[I - 1];
    const SCEV *Subscript = Subscripts[I];
    if (!SE.isKnownNonNegative(Subscript))
      return false;

    // TODO: It may be better that delinearization itself unifies the types of
    // all elements in Sizes and Subscripts.
    Type *WiderTy = SE.getWiderType(Subscript->getType(), Size->getType());
    Subscript = SE.getNoopOrSignExtend(Subscript, WiderTy);
    Size = SE.getNoopOrSignExtend(Size, WiderTy);
    if (!SE.isKnownPredicate(ICmpInst::ICMP_SLT, Subscript, Size)) {
      LLVM_DEBUG(dbgs() << "Range check failed: " << *Subscript << " <s "
                        << *Size << "\n");
      return false;
    }
  }

  // The offset computation is as follows:
  //
  //   Offset = I_n +
  //            S_n * I_{n-1} +
  //            ... +
  //            (S_2 * ... * S_n) * I_1
  //
  // Regarding this as a function from (I_1, I_2, ..., I_n) to integers, it
  // must be injective. To guarantee it, the above calculation must not
  // overflow. Since we have already checked that 0 <= I_k < S_k for k = 2..n,
  // the minimum and maximum values occur in the following cases:
  //
  //   Min = [I_1][0]...[0] = S_2 * ... * S_n * I_1
  //   Max = [I_1][S_2-1]...[S_n-1]
  //       = (S_2 * ... * S_n) * I_1 +
  //         (S_2 * ... * S_{n-1}) * (S_2 - 1) +
  //         ... +
  //         (S_n - 1)
  //       = (S_2 * ... * S_n) * I_1 +
  //         (S_2 * ... * S_n) - 1  (can be proven by induction)
  //       = Min + (S_2 * ... * S_n) - 1
  //
  // NOTE: I_1 can be negative, so Min is not just 0.
  const SCEV *Prod = SE.getOne(Sizes[0]->getType());
  for (const SCEV *Size : Sizes) {
    Prod = MulOverflow(Prod, Size);
    if (!Prod)
      return false;
  }
  const SCEV *Min = MulOverflow(Prod, Subscripts[0]);
  if (!Min)
    return false;

  // We have already checked that Min and Prod don't overflow, so it's enough
  // to check whether Min + Prod - 1 doesn't overflow.
  const SCEV *MaxPlusOne = AddOverflow(Min, Prod);
  if (!MaxPlusOne)
    return false;
  if (!SE.willNotOverflow(Instruction::Sub, /*IsSigned=*/true, MaxPlusOne,
                          SE.getOne(MaxPlusOne->getType())))
    return false;

  return true;
}

bool llvm::getIndexExpressionsFromGEP(ScalarEvolution &SE,
                                      const GetElementPtrInst *GEP,
                                      SmallVectorImpl<const SCEV *> &Subscripts,
                                      SmallVectorImpl<const SCEV *> &Sizes) {
  assert(Subscripts.empty() && Sizes.empty() &&
         "Expected output lists to be empty on entry to this function.");
  assert(GEP && "getIndexExpressionsFromGEP called with a null GEP");
  LLVM_DEBUG(dbgs() << "\nGEP to delinearize: " << *GEP << "\n");
  Type *Ty = nullptr;
  Type *IndexTy = SE.getEffectiveSCEVType(GEP->getPointerOperandType());
  bool DroppedFirstDim = false;
  for (unsigned i = 1; i < GEP->getNumOperands(); i++) {
    const SCEV *Expr = SE.getSCEV(GEP->getOperand(i));
    if (i == 1) {
      Ty = GEP->getSourceElementType();
      if (auto *Const = dyn_cast<SCEVConstant>(Expr))
        if (Const->getValue()->isZero()) {
          DroppedFirstDim = true;
          continue;
        }
      Subscripts.push_back(Expr);
      continue;
    }

    auto *ArrayTy = dyn_cast<ArrayType>(Ty);
    if (!ArrayTy) {
      LLVM_DEBUG(dbgs() << "GEP delinearize failed: " << *Ty
                        << " is not an array type.\n");
      Subscripts.clear();
      Sizes.clear();
      return false;
    }

    Subscripts.push_back(Expr);
    if (!(DroppedFirstDim && i == 2))
      Sizes.push_back(SE.getConstant(IndexTy, ArrayTy->getNumElements()));

    Ty = ArrayTy->getElementType();
  }
  LLVM_DEBUG({
    dbgs() << "Subscripts:\n";
    for (const SCEV *S : Subscripts)
      dbgs() << *S << "\n";
    dbgs() << "\n";
  });

  return !Subscripts.empty();
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

struct ExecutionDomain {
  ExecutionDomain(ScalarEvolution &SE) : SE(SE) {}

  void smin(const SCEV *S, const APInt &RHS) {
    auto Ite = fetch(S);
    Ite->second = Ite->second.intersectWith(
        ConstantRange::makeExactICmpRegion(ICmpInst::ICMP_SLT, RHS));
  }

  void smax(const SCEV *S, const APInt &RHS) {
    auto Ite = fetch(S);
    Ite->second = Ite->second.intersectWith(
        ConstantRange::makeExactICmpRegion(ICmpInst::ICMP_SGE, RHS));
  }

  void clamp(const APInt &Min, const SCEV *S, const APInt &Max) {
    auto Ite = fetch(S);
    Ite->second = Ite->second.intersectWith(ConstantRange(Min, Max + 1));
  }

  bool isKnownNonNegative(const SCEV *S) {
    auto Ite = Ranges.find(S);
    if (Ite != Ranges.end() && Ite->second.getSignedMin().isNonNegative())
      return true;
    return SE.isKnownNonNegative(S);
  }

  bool isKnownNonPositive(const SCEV *S) {
    auto Ite = Ranges.find(S);
    if (Ite != Ranges.end() && Ite->second.getSignedMin().isNonPositive())
      return true;
    return SE.isKnownNonPositive(S);
  }

private:
  using Cache = DenseMap<const SCEV *, ConstantRange>;
  ScalarEvolution &SE;
  DenseMap<const SCEV *, ConstantRange> Ranges;

  Cache::iterator fetch(const SCEV *S) {
    return Ranges.try_emplace(S, SE.getSignedRange(S)).first;
  }
};

const SCEV *findMax(const SCEV *S, ScalarEvolution &SE, ExecutionDomain &ED) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (!AR)
    return S;

  if (!AR->isAffine() /* || !AR->hasNoSignedWrap()*/)
    return nullptr;

  const SCEV *Rec = findMax(AR->getStart(), SE, ED);
  if (!Rec)
    return nullptr;

  const SCEV *Step = AR->getStepRecurrence(SE);
  if (ED.isKnownNonNegative(Step) || true) {
    const SCEV *BTC = SE.getBackedgeTakenCount(AR->getLoop());
    return dyn_cast<SCEVAddRecExpr>(
               SE.getAddRecExpr(Rec, Step, AR->getLoop(), AR->getNoWrapFlags()))
        ->evaluateAtIteration(BTC, SE);
  }
  if (ED.isKnownNonPositive(Step)) {
    return Rec;
  }
  return nullptr;
}

void collectTC(ScalarEvolution &SE, const Loop *L,
               DenseMap<const SCEV *, const SCEV *> &Map) {
  const SCEV *BTC = SE.getBackedgeTakenCount(L);
  IntegerType *IntTy = dyn_cast<IntegerType>(BTC->getType());
  const SCEV *TripCount = SE.getAddExpr(BTC, SE.getOne(IntTy));
  dbgs() << "TripCount for loop " << L->getHeader()->getName() << ": "
         << *TripCount << "\n";
  auto [Ite, Inserted] = Map.try_emplace(
      TripCount,
      SE.getSMaxExpr(SE.getZero(IntTy),
                     SE.getSMinExpr(TripCount, SE.getConstant(IntTy, 53))));
  if (Inserted) {
    dbgs() << "Adding TripCount rewrite: " << *TripCount << " -> "
           << *Ite->second << "\n";
  }

  for (const Loop *Child : L->getSubLoops())
    collectTC(SE, Child, Map);
}

struct MySCEVRewriter : public SCEVRewriteVisitor<MySCEVRewriter> {
  using SCEVRewriteVisitor::visit;

  MySCEVRewriter(ScalarEvolution &SE,
                 const DenseMap<const SCEV *, const SCEV *> &Map)
      : SCEVRewriteVisitor(SE) {
    for (const auto &KV : Map)
      RewriteResults[KV.first] = KV.second;
  }

  static MySCEVRewriter create(ScalarEvolution &SE, LoopInfo &LI) {
    DenseMap<const SCEV *, const SCEV *> Map;
    for (const Loop *L : LI.getTopLevelLoops())
      collectTC(SE, L, Map);
    return MySCEVRewriter(SE, Map);
  }
};

struct RelaxedStrides {
  SmallVector<std::pair<const SCEV *, const Loop *>, 4> Strides;
  const SCEV *Const;

  static std::optional<RelaxedStrides> create(ScalarEvolution &SE,
                                              const SCEV *Expr,
                                              const SCEV *EltSize,
                                              const Loop *OutermostLoop);

  bool validate(ScalarEvolution &SE, const SCEV *EltSize) const;

  void relaxedDelinarize(ScalarEvolution &SE,
                         SmallVectorImpl<const SCEV *> &Sizes,
                         SmallVectorImpl<const SCEV *> &Subscripts) const;

private:
  bool sort(ScalarEvolution &SE);
};

std::optional<const SCEV *> quotientIfDivisible(ScalarEvolution &SE,
                                                const SCEV *Numerator,
                                                const SCEV *Denominator) {
  const SCEV *Q, *R;
  SCEVDivision::divide(SE, Numerator, Denominator, &Q, &R);
  if (!R->isZero())
    return std::nullopt;
  return Q;
}

bool collectRelaxedStridesAux(ScalarEvolution &SE, const SCEV *Expr,
                              RelaxedStrides &RS, const Loop *OutermostLoop,
                              const SCEV *EltSize) {
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Expr);
  if (!AR || !AR->isAffine()) {
    if (!SE.isLoopInvariant(Expr, OutermostLoop))
      return false;

    std::optional<const SCEV *> Q = quotientIfDivisible(SE, Expr, EltSize);
    if (!Q)
      return false;

    RS.Const = *Q;
    return true;
  }

  const SCEV *Step = AR->getStepRecurrence(SE);
  std::optional<const SCEV *> Q = quotientIfDivisible(SE, Step, EltSize);
  if (!Q)
    return false;
  RS.Strides.push_back({*Q, AR->getLoop()});
  return collectRelaxedStridesAux(SE, AR->getStart(), RS, OutermostLoop,
                                  EltSize);
}

std::optional<RelaxedStrides>
RelaxedStrides::create(ScalarEvolution &SE, const SCEV *Expr,
                       const SCEV *EltSize, const Loop *OutermostLoop) {
  RelaxedStrides RS;
  if (!collectRelaxedStridesAux(SE, Expr, RS, OutermostLoop, EltSize))
    return std::nullopt;

  if (!RS.sort(SE))
    return std::nullopt;

  if (!RS.validate(SE, EltSize))
    return std::nullopt;

  // TODO: Handle Const.

  return RS;
}

bool RelaxedStrides::sort(ScalarEvolution &SE) {
  unsigned N = Strides.size();
  dbgs() << "RelaxedStrides::sort:\n";
  for (const auto &[Stride, Loop] : Strides) {
    dbgs() << "Stride: " << *Stride << " Loop: " << Loop->getHeader()->getName()
           << "\n";
  }
  for (unsigned UB = N; 0 < UB; UB--) {
    for (unsigned I = 0; I + 1 < UB; I++) {
      LLVM_DEBUG(dbgs() << "I: " << I << " J: " << (I + 1) << "\n");
      unsigned J = I + 1;
      const SCEV *SI = Strides[I].first;
      const SCEV *SJ = Strides[J].first;
      if (SE.isKnownPredicate(ICmpInst::ICMP_SLE, SI, SJ))
        continue;
      if (SE.isKnownPredicate(ICmpInst::ICMP_SGE, SI, SJ)) {
        std::swap(Strides[I], Strides[J]);
        continue;
      }

      LLVM_DEBUG(dbgs() << "Cannot determine order of strides: " << *SI
                        << " and " << *SJ << "\n");
      LLVM_DEBUG(SE.getSignedRange(SI).dump());
      LLVM_DEBUG(SE.getSignedRange(SJ).dump());
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "sorted\n");

  for (unsigned I = 0; I < N; I++) {
    const SCEV *SI = Strides[I].first;
    for (unsigned J = I + 1; J < N; J++) {
      const SCEV *SJ = Strides[J].first;
      if (!SE.isKnownPredicate(ICmpInst::ICMP_SLE, SI, SJ))
        return false;
    }
  }

  LLVM_DEBUG(dbgs() << "verified sorted strides\n");

  return true;
}

void RelaxedStrides::relaxedDelinarize(
    ScalarEvolution &SE, SmallVectorImpl<const SCEV *> &Sizes,
    SmallVectorImpl<const SCEV *> &Subscripts) const {
  const SCEV *Zero = SE.getZero(Strides[0].first->getType());
  const SCEV *One = SE.getOne(Strides[0].first->getType());
  for (const auto &[Stride, Loop] : Strides) {
    Sizes.push_back(Stride);
    Subscripts.push_back(
        SE.getAddRecExpr(Zero, One, Loop, SCEV::NoWrapFlags::FlagAnyWrap));
  }

  std::reverse(Sizes.begin(), Sizes.end());
  std::reverse(Subscripts.begin(), Subscripts.end());
}

enum Monotonicity {
  Increasing,
  Decreasing,
  Constant,
  Unknown,
};

struct SCEVMonotonicityVisitor
    : public SCEVVisitor<SCEVMonotonicityVisitor, Monotonicity> {
  using Base = SCEVVisitor<SCEVMonotonicityVisitor, Monotonicity>;

  SCEVMonotonicityVisitor(const SCEV *Var) : Var(Var) {}

  static Monotonicity evaluate(const SCEV *S, const SCEV *Var) {
    return SCEVMonotonicityVisitor(Var).visit(S);
  }

  Monotonicity visit(const SCEV *S) {
    if (S == Var)
      return Increasing;
    return Base::visit(S);
  }

  Monotonicity visitConstant(const SCEVConstant *) { return Constant; }

  Monotonicity visitVScale(const SCEVVScale *) { return Constant; }

  Monotonicity visitAddExpr(const SCEVAddExpr *S) { return visitNaryExpr(S); }

  Monotonicity visitMulExpr(const SCEVMulExpr *S) { return visitNaryExpr(S); }

  Monotonicity visitSignExtendExpr(const SCEVSignExtendExpr *S) {
    return visit(S->getOperand());
  }

  Monotonicity visitSMinExpr(const SCEVSMinExpr *S) { return visitNaryExpr(S); }

  Monotonicity visitSMaxExpr(const SCEVSMaxExpr *S) { return visitNaryExpr(S); }

  Monotonicity visitPtrToAddrExpr(const SCEVPtrToAddrExpr *) { return Unknown; }
  Monotonicity visitPtrToIntExpr(const SCEVPtrToIntExpr *) { return Unknown; }
  Monotonicity visitTruncateExpr(const SCEVTruncateExpr *) { return Unknown; }
  Monotonicity visitZeroExtendExpr(const SCEVZeroExtendExpr *) {
    return Unknown;
  }
  Monotonicity visitUDivExpr(const SCEVUDivExpr *) { return Unknown; }
  Monotonicity visitAddRecExpr(const SCEVAddRecExpr *) { return Unknown; }
  Monotonicity visitUMaxExpr(const SCEVUMaxExpr *) { return Unknown; }
  Monotonicity visitUMinExpr(const SCEVUMinExpr *) { return Unknown; }
  Monotonicity visitUnknown(const SCEVUnknown *) { return Unknown; }
  Monotonicity visitSequentialUMinExpr(const SCEVSequentialUMinExpr *) {
    return Unknown;
  }

private:
  const SCEV *Var;

  Monotonicity visitNaryExpr(const SCEVNAryExpr *S) {
    if (!S->hasNoSignedWrap())
      return Unknown;

    Monotonicity Result = Constant;
    for (const SCEV *Op : S->operands()) {
      Monotonicity OpMono = visit(Op);
      if (OpMono == Unknown)
        return Unknown;
      if (Result == Constant) {
        Result = OpMono;
        continue;
      }
      if (OpMono == Constant)
        continue;
      if (Result != OpMono)
        return Unknown;
    }
    return Result;
  }
};

struct InequalitySimpliler : public SCEVVisitor<InequalitySimpliler, void> {
  InequalitySimpliler(CmpPredicate Pred, const SCEV *LHS, const APInt &RHS,
                      ScalarEvolution &SE)
      : Pred(Pred), LHS(LHS), RHS(RHS), SE(SE) {}

  static std::tuple<CmpPredicate, const SCEV *, OverflowSafeSignedAPInt>
  simplify(CmpPredicate Pred, const SCEV *LHS, const APInt &RHS,
           ScalarEvolution &SE) {
    InequalitySimpliler Simplifier(Pred, LHS, RHS, SE);
    Simplifier.visit(LHS);
    return {Simplifier.Pred, Simplifier.LHS, Simplifier.RHS};
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
  CmpPredicate Pred;
  const SCEV *LHS;
  OverflowSafeSignedAPInt RHS;
  ScalarEvolution &SE;
};

bool RelaxedStrides::validate(ScalarEvolution &SE, const SCEV *EltSize) const {
  const SCEV *Zero = SE.getZero(EltSize->getType());
  const SCEV *Acc = Zero;

  auto AddOverflow = [&](const SCEV *A, const SCEV *B) -> const SCEV * {
    if (!SE.willNotOverflow(Instruction::Add, /*IsSigned=*/true, A, B))
      return nullptr;
    return SE.getAddExpr(A, B);
  };

  for (const auto &[Stride, Loop] : Strides) {
    LLVM_DEBUG(dbgs() << "Validating stride: " << *Stride << " for loop "
                      << Loop->getHeader()->getName() << "\n");
    LLVM_DEBUG(dbgs() << "Current Acc: " << *Acc << "\n");
    if (!SE.isKnownPredicate(ICmpInst::ICMP_SLT, Acc, Stride))
      return false;
    const SCEV *AR =
        SE.getAddRecExpr(Zero, Stride, Loop, SCEV::NoWrapFlags::FlagAnyWrap);
    // TODO: Maybe need to check no-wrap flags of AR.
    LLVM_DEBUG(dbgs() << "Adding to Acc: " << *Acc << " + " << *AR << "\n");
    Acc = AddOverflow(Acc, AR);
    if (!Acc)
      return false;
  }

  LLVM_DEBUG(dbgs() << "Total Acc: " << *Acc << "\n");
  if (!SE.willNotOverflow(Instruction::Mul, /*IsSigned=*/true, Acc, EltSize))
    return false;
  return true;
}

void printDelinearization(raw_ostream &O, Function *F, LoopInfo *LI,
                          ScalarEvolution *SE) {
  O << "Printing analysis 'Delinearization' for function '" << F->getName()
    << "':";

  MySCEVRewriter Rewriter = MySCEVRewriter::create(*SE, *LI);
  ExecutionDomain ED(*SE);

  for (Instruction &Inst : instructions(F)) {
    // Only analyze loads and stores.
    if (!isa<StoreInst>(&Inst) && !isa<LoadInst>(&Inst))
      continue;

    const BasicBlock *BB = Inst.getParent();
    Loop *L = LI->getLoopFor(BB);
    // Only delinearize the memory access in the innermost loop.
    // Do not analyze memory accesses outside loops.
    if (!L)
      continue;

    const SCEV *AccessFn = SE->getSCEVAtScope(getPointerOperand(&Inst), L);

    const SCEVUnknown *BasePointer =
        dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFn));
    // Do not delinearize if we cannot find the base pointer.
    if (!BasePointer)
      break;

    AccessFn = SE->getMinusSCEV(AccessFn, BasePointer);

    O << "\n";
    O << "Inst:" << Inst << "\n";
    O << "AccessFunction: " << *AccessFn << "\n";

    const SCEV *MaxValue = findMax(AccessFn, *SE, ED);
    if (MaxValue) {
      O << "Maximum value: " << *MaxValue << "\n";
    } else {
      O << "Cannot find the maximum value\n";
    }

    Value *Ptr = BasePointer->getValue();
    const DataLayout &DL = F->getDataLayout();
    bool CheckForNonNull, CheckForFreed;
    uint64_t DerefBytes =
        Ptr->getPointerDereferenceableBytes(DL, CheckForNonNull, CheckForFreed);
    if (DerefBytes && !CheckForNonNull && !CheckForFreed && MaxValue) {
      O << "Dereferenceable bytes: " << DerefBytes << "\n";
      APInt RHS0(DL.getIndexSize(0) * 8, DerefBytes, false, false);
      O << "RHS0: " << RHS0 << "\n";
      auto [Pred, LHS, RHS] = InequalitySimpliler::simplify(
          ICmpInst::ICMP_ULT, MaxValue, RHS0, *SE);
      if (Pred == ICmpInst::ICMP_ULT)
        O << "Simplified inequality: " << *LHS << " <u ";
      if (RHS)
        O << *RHS;
      else
        O << "unknown";
      O << "\n";
    }

    // AccessFn = Rewriter.visit(AccessFn);
    // O << "Rewritten: " << *AccessFn << "\n";

    SmallVector<const SCEV *, 3> Subscripts, Sizes;

    auto IsDelinearizationFailed = [&]() {
      return Subscripts.size() == 0 || Sizes.size() == 0 ||
             Subscripts.size() != Sizes.size();
    };

    delinearize(*SE, AccessFn, Subscripts, Sizes, SE->getElementSize(&Inst));
    if (UseFixedSizeArrayHeuristic && IsDelinearizationFailed()) {
      Subscripts.clear();
      Sizes.clear();
      delinearizeFixedSizeArray(*SE, AccessFn, Subscripts, Sizes,
                                SE->getElementSize(&Inst));
    }

      if (IsDelinearizationFailed()) {
        O << "failed to delinearize\n";
        continue;
      }

      O << "Base offset: " << *BasePointer << "\n";
      O << "ArrayDecl[UnknownSize]";
      int Size = Subscripts.size();
      for (int i = 0; i < Size - 1; i++)
        O << "[" << *Sizes[i] << "]";
      O << " with elements of " << *Sizes[Size - 1] << " bytes.\n";

      O << "ArrayRef";
      for (int i = 0; i < Size; i++)
        O << "[" << *Subscripts[i] << "]";
      O << "\n";

      bool IsValid = validateDelinearizationResult(*SE, Sizes, Subscripts);
      O << "Delinearization validation: " << (IsValid ? "Succeeded" : "Failed")
        << "\n";

      const SCEV *EltSize = SE->getElementSize(&Inst);
      if (const auto RS = RelaxedStrides::create(*SE, AccessFn, EltSize,
                                                 L->getOutermostLoop())) {
        SmallVector<const SCEV *, 4> RelaxedSizes, RelaxedSubscripts;
        RS->relaxedDelinarize(*SE, RelaxedSizes, RelaxedSubscripts);
        O << "Relaxed delinearization result:\n";
        O << "ArrayDecl[UnknownSize]";
        int Size = Subscripts.size();
        for (int I = 0; I < Size; I++)
          O << "[" << *Sizes[I] << "]";
        O << " with elements of " << *EltSize << " bytes.\n";

        O << "ArrayRef";
        for (int I = 0; I < Size; I++)
          O << "[" << *Subscripts[I] << "]";
        O << "\n";
      } else {
        O << "Relaxed delinearization failed.\n";
      }
  }
}

} // end anonymous namespace

DelinearizationPrinterPass::DelinearizationPrinterPass(raw_ostream &OS)
    : OS(OS) {}
PreservedAnalyses DelinearizationPrinterPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  printDelinearization(OS, &F, &AM.getResult<LoopAnalysis>(F),
                       &AM.getResult<ScalarEvolutionAnalysis>(F));
  return PreservedAnalyses::all();
}
