/* Kronecker factorizations of an M x N matrix: the factor shapes, how many
 * lists of them there are, and whether a list of them really factors a matrix.
 * No raylib and no PufferLib, so kronecker_factorization.c (the command line)
 * and the env both use it.
 *
 * A list of shapes (a1,b1) (a2,b2) ... factors an M x N matrix when the a's
 * multiply to M and the b's to N. Every shape but 1x1 is allowed.
 */

#ifndef KRON_H
#define KRON_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned __int128 kron_count_t;
#define KRON_COUNT_MAX (~(kron_count_t)0)

// Sorted divisors of n, and how many there are. out may be NULL to only count
static inline int kron_divisors(uint64_t n, uint64_t* out) {
    int count = 0;
    for (uint64_t i = 1; i <= n / i; i++) {
        if (n % i == 0) count += i == n / i ? 1 : 2;
    }
    if (out != NULL) {
        // The small ones from the front, their partners n / i from the back
        int lo = 0, hi = count;
        for (uint64_t i = 1; i <= n / i; i++) {
            if (n % i != 0) continue;
            out[lo++] = i;
            if (i != n / i) out[--hi] = n / i;
        }
    }
    return count;
}

static inline int kron_index(const uint64_t* sorted, int count, uint64_t x) {
    int lo = 0, hi = count - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (sorted[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// --- Counting ----------------------------------------------------------------
// Every m on the way down divides M and every n divides N, so the count of each
// (m, n) is kept in a table indexed by their places in M's and N's divisors.

typedef struct {
    uint64_t *dm, *dn;
    int nm, nn;
    kron_count_t* memo;
    unsigned char* known;
    int overflow;  // a count went past 2^128 - 1 and was held there
} KronCounter;

static inline kron_count_t kron_count_rec(KronCounter* c, int im, int in) {
    uint64_t m = c->dm[im], n = c->dn[in];
    if (m == 1 && n == 1) return 1;  // One way to finish the current list
    size_t key = (size_t)im * c->nn + in;
    if (c->known[key]) return c->memo[key];

    kron_count_t sum = 0;
    for (int ia = 0; ia <= im; ia++) {
        uint64_t a = c->dm[ia];
        if (m % a != 0) continue;
        for (int ib = 0; ib <= in; ib++) {
            uint64_t b = c->dn[ib];
            if (n % b != 0 || (a == 1 && b == 1)) continue;
            kron_count_t rest = kron_count_rec(c,
                kron_index(c->dm, c->nm, m / a), kron_index(c->dn, c->nn, n / b));
            if (sum > KRON_COUNT_MAX - rest) {
                sum = KRON_COUNT_MAX;
                c->overflow = 1;
            } else {
                sum += rest;
            }
        }
    }
    c->known[key] = 1;
    c->memo[key] = sum;
    return sum;
}

// Ordered lists of factor shapes for an M x N matrix, none of them 1x1. With
// at_least_two, the list of the one shape M x N is left out. *overflow is set
// if the count did not fit and is 2^128 - 1 instead.
static inline kron_count_t kron_count(uint64_t M, uint64_t N, int at_least_two, int* overflow) {
    *overflow = 0;
    if (M == 1 && N == 1) return 0;  // No permitted nonempty list
    KronCounter c = {0};
    c.nm = kron_divisors(M, NULL);
    c.nn = kron_divisors(N, NULL);
    c.dm = (uint64_t*)malloc(c.nm * sizeof(uint64_t));
    c.dn = (uint64_t*)malloc(c.nn * sizeof(uint64_t));
    kron_divisors(M, c.dm);
    kron_divisors(N, c.dn);
    c.memo = (kron_count_t*)malloc((size_t)c.nm * c.nn * sizeof(kron_count_t));
    c.known = (unsigned char*)calloc((size_t)c.nm * c.nn, 1);
    kron_count_t total = kron_count_rec(&c, c.nm - 1, c.nn - 1);
    *overflow = c.overflow;
    free(c.dm);
    free(c.dn);
    free(c.memo);
    free(c.known);
    if (total == KRON_COUNT_MAX) return total;
    return total - (at_least_two ? 1 : 0);
}

// --- Enumerating -------------------------------------------------------------

// The most factors any list can have: each one halves M or N at least
#define KRON_MAX_FACTORS 128

// Called with each list in turn; return nonzero to stop
typedef int (*KronVisit)(const uint64_t shapes[][2], int count, void* ctx);

typedef struct {
    uint64_t shapes[KRON_MAX_FACTORS][2];
    int min_length;
    KronVisit visit;
    void* ctx;
} KronWalk;

static inline int kron_enumerate_rec(KronWalk* w, uint64_t m, uint64_t n, int depth) {
    if (m == 1 && n == 1) {
        return depth >= w->min_length ? w->visit(w->shapes, depth, w->ctx) : 0;
    }
    int nm = kron_divisors(m, NULL), nn = kron_divisors(n, NULL);
    uint64_t* dm = (uint64_t*)malloc(nm * sizeof(uint64_t));
    uint64_t* dn = (uint64_t*)malloc(nn * sizeof(uint64_t));
    kron_divisors(m, dm);
    kron_divisors(n, dn);
    int stop = 0;
    for (int ia = 0; ia < nm && !stop; ia++) {
        for (int ib = 0; ib < nn && !stop; ib++) {
            if (dm[ia] == 1 && dn[ib] == 1) continue;
            w->shapes[depth][0] = dm[ia];
            w->shapes[depth][1] = dn[ib];
            stop = kron_enumerate_rec(w, m / dm[ia], n / dn[ib], depth + 1);
        }
    }
    free(dm);
    free(dn);
    return stop;
}

// Every list of factor shapes for an M x N matrix, in the order count counts them
static inline void kron_enumerate(uint64_t M, uint64_t N, int at_least_two, KronVisit visit, void* ctx) {
    KronWalk w = {.min_length = at_least_two ? 2 : 1, .visit = visit, .ctx = ctx};
    kron_enumerate_rec(&w, M, N, 0);
}

// --- Checking a list against a matrix ----------------------------------------
// Matrices are row-major doubles.

// out = A (ar x ac) kron B (br x bc), which is ar*br x ac*bc
static inline void kron_product(const double* A, int ar, int ac, const double* B, int br, int bc, double* out) {
    int cols = ac * bc;
    for (int i = 0; i < ar; i++)
        for (int j = 0; j < ac; j++)
            for (int k = 0; k < br; k++)
                for (int l = 0; l < bc; l++)
                    out[(i * br + k) * cols + j * bc + l] = A[i * ac + j] * B[k * bc + l];
}

// The nearest A (a x b) kron B (m/a x n/b) to T (m x n), by Van Loan and
// Pitsianis: rearrange T so that each row is one entry of A times all of B,
// and the best pair is the best rank-1 fit of that, found by power iteration.
// It is exact when T is such a product.
static inline void kron_split(const double* T, int m, int n, int a, int b, double* A, double* B) {
    int p = m / a, q = n / b;
    int rows = a * b, cols = p * q;
#define KRON_R(r, c) T[((r) / b * p + (c) / q) * n + (r) % b * q + (c) % q]

    // Start from the row of the rearrangement with the most in it
    int best = 0;
    double best_norm = -1.0;
    for (int r = 0; r < rows; r++) {
        double s = 0.0;
        for (int c = 0; c < cols; c++) s += KRON_R(r, c) * KRON_R(r, c);
        if (s > best_norm) { best_norm = s; best = r; }
    }
    for (int c = 0; c < cols; c++) B[c] = KRON_R(best, c);

    for (int iter = 0; iter < 30; iter++) {
        double bb = 0.0;
        for (int c = 0; c < cols; c++) bb += B[c] * B[c];
        if (bb == 0.0) break;
        for (int r = 0; r < rows; r++) {
            double s = 0.0;
            for (int c = 0; c < cols; c++) s += KRON_R(r, c) * B[c];
            A[r] = s / bb;
        }
        double aa = 0.0;
        for (int r = 0; r < rows; r++) aa += A[r] * A[r];
        if (aa == 0.0) break;
        for (int c = 0; c < cols; c++) {
            double s = 0.0;
            for (int r = 0; r < rows; r++) s += KRON_R(r, c) * A[r];
            B[c] = s / aa;
        }
    }
#undef KRON_R
}

// Rebuild S (M x N) from the list of shapes and say how far off it is: the
// Frobenius norm of the difference over that of S. Each factor is split off
// what is left in turn, and the product of the factors so far is kept, so the
// end is the whole matrix the list gives. work holds 5 * M * N doubles.
static inline double kron_rebuild_error(const double* S, int M, int N, const int shapes[][2], int count,
        double* work) {
    int size = M * N;
    double *rest = work, *A = work + size, *B = work + 2 * size;
    double *prod = work + 3 * size, *next = work + 4 * size;
    for (int i = 0; i < size; i++) rest[i] = S[i];
    int m = M, n = N;      // what is left to factor
    int pr = 1, pc = 1;    // the product so far
    prod[0] = 1.0;
    for (int k = 0; k < count; k++) {
        int a = shapes[k][0], b = shapes[k][1];
        if (k == count - 1) {
            for (int i = 0; i < m * n; i++) A[i] = rest[i];
        } else {
            kron_split(rest, m, n, a, b, A, B);
            m /= a;
            n /= b;
            for (int i = 0; i < m * n; i++) rest[i] = B[i];
        }
        kron_product(prod, pr, pc, A, a, b, next);
        pr *= a;
        pc *= b;
        double* t = prod; prod = next; next = t;
    }
    double diff = 0.0, norm = 0.0;
    for (int i = 0; i < size; i++) {
        double d = prod[i] - S[i];
        diff += d * d;
        norm += S[i] * S[i];
    }
    return norm > 0.0 ? sqrt(diff / norm) : sqrt(diff);
}

#endif
