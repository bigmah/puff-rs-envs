/* Count, and list, the Kronecker factorizations of an M x N matrix.
 *
 *   ./build/kronecker_factorization M N [--count-only] [--limit N]
 *
 * Every ordered list of factor shapes (a, b), none of them 1x1, whose a's
 * multiply to M and b's to N, with at least two in it: one list a line, as
 * 2x3 ⊗ 4x1, then how many there are.
 */

#include <stdio.h>
#include <string.h>
#include "kron.h"

static void usage(void) {
    fprintf(stderr, "usage: kronecker_factorization M N [--count-only] [--limit N]\n");
    exit(2);
}

static int parse_dim(const char* s, uint64_t* out) {
    char* end;
    long long v = strtoll(s, &end, 10);
    if (*s == '\0' || *end != '\0') usage();
    if (v < 1) return 0;
    *out = (uint64_t)v;
    return 1;
}

typedef struct {
    long long limit;  // -1 for no limit
    long long printed;
} Printer;

static int print_list(const uint64_t shapes[][2], int count, void* ctx) {
    Printer* p = (Printer*)ctx;
    if (p->limit >= 0 && p->printed >= p->limit) return 1;
    for (int k = 0; k < count; k++) {
        printf("%s%llux%llu", k ? " ⊗ " : "",
            (unsigned long long)shapes[k][0], (unsigned long long)shapes[k][1]);
    }
    printf("\n");
    p->printed++;
    return 0;
}

// With commas between the thousands, as Python's f"{n:,}"
static void print_count(kron_count_t n) {
    char digits[48];
    int len = 0;
    do {
        digits[len++] = '0' + (int)(n % 10);
        n /= 10;
    } while (n > 0);
    for (int i = len - 1; i >= 0; i--) {
        putchar(digits[i]);
        if (i > 0 && i % 3 == 0) putchar(',');
    }
    putchar('\n');
}

int main(int argc, char** argv) {
    const char* dims[2];
    int num_dims = 0;
    int count_only = 0;
    long long limit = -1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--count-only") == 0) {
            count_only = 1;
        } else if (strcmp(argv[i], "--limit") == 0) {
            if (++i >= argc) usage();
            char* end;
            limit = strtoll(argv[i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0') usage();
        } else if (num_dims < 2) {
            dims[num_dims++] = argv[i];
        } else {
            usage();
        }
    }
    if (num_dims != 2) usage();

    uint64_t M, N;
    if (!parse_dim(dims[0], &M) || !parse_dim(dims[1], &N)) {
        fprintf(stderr, "Dimensions must be positive integers.\n");
        return 1;
    }

    if (!count_only) {
        Printer p = {.limit = limit};
        kron_enumerate(M, N, 1, print_list, &p);
    }

    int overflow;
    kron_count_t count = kron_count(M, N, 1, &overflow);
    if (overflow) {
        printf("more than ");
    }
    print_count(count);
    return 0;
}
