/* Kronecker factorization: given a real matrix S that is a Kronecker product,
 * name the shapes of its factors, with as few numbers in them as there can be.
 * A PufferLib 5.0 env, built and trained like the platformer.
 *
 * Each episode, S (rows x cols) is made as A1 ⊗ A2 ⊗ ... of random factors with
 * random Gaussian entries, their shapes picked one at a time from the shapes
 * that are left (kron.h's factor shapes), at least two of them. The policy
 * then names factor shapes one at a time, first to last.
 *
 * Actions: two discrete heads, a divisor of the rows left and one of the
 * columns left, each by its place in their sorted divisors (wrapping around).
 * The pair is the next factor's shape; 1x1 means "the rest is the last factor".
 * The episode ends when the shapes account for all of S.
 *
 * Rewards, once, at the end: the list is checked by rebuilding S from it --
 * each factor split off what is left by the nearest Kronecker product, the
 * factors multiplied back together -- and it is a factorization if that comes
 * back to S. One pays solve_reward * best_params / params, where params is the
 * numbers in the factors, a1*b1 + a2*b2 + ..., and best_params those in the
 * factors S was made from: 1 for the fewest. Anything else costs fail_penalty,
 * as does a single factor (S itself) when at_least_two is set.
 *
 * Random factors cannot be factored further, so every factorization of S
 * groups neighbouring factors it was made from (or swaps a row vector and the
 * column vector beside it, which costs the same), and grouping only adds
 * numbers: the factors S was made from are the fewest.
 *
 * Observations: S over its RMS in the top left of a KRON_MAX x KRON_MAX grid,
 * then one-hots of the rows and columns left to factor.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "kron.h"

typedef float obs_t;
#include "pufferenv.h"

// The largest rows and cols the config may ask for. No number up to 32 has more
// than 8 divisors (24 and 30 have 8), which is what the action heads are.
#define KRON_MAX 32
#define KRON_DIVS 8
#define KRON_FACTORS 10  // 32 x 32 halves at most 5 + 5 times

#define OBS_SIZE (KRON_MAX * KRON_MAX + 2 * KRON_MAX)
#define NUM_ATNS 2
#define ACT_SIZES {KRON_DIVS, KRON_DIVS}

#define SCREEN_W 900
#define SCREEN_H 700

// Required struct. Only use floats!
struct Log {
    float perf;            // fraction of episodes that end on a true factorization
    float score;           // best_params / params when it does, 0 when not
    float optimal;         // fraction that end on one with the fewest numbers
    float episode_return;
    float episode_length;  // factors named
    float params;          // numbers in the factors named, in true factorizations
    float best_params;     // numbers in the factors S was made from
    float n;               // Required as the last field
};

typedef struct {
    int shown_shapes[KRON_FACTORS][2];  // the last list named, for the window
    int shown_count;
    int shown_true[KRON_FACTORS][2];
    int shown_true_count;
    int shown_ok;
} Client;

// Required struct
struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    Client* client;
    int num_agents;
    unsigned int rng;  // set by vecenv to the env's index
    int rows, cols;
    int at_least_two;
    float solve_reward;
    float fail_penalty;
    float tolerance;   // largest relative error that still counts as S
    double S[KRON_MAX * KRON_MAX];
    float rms;
    int true_shapes[KRON_FACTORS][2];  // the factors S was made from
    int true_count;
    int best_params;
    int shapes[KRON_FACTORS][2];       // the factors named so far
    int count;
    int m, n;                          // rows and columns not yet accounted for
};
typedef Env KroneckerFactorization;

static const Color PUFF_RED = {187, 0, 0, 255};
static const Color PUFF_CYAN = {0, 187, 187, 255};
static const Color PUFF_WHITE = {241, 241, 241, 241};
static const Color PUFF_BACKGROUND = {6, 24, 24, 255};

// xorshift32. vecenv seeds env i with i, so the state is mixed away from 0 once in puf_init
static unsigned int next_rand(KroneckerFactorization* env) {
    unsigned int x = env->rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    env->rng = x;
    return x;
}

static double uniform(KroneckerFactorization* env) {
    return ((next_rand(env) >> 8) + 0.5) / 16777216.0;
}

// Box-Muller
static double gaussian(KroneckerFactorization* env) {
    return sqrt(-2.0 * log(uniform(env))) * cos(6.283185307179586 * uniform(env));
}

// A random factor shape for an m x n that is left: any divisor pair but 1x1
static void random_shape(KroneckerFactorization* env, int m, int n, int* a, int* b) {
    uint64_t dm[KRON_DIVS], dn[KRON_DIVS];
    int nm = kron_divisors(m, dm), nn = kron_divisors(n, dn);
    int pick = next_rand(env) % (nm * nn - 1) + 1;  // 0 would be 1x1
    *a = (int)dm[pick / nn];
    *b = (int)dn[pick % nn];
}

void compute_observations(KroneckerFactorization* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    for (int i = 0; i < env->rows; i++) {
        for (int j = 0; j < env->cols; j++) {
            obs[i * KRON_MAX + j] = env->S[i * env->cols + j] / env->rms;
        }
    }
    obs[KRON_MAX * KRON_MAX + env->m - 1] = 1.0f;
    obs[KRON_MAX * KRON_MAX + KRON_MAX + env->n - 1] = 1.0f;
}

// Required function
void puf_reset(KroneckerFactorization* env) {
    // The shapes S is made from: at least two, so it has a factorization to find
    do {
        env->true_count = 0;
        int m = env->rows, n = env->cols;
        while (m > 1 || n > 1) {
            int a, b;
            random_shape(env, m, n, &a, &b);
            env->true_shapes[env->true_count][0] = a;
            env->true_shapes[env->true_count][1] = b;
            env->true_count++;
            m /= a;
            n /= b;
        }
    } while (env->true_count < 2);

    double factor[KRON_MAX * KRON_MAX], next[KRON_MAX * KRON_MAX];
    int pr = 1, pc = 1;
    env->S[0] = 1.0;
    env->best_params = 0;
    for (int k = 0; k < env->true_count; k++) {
        int a = env->true_shapes[k][0], b = env->true_shapes[k][1];
        for (int i = 0; i < a * b; i++) factor[i] = gaussian(env);
        kron_product(env->S, pr, pc, factor, a, b, next);
        pr *= a;
        pc *= b;
        memcpy(env->S, next, pr * pc * sizeof(double));
        env->best_params += a * b;
    }
    double sum = 0.0;
    for (int i = 0; i < env->rows * env->cols; i++) sum += env->S[i] * env->S[i];
    env->rms = (float)sqrt(sum / (env->rows * env->cols));
    if (env->rms == 0.0f) env->rms = 1.0f;

    env->count = 0;
    env->m = env->rows;
    env->n = env->cols;
    compute_observations(env);
}

static void end_episode(KroneckerFactorization* env) {
    Agent* agent = &env->agents[0];
    int params = 0;
    for (int k = 0; k < env->count; k++) params += env->shapes[k][0] * env->shapes[k][1];

    int ok = !(env->at_least_two && env->count < 2);
    if (ok) {
        double work[5 * KRON_MAX * KRON_MAX];
        ok = kron_rebuild_error(env->S, env->rows, env->cols, (const int (*)[2])env->shapes,
            env->count, work) <= env->tolerance;
    }

    float reward = ok ? env->solve_reward * env->best_params / params : -env->fail_penalty;
    agent->rewards[0] = reward;
    agent->terminals[0] = 1.0f;
    env->log.perf += ok;
    env->log.score += ok ? (float)env->best_params / params : 0.0f;
    env->log.optimal += ok && params <= env->best_params;
    env->log.episode_return += reward;
    env->log.episode_length += env->count;
    env->log.params += ok ? params : 0.0f;
    env->log.best_params += env->best_params;
    env->log.n += 1;

    if (env->client != NULL) {
        Client* c = env->client;
        memcpy(c->shown_shapes, env->shapes, sizeof(env->shapes));
        c->shown_count = env->count;
        memcpy(c->shown_true, env->true_shapes, sizeof(env->true_shapes));
        c->shown_true_count = env->true_count;
        c->shown_ok = ok;
    }
    puf_reset(env);
}

// Required function
void puf_step(KroneckerFactorization* env) {
    Agent* agent = &env->agents[0];
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;

    uint64_t dm[KRON_DIVS], dn[KRON_DIVS];
    int nm = kron_divisors(env->m, dm), nn = kron_divisors(env->n, dn);
    float a_i = agent->actions[0];
    float a_j = agent->actions[1];
    int i = (a_i >= 0.0f && a_i < KRON_DIVS) ? (int)a_i : 0;
    int j = (a_j >= 0.0f && a_j < KRON_DIVS) ? (int)a_j : 0;
    int a = (int)dm[i % nm], b = (int)dn[j % nn];
    if (a == 1 && b == 1) {
        a = env->m;
        b = env->n;
    }

    env->shapes[env->count][0] = a;
    env->shapes[env->count][1] = b;
    env->count++;
    env->m /= a;
    env->n /= b;
    if (env->m == 1 && env->n == 1) {
        end_episode(env);
        return;
    }
    compute_observations(env);
}

static void draw_shapes(const int shapes[][2], int count, int x, int y, Color color) {
    char text[256] = "";
    for (int k = 0; k < count; k++) {
        char part[32];
        snprintf(part, sizeof(part), "%s%dx%d", k ? " (x) " : "", shapes[k][0], shapes[k][1]);
        strncat(text, part, sizeof(text) - strlen(text) - 1);
    }
    DrawText(text, x, y, 20, color);
}

// Required function. Creates the window on first call
void puf_render(KroneckerFactorization* env) {
    if (env->client == NULL) {
        InitWindow(SCREEN_W, SCREEN_H, "PufferLib Kronecker Factorization");
        SetTargetFPS(2);
        env->client = (Client*)calloc(1, sizeof(Client));
    }

    // Standard across puffer envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // S as a heat map: red negative, cyan positive
    int top = 140, side = SCREEN_H - top - 20;
    int cell = side / (env->rows > env->cols ? env->rows : env->cols);
    for (int i = 0; i < env->rows; i++) {
        for (int j = 0; j < env->cols; j++) {
            float v = (float)(env->S[i * env->cols + j] / env->rms) / 2.0f;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            Color c = v >= 0.0f ? PUFF_CYAN : PUFF_RED;
            c.a = (unsigned char)(255 * fabsf(v));
            DrawRectangle(20 + j * cell, top + i * cell, cell - 1, cell - 1, c);
        }
    }

    Client* c = env->client;
    DrawText(TextFormat("S is %d x %d   ESC quit", env->rows, env->cols), 16, 12, 20, PUFF_WHITE);
    DrawText("Made from:", 16, 44, 20, PUFF_WHITE);
    draw_shapes(c->shown_true, c->shown_true_count, 140, 44, PUFF_WHITE);
    DrawText("Named:", 16, 76, 20, PUFF_WHITE);
    draw_shapes(c->shown_shapes, c->shown_count, 140, 76, c->shown_ok ? PUFF_CYAN : PUFF_RED);
    DrawText(c->shown_count == 0 ? "" : c->shown_ok ? "rebuilds S" : "does not rebuild S",
        16, 108, 20, c->shown_ok ? PUFF_CYAN : PUFF_RED);

    EndDrawing();
    puf_web_vsync();
}

// Required function. Do not free the agent buffers: the vecenv owns them
void puf_close(KroneckerFactorization* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

// Required function: read this env's settings from [env] in its config
void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->rows = dict_get(kwargs, "rows");
    env->cols = dict_get(kwargs, "cols");
    env->at_least_two = dict_get(kwargs, "at_least_two");
    env->solve_reward = dict_get(kwargs, "solve_reward");
    env->fail_penalty = dict_get(kwargs, "fail_penalty");
    env->tolerance = dict_get(kwargs, "tolerance");
    if (env->rows < 1 || env->cols < 1 || env->rows > KRON_MAX || env->cols > KRON_MAX
            || env->rows * env->cols < 2) {
        fprintf(stderr, "kronecker_factorization: rows and cols must be 1 to %d, and not both 1\n",
            KRON_MAX);
        exit(1);
    }
    // S is made from two factors or more, which a prime number of entries cannot be
    int overflow;
    if (kron_count(env->rows, env->cols, 1, &overflow) == 0) {
        fprintf(stderr, "kronecker_factorization: a %d x %d matrix has no factorization into two or more\n",
            env->rows, env->cols);
        exit(1);
    }
    env->rng = env->rng * 2654435761u + 0x9e3779b9u;
    if (env->rng == 0) env->rng = 1;
}

// Required function: the Log averaged over episodes, by name
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "optimal", log->optimal);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "params", log->params / (log->perf > 0.0f ? log->perf : 1.0f));
    dict_set(out, "best_params", log->best_params);
}
