/* Sort: put a shuffled list of the numbers 0..SORT_N-1 in order in as few swaps as
 * possible. A PufferLib 5.0 env, built and trained like the platformer.
 *
 * Actions: two discrete heads, each an index 0..SORT_N-1
 *   swap the numbers at i and j; i == j swaps nothing and says "done"
 *
 * Rewards: every swap costs step_cost. Saying done pays solve_reward if the
 * list is sorted. If it is not, or the swaps run out (max_steps), it costs
 * fail_penalty scaled by how far the list still is from sorted: the full
 * penalty SORT_N-1 swaps away, a seventh of it one swap away for SORT_N = 8.
 * The list is looked at only then, once an episode, but an almost sorted list
 * beats an untouched one, which is what gives the policy something to climb. The fewest swaps that sort a list is SORT_N
 * minus its number of cycles, which is what score measures against.
 *
 * Observations: one-hot, SORT_N positions by SORT_N values, then the swaps used as a
 * fraction of max_steps.
 *
 * SORT_N is fixed at compile time, since the action heads are sized by it:
 * change SORT_N below and rebuild for another length.
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

typedef float obs_t;
#include "pufferenv.h"

#ifndef SORT_N
#define SORT_N 8
#endif

#define OBS_SIZE (SORT_N * SORT_N + 1)
#define NUM_ATNS 2
#define ACT_SIZES {SORT_N, SORT_N}

#define SCREEN_W 800
#define SCREEN_H 500

// Required struct. Only use floats!
struct Log {
    float perf;            // fraction of episodes that end sorted
    float score;           // fewest swaps / swaps used when sorted, 0 when not
    float episode_return;
    float episode_length;  // agent steps, the final "done" included
    float swaps;           // swaps used
    float extra_swaps;     // swaps beyond the fewest, in episodes that end sorted
    float wrong;           // fraction of episodes that say done on an unsorted list
    float timeouts;        // fraction of episodes that run out of swaps
    float n;               // Required as the last field
};

typedef struct {
    int last_i, last_j;  // the last swap, highlighted
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
    int max_steps;     // swaps allowed before the episode fails
    float step_cost;
    float solve_reward;
    float fail_penalty;
    int list[SORT_N];
    int min_swaps;     // fewest swaps that sort the list the episode started with
    int swaps;
    float episode_return;
    int wins, losses;  // lifetime counters for the window
};
typedef Env Sort;

static const Color PUFF_RED = {187, 0, 0, 255};
static const Color PUFF_CYAN = {0, 187, 187, 255};
static const Color PUFF_WHITE = {241, 241, 241, 241};
static const Color PUFF_BACKGROUND = {6, 24, 24, 255};
static const Color HIGHLIGHT = {255, 200, 40, 255};

// xorshift32. vecenv seeds env i with i, so the state is mixed away from 0 once in puf_init
static unsigned int next_rand(Sort* env) {
    unsigned int x = env->rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    env->rng = x;
    return x;
}

static int is_sorted(Sort* env) {
    for (int k = 0; k < SORT_N; k++) {
        if (env->list[k] != k) return 0;
    }
    return 1;
}

// SORT_N minus the number of cycles in the permutation
static int count_min_swaps(Sort* env) {
    int seen[SORT_N] = {0};
    int cycles = 0;
    for (int k = 0; k < SORT_N; k++) {
        if (seen[k]) continue;
        cycles++;
        for (int m = k; !seen[m]; m = env->list[m]) seen[m] = 1;
    }
    return SORT_N - cycles;
}

void compute_observations(Sort* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    for (int k = 0; k < SORT_N; k++) {
        obs[k * SORT_N + env->list[k]] = 1.0f;
    }
    obs[SORT_N * SORT_N] = (float)env->swaps / env->max_steps;
}

// Required function
void puf_reset(Sort* env) {
    // Fisher-Yates. The odd sorted list (1 in SORT_N!) is kept: the right move is to say done
    for (int k = 0; k < SORT_N; k++) env->list[k] = k;
    for (int k = SORT_N - 1; k > 0; k--) {
        int m = next_rand(env) % (k + 1);
        int t = env->list[k];
        env->list[k] = env->list[m];
        env->list[m] = t;
    }
    env->min_swaps = count_min_swaps(env);
    env->swaps = 0;
    env->episode_return = 0.0f;
    compute_observations(env);
}

enum { END_SORTED, END_WRONG, END_TIMEOUT };

// The cost of ending unsorted: fail_penalty for a list as far from sorted as one can be
static float unsorted_penalty(Sort* env) {
    return env->fail_penalty * count_min_swaps(env) / (SORT_N - 1);
}

static void end_episode(Sort* env, int outcome) {
    Agent* agent = &env->agents[0];
    int sorted = outcome == END_SORTED;
    env->episode_return += agent->rewards[0];
    env->log.perf += sorted;
    // A list that starts sorted and is left alone is a perfect 1
    env->log.score += !sorted ? 0.0f : env->swaps == 0 ? 1.0f : (float)env->min_swaps / env->swaps;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->swaps + (outcome != END_TIMEOUT);
    env->log.swaps += env->swaps;
    env->log.extra_swaps += sorted ? env->swaps - env->min_swaps : 0.0f;
    env->log.wrong += outcome == END_WRONG;
    env->log.timeouts += outcome == END_TIMEOUT;
    env->log.n += 1;
    env->wins += sorted;
    env->losses += !sorted;
    agent->terminals[0] = 1.0f;
    puf_reset(env);
}

// Required function
void puf_step(Sort* env) {
    Agent* agent = &env->agents[0];
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;

    float a_i = agent->actions[0];
    float a_j = agent->actions[1];
    int i = (a_i >= 0.0f && a_i < SORT_N) ? (int)a_i : 0;
    int j = (a_j >= 0.0f && a_j < SORT_N) ? (int)a_j : 0;
    if (env->client != NULL) {
        env->client->last_i = i;
        env->client->last_j = j;
    }

    if (i == j) {
        if (is_sorted(env)) {
            agent->rewards[0] = env->solve_reward;
            end_episode(env, END_SORTED);
        } else {
            agent->rewards[0] = -unsorted_penalty(env);
            end_episode(env, END_WRONG);
        }
        return;
    }

    int t = env->list[i];
    env->list[i] = env->list[j];
    env->list[j] = t;
    env->swaps++;
    agent->rewards[0] = -env->step_cost;

    if (env->swaps >= env->max_steps) {
        agent->rewards[0] -= unsorted_penalty(env);
        end_episode(env, END_TIMEOUT);
        return;
    }
    env->episode_return += agent->rewards[0];
    compute_observations(env);
}

// Required function. Creates the window on first call
void puf_render(Sort* env) {
    if (env->client == NULL) {
        InitWindow(SCREEN_W, SCREEN_H, "PufferLib Sort");
        SetTargetFPS(4);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->last_i = env->client->last_j = -1;
    }

    // Standard across puffer envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    int margin = 40;
    int top = 100;
    int bar_w = (SCREEN_W - 2 * margin) / SORT_N;
    int max_h = SCREEN_H - top - margin;
    for (int k = 0; k < SORT_N; k++) {
        int h = max_h * (env->list[k] + 1) / SORT_N;
        Color c = env->list[k] == k ? PUFF_CYAN : PUFF_RED;
        if (k == env->client->last_i || k == env->client->last_j) c = HIGHLIGHT;
        DrawRectangle(margin + k * bar_w + 4, SCREEN_H - margin - h, bar_w - 8, h, c);
        DrawText(TextFormat("%d", env->list[k]), margin + k * bar_w + bar_w / 2 - 6,
            SCREEN_H - margin + 8, 20, PUFF_WHITE);
    }

    DrawText(TextFormat("Swaps %d   Fewest %d   Sorted %d   Failed %d",
        env->swaps, env->min_swaps, env->wins, env->losses), 16, 12, 20, PUFF_WHITE);
    DrawText("ESC quit", 16, 40, 20, PUFF_WHITE);

    EndDrawing();
    puf_web_vsync();
}

// Required function. Do not free the agent buffers: the vecenv owns them
void puf_close(Sort* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

// Required function: read this env's settings from [env] in its config
void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_steps = dict_get(kwargs, "max_steps");
    env->step_cost = dict_get(kwargs, "step_cost");
    env->solve_reward = dict_get(kwargs, "solve_reward");
    env->fail_penalty = dict_get(kwargs, "fail_penalty");
    env->rng = env->rng * 2654435761u + 0x9e3779b9u;
    if (env->rng == 0) env->rng = 1;
}

// Required function: the Log averaged over episodes, by name
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "swaps", log->swaps);
    dict_set(out, "extra_swaps", log->extra_swaps);
    dict_set(out, "wrong", log->wrong);
    dict_set(out, "timeouts", log->timeouts);
}
