/* Bloons TD 3: survive as many rounds as possible. A PufferLib 5.0 env around the
 * native port of the game (github.com/bigmah/bloons, native/btd3), whose
 * simulation matches the Flash original frame for frame. The game runs in Rust;
 * native/btd3-env there builds it into a static library with a small C ABI
 * (btd3_*), which build.sh links in. Nothing is drawn while training.
 *
 * One step is one decision, then ticks_per_step frames of the game at 40 fps. Each
 * round starts as soon as the last one ends, and towers go up while it runs.
 * Rounds past 50 are freeplay; the game ends when the lives run out or max_round
 * is cleared.
 *
 * Actions: three discrete heads
 *   kind: 0 wait, 1..8 build tower 1..8 (dart, tack, boomerang, bomb, ice, super,
 *         spikeopult, beacon), 9..12 buy upgrade 1..4 of the tower nearest the cell
 *   x, y: a 20 px cell of the 460 x 480 map, 23 x 24 of them
 * An action the game refuses (no cash, on the track, no tower there) does nothing.
 *
 * Rewards: round_reward per round cleared, pop_reward per layer popped, minus
 * life_penalty per life lost and refuse_penalty per refused action.
 *
 * Observations: per cell, whether a tower can't go there in an empty map, the
 * tower kind there and its upgrades, and the bloons in it; then lives, cash, round,
 * which towers are affordable, this and the next round's bloons, the bloons alive
 * by rank; then the bloons by how far along the track they are.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

typedef float obs_t;
#include "pufferenv.h"

#ifndef BTD3_DATA
#error "BTD3_DATA names the game's data/ directory; build.sh sets it"
#endif

#define GRID_W 23
#define GRID_H 24
#define CELL 20
#define KINDS 13
#define OBS_SIZE (GRID_W * GRID_H * 14 + 46 + 64)
#define NUM_ATNS 3
#define ACT_SIZES {KINDS, GRID_W, GRID_H}

// The Rust side, native/btd3-env/src/lib.rs
typedef struct Btd3Env Btd3Env;
typedef struct {
    int32_t rounds_cleared, lives_lost, pops, acted, cash_spent, done, lost;
    int32_t round, lives, cash, towers, upgrades;
} Btd3Step;
int32_t btd3_obs_size(void);
Btd3Env* btd3_new(const char* data_dir, int32_t track, int32_t difficulty, int32_t max_round, uint64_t seed);
void btd3_free(Btd3Env* env);
void btd3_reset(Btd3Env* env, uint64_t seed);
void btd3_step(Btd3Env* env, int32_t kind, int32_t x, int32_t y, int32_t ticks, Btd3Step* out);
void btd3_observe(const Btd3Env* env, float* obs);
int32_t btd3_towers(const Btd3Env* env, float* out, int32_t max);
int32_t btd3_bloons(const Btd3Env* env, float* out, int32_t max);
int32_t btd3_path(const Btd3Env* env, float* out, int32_t max);

// Required struct. Only use floats!
struct Log {
    float perf;            // round reached / 50
    float score;           // round reached
    float won;             // fraction of games that clear round 50
    float episode_return;
    float episode_length;  // agent steps
    float lives;           // lives left at the end
    float towers;          // towers standing at the end
    float upgrades;        // upgrades bought, at the end
    float pops;
    float refused;         // fraction of build and upgrade actions the game refused
    float waits;           // fraction of steps that waited
    float n;               // Required as the last field
};

typedef struct {
    float* path;
    int path_n;
    float towers[5 * 256];
    float bloons[3 * 4096];
    int last_kind, last_x, last_y, last_acted;
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
    Btd3Env* game;
    int track, difficulty, max_round, ticks_per_step;
    float round_reward, pop_reward, life_penalty, refuse_penalty;
    unsigned int episodes;
    Btd3Step last;
    float episode_return;
    int steps, pops, asks, refusals, waits;
};
typedef Env Bloons;

static const Color PUFF_RED = {187, 0, 0, 255};
static const Color PUFF_CYAN = {0, 187, 187, 255};
static const Color PUFF_WHITE = {241, 241, 241, 241};
static const Color PUFF_BACKGROUND = {6, 24, 24, 255};

static uint64_t episode_seed(Bloons* env) {
    uint64_t x = ((uint64_t)env->rng << 32) ^ (env->episodes * 0x9e3779b97f4a7c15ull) ^ 0xb100b5ull;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return x;
}

// Required function
void puf_reset(Bloons* env) {
    btd3_reset(env->game, episode_seed(env));
    env->episodes++;
    memset(&env->last, 0, sizeof(env->last));
    env->episode_return = 0.0f;
    env->steps = env->pops = env->asks = env->refusals = env->waits = 0;
    btd3_observe(env->game, env->agents[0].observations);
}

static void end_episode(Bloons* env) {
    Btd3Step* s = &env->last;
    Log* log = &env->log;
    // The round the game ended on: lost during it, or the last one cleared
    float round = (float)s->round;
    log->perf += round / 50.0f;
    log->score += round;
    log->won += s->round > 50 || (s->round == 50 && !s->lost);
    log->episode_return += env->episode_return;
    log->episode_length += env->steps;
    log->lives += s->lives;
    log->towers += s->towers;
    log->upgrades += s->upgrades;
    log->pops += env->pops;
    log->refused += env->asks ? (float)env->refusals / env->asks : 0.0f;
    log->waits += env->steps ? (float)env->waits / env->steps : 0.0f;
    log->n += 1;
    env->agents[0].terminals[0] = 1.0f;
    puf_reset(env);
}

// Required function
void puf_step(Bloons* env) {
    Agent* agent = &env->agents[0];
    int kind = (int)agent->actions[0];
    int x = (int)agent->actions[1];
    int y = (int)agent->actions[2];
    Btd3Step* s = &env->last;
    btd3_step(env->game, kind, x, y, env->ticks_per_step, s);

    float r = env->round_reward * s->rounds_cleared + env->pop_reward * s->pops
        - env->life_penalty * s->lives_lost;
    if (s->acted == 0) r -= env->refuse_penalty;
    agent->rewards[0] = r;
    agent->terminals[0] = 0.0f;

    env->episode_return += r;
    env->steps++;
    env->pops += s->pops;
    env->asks += s->acted >= 0;
    env->refusals += s->acted == 0;
    env->waits += s->acted < 0;
    if (env->client != NULL) {
        env->client->last_kind = kind;
        env->client->last_x = x;
        env->client->last_y = y;
        env->client->last_acted = s->acted;
    }
    if (s->done) {
        end_episode(env);
        return;
    }
    btd3_observe(env->game, agent->observations);
}

static Color tower_color(int kind) {
    static const Color colors[8] = {
        {150, 100, 50, 255},   // dart
        {230, 120, 200, 255},  // tack
        {240, 200, 40, 255},   // boomerang
        {40, 40, 40, 255},     // bomb
        {140, 220, 255, 255},  // ice
        {60, 90, 230, 255},    // super
        {120, 160, 60, 255},   // spikeopult
        {255, 255, 255, 255},  // beacon
    };
    return colors[kind < 0 ? 0 : kind > 7 ? 7 : kind];
}

static Color bloon_color(int rank) {
    static const Color colors[10] = {
        {230, 30, 30, 255}, {40, 120, 230, 255}, {60, 190, 60, 255}, {250, 220, 40, 255},
        {20, 20, 20, 255}, {245, 245, 245, 255}, {120, 120, 130, 255}, {200, 80, 220, 255},
        {170, 100, 40, 255}, {60, 90, 200, 255},
    };
    return colors[rank < 1 ? 0 : rank > 10 ? 9 : rank - 1];
}

static const char* KIND_NAMES[KINDS] = {
    "wait", "dart", "tack", "boomerang", "bomb", "ice", "super", "spikeopult", "beacon",
    "upgrade 1", "upgrade 2", "upgrade 3", "upgrade 4",
};

#define SCALE 2
#define SCREEN_W (460 * SCALE)
#define SCREEN_H (480 * SCALE + 60)

// Required function. Creates the window on first call
void puf_render(Bloons* env) {
    if (env->client == NULL) {
        InitWindow(SCREEN_W, SCREEN_H, "PufferLib Bloons TD 3");
        SetTargetFPS(30);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->path = (float*)malloc(2 * 8192 * sizeof(float));
        env->client->path_n = btd3_path(env->game, env->client->path, 8192);
        env->client->last_acted = -1;
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    Client* c = env->client;

    BeginDrawing();
    ClearBackground((Color){70, 140, 50, 255});
    for (int i = 1; i < c->path_n; i++) {
        Vector2 a = {c->path[2 * i - 2] * SCALE, c->path[2 * i - 1] * SCALE};
        Vector2 b = {c->path[2 * i] * SCALE, c->path[2 * i + 1] * SCALE};
        DrawLineEx(a, b, 22 * SCALE, (Color){200, 170, 110, 255});
    }
    int nt = btd3_towers(env->game, c->towers, 256);
    for (int i = 0; i < nt; i++) {
        float* t = &c->towers[5 * i];
        Color col = tower_color((int)t[2]);
        DrawCircleLines((int)(t[0] * SCALE), (int)(t[1] * SCALE), t[3] * SCALE, Fade(col, 0.35f));
        DrawCircle((int)(t[0] * SCALE), (int)(t[1] * SCALE), 9 * SCALE, col);
        int mask = (int)t[4];
        for (int k = 0; k < 4; k++) {
            if (mask & (1 << k)) {
                DrawCircle((int)(t[0] * SCALE) - 9 + 6 * k, (int)(t[1] * SCALE) + 13, 2, PUFF_RED);
            }
        }
    }
    int nb = btd3_bloons(env->game, c->bloons, 4096);
    for (int i = 0; i < nb; i++) {
        float* b = &c->bloons[3 * i];
        int rank = (int)b[2];
        float radius = rank >= 10 ? 16 : rank == 9 ? 9 : 6;
        DrawCircle((int)(b[0] * SCALE), (int)(b[1] * SCALE), radius * SCALE / 2 + 2, bloon_color(rank));
    }
    if (c->last_acted >= 0) {
        DrawRectangleLines(c->last_x * CELL * SCALE, c->last_y * CELL * SCALE, CELL * SCALE, CELL * SCALE,
            c->last_acted ? PUFF_CYAN : PUFF_RED);
    }
    DrawRectangle(0, 480 * SCALE, SCREEN_W, 60, PUFF_BACKGROUND);
    Btd3Step* s = &env->last;
    DrawText(TextFormat("Round %d   Lives %d   Cash %d   Towers %d   Upgrades %d",
        s->round, s->lives, s->cash, s->towers, s->upgrades), 12, 480 * SCALE + 8, 20, PUFF_WHITE);
    DrawText(TextFormat("last: %s (%d, %d) %s   ESC quit", KIND_NAMES[c->last_kind % KINDS], c->last_x, c->last_y,
        c->last_acted < 0 ? "" : c->last_acted ? "ok" : "refused"), 12, 480 * SCALE + 34, 18, PUFF_WHITE);
    EndDrawing();
    puf_web_vsync();
}

// Required function. Do not free the agent buffers: the vecenv owns them
void puf_close(Bloons* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client->path);
        free(env->client);
        env->client = NULL;
    }
    btd3_free(env->game);
    env->game = NULL;
}

// Required function: read this env's settings from [env] in its config
void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->track = dict_get(kwargs, "track");
    env->difficulty = dict_get(kwargs, "difficulty");
    env->max_round = dict_get(kwargs, "max_round");
    env->ticks_per_step = dict_get(kwargs, "ticks_per_step");
    env->round_reward = dict_get(kwargs, "round_reward");
    env->pop_reward = dict_get(kwargs, "pop_reward");
    env->life_penalty = dict_get(kwargs, "life_penalty");
    env->refuse_penalty = dict_get(kwargs, "refuse_penalty");
    if (btd3_obs_size() != OBS_SIZE) {
        fprintf(stderr, "bloons: the game observes %d floats and this header says %d; rebuild\n",
            btd3_obs_size(), OBS_SIZE);
        exit(1);
    }
    // Tracks are 1..8 in the config, as in the game's menu
    env->game = btd3_new(BTD3_DATA, env->track - 1, env->difficulty, env->max_round, env->rng);
    if (env->game == NULL) exit(1);
}

// Required function: the Log averaged over episodes, by name
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "won", log->won);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "lives", log->lives);
    dict_set(out, "towers", log->towers);
    dict_set(out, "upgrades", log->upgrades);
    dict_set(out, "pops", log->pops);
    dict_set(out, "refused", log->refused);
    dict_set(out, "waits", log->waits);
}
