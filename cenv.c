/* The bridge to a PufferLib 5.0 C env: one env at a time, through a C ABI, built into a
 * shared library per env, which the trainer loads (src/cenv.rs).
 *
 *   ./build.sh platformer    ->  build/cenv_platformer.dylib (.so on Linux)
 *
 * Nothing else is here. The trainer's own vecenv (src/vecenv.rs) makes the envs, owns
 * the buffers they read and write, steps them on its threads and averages their logs,
 * for C envs and Rust envs alike. This makes one env, points its agents at their slices
 * of those buffers, and calls the env's puf_* functions on it.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include ENV_HEADER

#define CENV_API __attribute__((visibility("default")))

static const int act_sizes[] = ACT_SIZES;

CENV_API const char* cenv_name(void) {
    return PUFFER_ENV_NAME;
}

CENV_API int cenv_num_atns(void) {
    return NUM_ATNS;
}

CENV_API const int* cenv_act_sizes(void) {
    return act_sizes;
}

// 1 for unsigned char observations, 4 for float
CENV_API int cenv_obs_bytes(void) {
    return sizeof(obs_t);
}

// Only final once an env is made: an env may size its observation from its config
CENV_API int cenv_obs_size(void) {
    return OBS_SIZE;
}

// The [env] section of the config, as written there, for every env to read in puf_init.
// It lives as long as the envs made from it.
CENV_API Dict* cenv_config(int num_keys, const char** keys, const char** values) {
    Dict* kwargs = (Dict*)calloc(1, sizeof(Dict));
    kwargs->name = dict_strdup("env");
    for (int i = 0; i < num_keys; i++) {
        puf_ini_set(kwargs, keys[i], values[i]);
    }
    return kwargs;
}

CENV_API void cenv_config_free(Dict* kwargs) {
    dict_clear(kwargs);
    free(kwargs);
}

// The index-th env, told its index in rng as PufferLib's vecenv tells it
CENV_API Env* cenv_new(Dict* kwargs, int index) {
    Env* env = (Env*)calloc(1, sizeof(Env));
    env->rng = index;
    puf_init(env, kwargs);
    env->tag = 0;
    env->boundary_reached = 0;
    return env;
}

CENV_API int cenv_num_agents(Env* env) {
    return env->num_agents;
}

// Point the env's agents at their slices of the trainer's buffers, agent after agent
CENV_API void cenv_bind(Env* env, obs_t* observations, float* actions, float* rewards,
        float* terminals, unsigned char* action_mask, int mask_size) {
    for (int s = 0; s < env->num_agents; s++) {
        Agent* agent = &env->agents[s];
        agent->observations = observations + (size_t)s * OBS_SIZE;
        agent->actions = actions + (size_t)s * NUM_ATNS;
        agent->rewards = rewards + s;
        agent->terminals = terminals + s;
        agent->action_mask = action_mask + (size_t)s * mask_size;
    }
}

CENV_API void cenv_reset(Env* env) {
    puf_reset(env);
}

CENV_API void cenv_step(Env* env) {
    puf_step(env);
}

// The Log is floats with n last: how many there are, a copy of this env's, and a clear
CENV_API int cenv_log_fields(void) {
    return sizeof(Log) / sizeof(float);
}

CENV_API void cenv_log_take(Env* env, float* out, int clear) {
    memcpy(out, &env->log, sizeof(Log));
    if (clear) {
        memset(&env->log, 0, sizeof(Log));
    }
}

/* What puf_log makes of a Log the trainer has averaged, by name: up to max names of
 * key_bytes each, and their values. Returns how many it wrote. */
CENV_API int cenv_log_names(const float* averaged, int max, int key_bytes, char* keys, double* values) {
    Log log;
    memcpy(&log, averaged, sizeof(Log));
    Dict out = {0};
    puf_log(&log, &out);
    int count = out.size < max ? out.size : max;
    for (int i = 0; i < count; i++) {
        snprintf(keys + (size_t)i * key_bytes, key_bytes, "%s", out.items[i].key);
        values[i] = out.items[i].value;
    }
    dict_clear(&out);
    return count;
}

CENV_API void cenv_render(Env* env) {
    puf_render(env);
}

// Whether a raylib window an env opened has been closed. An env that draws some other way never is.
CENV_API int cenv_window_closed(void) {
    return IsWindowReady() && WindowShouldClose();
}

CENV_API void cenv_close(Env* env) {
    puf_close(env);
    free(env);
}
