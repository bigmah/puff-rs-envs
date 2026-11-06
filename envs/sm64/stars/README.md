# A new star

How a star gets its long run, the way Peach's Secret Slide (`pss-2`, 700 frames
from a blank slate) and Shoot into the Wild Blue (`wf-3`, 656) did. A star is
`<course>-<number>`; `./build/sm64_tool courses` lists the courses. Everything a
star keeps lands under its own name -- `build/sm64/<star>/`,
`checkpoints/<star>/`, `logs/<star>/` -- so stars never get in each other's way.
Below, `STAR=wf-3` stands for whichever star it is.

## 1. Name the star right

The number is the act select's, which is the order the course's stars are
listed in the game, not the order they are usually collected or the act a guide
files them under. Shoot into the Wild Blue looked like `wf-5` and is `wf-3`.
Check it against the objects in step 3: a star object's behavior parameters
name its index, and the index plus one is the number.

Stars 1 to 6 are entered for their own act unless `act` says otherwise; the
hundred coins (7) for act 1. A star that is in the course in every act (Wild
Blue is) still gets its own act by default, which is fine.

## 2. The savestate

```sh
./build/sm64_tool state --env.star $STAR
```

Half a second. It prints where Mario lands and the level, course and act he is
in; check the act is the one meant. Training makes it on its own if it is
missing, but making it first is how the next step gets a game to look at.

## 3. Where the star is

The policy sees the way to the star, and progress pays for closing on it, only
when `star_x`, `star_y`, `star_z` are set. A star placed by the level script is
an object in the pool from the first frame, so its place can be read straight
out of memory: Super Mario 64 (USA) keeps 240 objects of 0x260 bytes at
`0x8033D488`, each with `activeFlags` (s16) at +0x74, its position (three f32)
at +0xA0, its behavior parameters at +0x188 and its behavior at +0x20C. Every
star placed in a course has the same behavior (`800eefbc` in this recompile),
and the second byte of its parameters is its index: `02000000` is star 3.

For wf-3 this came from a throwaway copy of `sm64.c` with one extra command,
built by hand with the flags `build.sh` uses for `sm64_tool`:

```c
SM64* env = make_env(0, 0);
puf_reset(env);
for (int i = 0; i < 30; i++) { n64gym_pad(&env->gym, 0, 0.0f, 0.0f); n64gym_step(&env->gym, 1); }
for (int i = 0; i < 240; i++) {
    uint32_t o = 0x8033D488u + (uint32_t)i * 0x260u;
    if (!n64_s16(&env->gym, o + 0x74)) continue;
    printf("obj %3d bhv %08x pos %8.0f %8.0f %8.0f bp %08x\n", i, n64_u32(&env->gym, o + 0x20C),
           n64_f32(&env->gym, o + 0xA0), n64_f32(&env->gym, o + 0xA4), n64_f32(&env->gym, o + 0xA8),
           n64_u32(&env->gym, o + 0x188));
}
```

Dumping two acts and diffing the behaviors is a quick way to see which objects
an act adds.

A star that is *spawned* -- from a box, a boss, the red coins, the slide's timer
-- is not in the pool until it spawns, so there is nothing to read. Leave the
place unset; the first run that takes it prints where Mario touched it
(`sm64: <star> taken at (x, y, z)`), and that goes in the ini for the next run.
Without a place, progress pays nothing and the run leans on novelty and the
archive alone, so expect it to take longer to find.

## 4. stars/$STAR.ini

Only what differs from `sm64.ini`. `wf-3.ini` and `pss-2.ini` are the two to
copy from. The settings that have mattered:

- `star_x`, `star_y`, `star_z`: from step 3.
- `max_ticks`: the clock. Long enough for an explorer to get there, not for a
  run that knows the way: 1200 (40 s) for wf-3, whose record is 656. If the
  star itself has a time limit (pss-2's 21 seconds), the clock is that.
- `explore_ticks`: a longer clock for episodes that start in the archive away
  from the frontier, so the archive can build the way in stages: 1.5 times
  `max_ticks` has worked.
- `respawn = 1` for any course Mario can fall out of. Without it, 70% of
  Whomp's Fortress episodes ended early in a death. Expect deaths per episode
  in the single figures early on; that is falls, not a broken reload (a reload
  that failed every time would read in the hundreds).
- `novelty_episode = 0.0` once the star is placed: progress points the way, and
  novelty that never fades only pays for wandering.

## 5. Launching without make

`make sm64-speedrun STAR=$STAR` is the run, but every `make sm64-*` target runs
`./build.sh sm64` first, and on a Mac without Xcode that stops at RT64's Metal
shaders. When the env's C has not changed since the last build, the trainer,
`build/vecenv_sm64.dylib` and the game host are all current and the trainer can
be run directly with the target's arguments. Keep this as `logs/$STAR/run.sh`:

```sh
#!/bin/bash
cd /Users/tonyradtke/dev/puff-rs
STAR=wf-3
exec ./target/release/pufferl train sm64 --config envs/sm64/stars/$STAR.ini --env.star $STAR \
  --base.checkpoint-dir checkpoints/$STAR --base.log-dir logs/$STAR \
  --env.go-explore 0.75 --env.go-explore-star 0.75 --env.go-explore-frames 1 --env.go-explore-clock 1 --env.ghost 0.02 \
  --train.norm-adv 1 --train.learning-rate 0.005 --train.ent-coef 0.007 --train.ent-coef-rate 0.0025 \
  --train.ent-coef-damping 0.5 "$@"
```

If the env's C has changed, the library has to be rebuilt first, which needs
Xcode's Metal toolchain or `SM64_RENDERER=0` (a host that cannot draw, so no
watching in a window).

## 6. Smoke tests

Two short runs before the long one, each of which should end by itself and
write `logs/$STAR/sm64/<run id>.ini`:

```sh
logs/$STAR/run.sh --train.total-timesteps 500_000     # about 3.5 minutes
logs/$STAR/run.sh --train.total-timesteps 1_400_000   # about 10 minutes
```

In the second one, `archive` should keep growing (wf-3: 7.6K to 12.7K cubes)
and `closest` should come down (5,100 to 3,200 units). Steps a second start
around 2 to 3K with eight games.

## 7. The long run

Twelve hours from an empty archive and a new policy. Two things to plan for:

- **Steps a second fall as the archive grows** -- 2.8K at the start of wf-3,
  2.0K by four hours; pss-2 went from 3.5K to 1.1 to 1.9K over twenty. So
  `total_timesteps` cannot hit a wall-clock time exactly. 80M was about right
  for twelve hours of wf-3. Give it more than the time needs and stop it on
  the clock instead.
- **A run can die early.** wf-3's first run exited at 4h26m when one game's
  savestate would not reload at a reset ("thread 1 is waiting somewhere else
  than it was in the state"): `puf_reset` exits rather than retrying. Until
  that is fixed, a long run needs something to carry it on.

The trainer exits on SIGINT at once: checkpoints (every 50 epochs) and the
fastest runs are on disk, but it writes no log. Start the run, then the
watchdog, both under `nohup` so they outlive the terminal:

```sh
nohup logs/$STAR/run.sh --train.total-timesteps 100_000_000 > logs/$STAR/big-run.console 2>&1 < /dev/null & disown
nohup logs/$STAR/watchdog.sh > /dev/null 2>&1 < /dev/null & disown
```

`logs/$STAR/watchdog.sh` stops whatever is training at the deadline, and if
training stops before then (it crashed, or finished its steps), carries on from
the newest checkpoint with the archive seeded from the runs on file:

```sh
#!/bin/bash
cd /Users/tonyradtke/dev/puff-rs
set -m   # each background job its own process group, so SIGINT reaches its games
STAR=wf-3
END=$(( $(date +%s) + 12*3600 ))
N=1
CONSOLE=logs/$STAR/big-run.console
PID=$(pgrep -f "pufferl train sm64" | head -1)
while [ "$(date +%s)" -lt $END ]; do
    if [ -z "$PID" ] || ! kill -0 $PID 2>/dev/null; then
        RUN=$(ls -t checkpoints/$STAR/sm64 | head -1)
        CKPT=checkpoints/$STAR/sm64/$RUN/$(ls checkpoints/$STAR/sm64/$RUN | tail -1)
        N=$((N + 1))
        echo "$(date): carrying on from $CKPT" >> logs/$STAR/watchdog.log
        CONSOLE=logs/$STAR/big-run-$N.console
        logs/$STAR/run.sh --train.total-timesteps 100_000_000 --base.load-model-path "$CKPT" \
            --env.go-explore-seed 1 > $CONSOLE 2>&1 < /dev/null &
        PID=$!
    fi
    sleep 30
done
tail -c 2700 $CONSOLE > logs/$STAR/final-dashboard.txt
kill -INT -- -$(ps -o pgid= -p $PID | tr -d ' ')
echo "$(date): stopped at 12 hours" >> logs/$STAR/watchdog.log
```

Check the process group holds only the trainer and its eight `n64b-run` games
before trusting the kill: `ps -axo pid,pgid,comm | awk -v g=<pgid> '$2==g'`.

## 8. While it runs

`tail -c 2700 logs/$STAR/big-run.console` is the dashboard. The numbers:

- `closest`: units from the star at the nearest this log interval. Noisy; the
  trend is what counts.
- `perf`: the share of episodes taking the star. Per interval, so 0 on one
  check is not a stall. wf-3 had none for its first five hours or so and was
  at 45% by the end.
- `star_cubes`: archive cubes on the way to a star, 0 until the first.
- `start_perf`: stars taken from where the task starts rather than from the
  archive. 0 for the whole of both pss-2 and wf-3: every record so far is a run
  the archive assembled from starts partway along.
- `entropy`: should hold between about 2 and 3.

And `build/sm64/$STAR/fastest/`: the ten fastest runs to the star, named for
their frames. The folder keeps only ten, so copy a record out if it matters.

## 9. Keeping the record

```sh
./build/sm64_tool replay --env.star $STAR --config envs/sm64/stars/$STAR.ini
mkdir -p demos/<name> && cp build/sm64/$STAR/fastest.demo demos/<name>/star-NNN.demo
SM64_TRACE=30 ./build/sm64_tool replay demos/<name>/star-NNN.demo --config envs/sm64/stars/$STAR.ini > demos/<name>/route-NNN.txt
```

Before trusting a record, look at the end of the trace (`SM64_TRACE=2`): Mario
should be beside the star's place, in the action that takes it (`0x1904`, the
fall after grabbing a star, for wf-3). The goal is the save file's bit for
exactly this star, so another star in the course does not count, but the trace
is how to see the route makes sense. Then a short `demos/<name>/README.md` on
what the run does and how the hours went, and commit the ini and the demo
folder; `build/`, `checkpoints/` and `logs/` are ignored.
