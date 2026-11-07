# puff-rs-envs

Envs for [puff-rs](https://github.com/bigmah/puff-rs), PufferLib 5.0 in Rust: above all an agent that speedruns stars in the real Super Mario 64.

<p align="center">
  <a href="demos/peach-slide/star-674-touch.mp4">
    <img src="demos/peach-slide/star-674-touch.webp" width="480" alt="Mario takes the star of Peach's Secret Slide in 674 frames, jumping off the upper track and falling onto the lower one">
  </a>
</p>

Above: the under-21-seconds star of Peach's Secret Slide in **674 frames (22.5 s)** from the course's start to touching the star, by a route training found on its own. Ten seconds in, it jumps off the upper track and free-falls 6,750 units onto the lower one. Click it for the [full-quality video](demos/peach-slide/star-674-touch.mp4).

| star | frames | how |
|---|---|---|
| Peach's Secret Slide, under 21 s (`pss-2`) | **674** | robustifying and Go-Explore, from runs on file ([`demos/peach-slide/`](demos/peach-slide/)) |
| Peach's Secret Slide, under 21 s (`pss-2`) | **700** | from a blank slate: a new policy and an empty archive, 140M steps |
| Shoot into the Wild Blue (`wf-3`) | **656** | from a blank slate, twelve hours ([`demos/wild-blue/`](demos/wild-blue/)) |

What's here:

- **Super Mario 64** (`envs/sm64/`). The real cartridge as an environment. [N64Bundler](https://github.com/bigmah/n64bundler) statically recompiles your ROM to native code, one copy of the game runs per agent, and the task is to get one named star as fast as possible.
- **Bloons TD 3** (`envs/bloons/`), on the native Rust port of the game's simulation.
- **Small PufferLib C envs**: a platformer (`envs/platformer/`, trains to the flag in 45 seconds), `sort` and `kronecker_factorization`.

The trainer isn't here. Cargo fetches puff-rs, and `src/main.rs` hands it this repo's envs.

## Quick start

You need Rust ([rustup](https://rustup.rs)) and a C compiler. The first `make` target that needs something fetches and builds it: puff-rs (through cargo), the submodules, raylib, N64Bundler and the recompiled game. `make setup` does the fetching and the trainer's build up front if you'd rather.

**macOS**

```sh
brew install libomp cmake ninja   # sm64 only
```

sm64 also needs Xcode, whose Metal toolchain compiles the renderer's shaders. Xcode 26 downloads it separately (`xcodebuild -downloadComponent MetalToolchain`), and the build says so if it's missing.

**Linux** (Debian and Ubuntu names)

```sh
sudo apt install build-essential clang cmake ninja-build git curl python3 libsdl2-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev
```

### Super Mario 64

The one thing this needs from you is your own dump of the cartridge, Super Mario 64 (USA). Put it at `sm64.z64` in the top of the checkout, or point `SM64_ROM` at it. `build.sh` checks that it is that exact dump (game id `NSME`, ROM hash `8a90daa33e09a265`), because the env reads Mario out of memory at the addresses that cartridge uses. No game data goes into git: the ROM is read where it is, and everything derived from it lands in the gitignored `build/`.

```sh
cp ~/roms/sm64.z64 .
make sm64-train               # train on the star sm64.ini names (pss-2), 8 games at once
make sm64-speedrun STAR=wf-3  # everything at once, for the fastest run to a star
make sm64-watch               # watch that star's latest checkpoint play, in a window
make sm64-replay              # watch the fastest run to it on file
```

The first sm64 build takes a few minutes to build N64Bundler, and about twenty seconds to recompile the game. Every build after that finds both done. [`envs/sm64/stars/README.md`](envs/sm64/stars/README.md) walks through giving a new star its long run.

### The other envs

```sh
make play                     # play the platformer yourself: A/D move, W/Space jump, S drop, R restart
make train                    # train it; checkpoints go to checkpoints/platformer/
make eval                     # watch the most trained checkpoint
make train ENV=bloons         # any env in envs/
```

Pass trainer flags with `ARGS`, e.g. `make train ARGS="--train.total-timesteps 10_000_000"`.

## How this repo uses puff-rs

puff-rs is a library: an `Env` trait, a vecenv that steps envs on threads, and the trainer on wgpu. A program adds its envs to an `Envs` by name and hands that to `pufferl::cli::main`. Here every env is a PufferLib 5.0 C env (`puf_init`, `puf_step`, `struct Log` and the rest, as in PufferLib's ocean), so `src/main.rs` adds none by name. Instead it gives a fallback, `src/cenv.rs`, which finds `envs/<name>/<name>.h` and its ini, and loads `build/cenv_<name>.dylib` (`.so` on Linux). `build.sh` builds that by compiling `cenv.c` around the env: a C ABI that makes, binds, steps, logs and closes one env at a time. Threads, buffers and log averaging are puff-rs's.

A Rust env would be a crate implementing `pufferl::env::Env` and one `.add` line in `src/main.rs`, as in puff-rs's `examples/squared`.

**A private env.** Copy this repo into a private one and add the env there. It depends on puff-rs by its public git URL like this one does, so nothing about the env has to leave the private repo.

**Working on puff-rs at the same time.** `cp .cargo/config.toml.example .cargo/config.toml` builds against a puff-rs checkout beside this one instead of GitHub's. That file is gitignored.

## Layout

```
src/main.rs           pufferl with this repo's envs
src/cenv.rs, cenv.c   the C env bridge: loader, and the C ABI build.sh compiles around an env
build.sh              builds an env into build/cenv_<env>.dylib (.so on Linux)
envs/sm64/            sm64.h (the env), n64b_gym.h (the game process), sm64.c (sm64_tool),
                      sm64.ini, stars/<star>.ini
envs/bloons/          bloons.h (the env, around the bloons repo's native/btd3-env), bloons.ini
envs/platformer/      platformer.h (the env), platformer.c (human play), platformer.ini
envs/sort/, envs/kronecker_factorization/
bevy/                 puffer-bevy: envs with Bevy + Avian physics (BevyEnv, Arenas),
                      CartPole and Pusher, bevy-envs (pufferl for them), bevy-bench
demos/peach-slide/    the slide's record runs: inputs, route traces, videos, notes
demos/wild-blue/      Shoot into the Wild Blue's record run
PLAN-text-agent.md    a plan for a text-prompted agent, frames in and actions out
vendor/PufferLib      submodule, branch 5.0: the C env headers (pufferenv.h, ini.h)
vendor/n64bundler     submodule
```

`build/`, `target/`, `checkpoints/`, `logs/` and ROMs are gitignored.

## Super Mario 64

The game is not emulated and not reimplemented. N64Bundler recompiles the cartridge into native code, and `n64b-run` runs it. The env starts one of those per agent, in a process of its own, and reads Mario out of the console's memory. That memory is mapped into both processes, so an observation is a memory load rather than a request. Three things N64Bundler provides make this usable for training:

- **The game runs off the clock.** It advances a frame only when the env asks, as fast as the machine manages. Headless, that's about 5,000 frames a second per game.
- **Steps are deterministic.** A frame is over when every thread of the game is parked waiting on the hardware. That is checked rather than waited out, so the same inputs from the same state give the same Mario to the bit.
- **Savestates.** The console's memory plus the registers of each of the game's threads, 8 MB. Every episode starts by loading one.

### Picking a star

A star is `<course>-<number>`, numbered the way the game's act select does: `wf-1` is Whomp's Fortress's first, `bob-7` is Bob-omb Battlefield's hundred coins, and `pss-2` is Peach's Secret Slide's second. A course alone (`bob`) means any star in it. `./build/sm64_tool courses` lists all 24 courses.

`star` in `[env]` names the goal (`sm64.ini` says `pss-2`), and every `make` target takes `STAR=`. A star that needs settings of its own keeps them in `envs/sm64/stars/<star>.ini`, which the Makefile passes as `--config` when the file exists. The slide gets a 1,500-frame clock and respawning, for instance. Any other star needs no file.

The first time a star is asked for, the env plays through the intro, points the castle's front door at the course's entry, walks Mario in, picks the star's act, and saves the game once he answers the pad. This works from nothing for every one of the 24 courses. `make sm64-state` remakes it.

### When a star is won, and which

The episode ends the frame Mario *touches* the star. A star that a box, a boss or a timer awards is spawned first, with about a hundred frames of cutscene, and a policy that stops at the spawn hasn't learned to take it.

Which star counts is its own bit in the save file's star flags. Any other star isn't the goal. That settled something about the slide: every slide run on file takes its *second* star, the one for reaching the bottom inside 21 seconds of the slide's own timer, and not the box. A run slower than that gets no star at all, however clean its ending.

### Actions and observations

- **Actions:** four discrete heads. A stick direction, which is none or one of sixteen *relative to the way Mario is facing*, and A, B and Z held or not. The game adds the camera's yaw to the stick, so a policy pushing pad-relative "up" would run wherever the camera drifted. The env measures that offset each frame and corrects for it.
- **Observations:** 290 floats. Everything about Mario that can be read out of memory: position, velocity, facing, spin, slide velocity, the stick as the game read it, the camera, the floor, wall and ceiling (their normals turned into his frame), his action and the one before (one-hot), timers, input and Mario flags, health.
- **Floor probes:** the height of the floor at 8 directions round him and 4 distances (150 to 2,000 units), found by walking the course's collision in RAM the way the game's own `find_floor` does. `SM64_TRACE_FLOORS=1` checks them against the game during a replay. They match it to a tenth of a unit, and they see the slide's 6,750-unit drop before the jump.
- **The star:** with `star_x`, `star_y`, `star_z` set (in `stars/<star>.ini`), its offset, distance and direction relative to his facing. A star with no place set prints where Mario touched it the first time a run takes it.
- **The picture** (optional). `picture_width` and `picture_height` append the game's frame, shrunk to that size, from the console's video interface. `state = 0` zeroes the numbers except the clock. Drawing costs half to three quarters of the throughput. `make sm64-picture` trains with an 80×60 one.
- One agent step is two frames (`frameskip`), so fifteen decisions a second.

### The reward

- **The star** pays 1 and ends the episode.
- **The clock.** Every frame costs `time_penalty / max_ticks`, so sooner is better, and running out of time costs `time_penalty` in all.
- **Progress** (with the star's place set). Each new closest approach to the star pays `progress` times the distance gained over the whole way from the start. With `progress_hurry`, the pay shrinks as the clock runs, so the same ground covered sooner pays more.
- **Novelty.** The course is cut into cubes 500 units a side. The first time in an episode Mario enters one, he's paid `novelty_episode + novelty / sqrt(n)`, where `n` counts the episodes that have entered it. A cube with no floor under him isn't a place and pays nothing.
- **Deaths and warps** out of the course charge whatever is left of the clock at once, so dying is never a way to stop it early. With `respawn = 1` a death puts the game back at the savestate and the episode carries on with the clock where it was.

Without a place for the star, nothing pays for getting nearer it, and the policy learns about the star by taking it.

In the log, `perf` is the fraction of episodes that take the star, and `frames` is what's being minimized. Both include episodes that began partway there, from the archive. `start_perf / from_start` is the star rate from the real start, which is the one that counts.

### Go-Explore

Novelty alone wasn't enough, so the env also runs [Go-Explore](https://arxiv.org/abs/1901.10995).

- **The archive.** For every cell any episode has entered, it keeps the shortest run of pad inputs that reached it. The game is deterministic, so the inputs *are* the place, at a few kilobytes against a savestate's 8 MB. A `go_explore` share of episodes pick a cell, weighted toward rarely visited ones (and with `go_explore_star`, toward cells on a way to the star), replay its inputs, and only then start the clock.
- **Cells** are a 500-unit cube plus, with `cell_speed` and `cell_heading`, a speed bucket, airborne or not, and one of eight headings. That way two ways through one place are kept apart.
- **The frontier** (`go_explore_near`, with the star's place set). This share of starts comes from the archive cells most on pace for the star. A cell's pace is how early the archive reached it, as a share of the clock, plus how far it still is from the star, as a share of the way from the start. Starts are drawn from the cells within 0.15 of the best.
- **The way along** (`frontier_way = 1`). "How far it still is" is measured along the steps episodes have actually taken, not as the crow flies. Every step from one 500-unit cube into a neighbouring one is kept, and a shortest-path search back from the cube nearest the star gives each cube its way left. On the slide, straight-line distance doesn't improve at all from frame 240 to 450 of the 674 run, so it ranked the whole lower track behind the upper one. `reach` in the log is the least way left of any place the archive gets to inside `max_ticks`, and `pace` is the best pace.
- **`explore_ticks`** gives archive starts that aren't from the frontier a clock of their own, starting at 0, so they explore while the starts from the top and from the frontier race on `max_ticks`. `racing` in the log is the share that race.
- **`go_explore_frames`** is the longest way the archive keeps, in episodes' worth of frames. The default is 3; for a speedrun it's 1, since a place reached later than the clock allows isn't on any run inside it.
- **`go_explore_clock = 1`** starts an episode from the archive with the clock at the frames its way there took, not at 0. For a star with a time limit, set `max_ticks` to that limit, and the clock is the game's.
- **Seeding** (`go_explore_seed = 1`). The archive starts with the runs on file instead of empty, so what separate runs found can meet.
- **The ghost** (`ghost`). Entering a cube on a way to the star sooner than the archive's way into it, on the ground, pays per frame of the difference. The archive then keeps the new way, so only a faster run is ever paid.
- **Phase 1 without a policy.** `make sm64-explore` plays at random from the archive and keeps the fastest run to the star.
- **Phase 2, robustifying.** `make sm64-robustify` is the backward algorithm ([Salimans and Chen, 2018](https://arxiv.org/abs/1812.03381)). A `backward` share of episodes start on the fastest run on file, at a frontier that moves back from the star as the policy learns to take it from there. Novelty and the archive are off, so only the star and the clock pay. Frontier starts are savestates, saved the first time an episode reaches them. Any run that beats the demo becomes the demo.

### Commands

Every target takes `STAR=<star>` and passes `ARGS` through.

| target | what it does |
|---|---|
| `make sm64-train` | train on the star, the clock and novelty, with the archive |
| `make sm64-explore` | Go-Explore phase 1 with no policy (`GAMES=8 SECONDS=600`) |
| `make sm64-robustify` | Go-Explore phase 2, working back along the fastest run on file |
| `make sm64-speedrun` | everything at once for the fastest run: fine cells, the frontier, the ghost, progress, a capped archive |
| `make sm64-picture` | train with an 80×60 picture of the game in the observation |
| `make sm64-watch` | watch the star's most trained checkpoint, in a window, from the real start |
| `make sm64-replay` | watch the fastest run on file |
| `make sm64-demo` | no policy at all: hold forward and jump, in a window |
| `make sm64-bench` | agent steps a second with `GAMES` games |
| `make sm64-state` | remake the star's savestate |
| `make sm64` | just build the env and `sm64_tool` |

`./build/sm64_tool` is the env without the trainer. It reads the same config and takes the same flags, so `./build/sm64_tool explore 8 600 --env.star wf-1` explores with the settings training would use.

```sh
./build/sm64_tool courses                       # the courses, and the names their stars go by
./build/sm64_tool replay [watch] [file.demo]    # play a run back and check it still takes the star
./build/sm64_tool record file.demo out.mp4      # write a run as a video (needs ffmpeg)
./build/sm64_tool trim [file...]                # cut runs where the star is taken
./build/sm64_tool probe [forward|random]        # step a game headless and print what Mario is doing
SM64_TRACE=30 ./build/sm64_tool replay ...      # print where Mario is every thirty frames
```

### What a star keeps, and where

Everything is per star. `build/sm64/<star>/` holds:

- `start.state`: the savestate every episode starts from
- `fastest.demo`: the fastest run on file
- `fastest/`: the ten fastest
- `seeds/`: runs kept by hand to seed the archive
- `states-<hash>/`: the frontier savestates of robustifying

Checkpoints and logs go to `checkpoints/<star>/` and `logs/<star>/`, so `latest` is that star's policy.

Any run that takes the star faster than the demo on file overwrites it, test runs included. **Set `SM64_DATA` to a scratch folder for anything that shouldn't touch the real files.** `SM64_HOST`, `SM64_MODULE`, `SM64_ROM`, `SM64_CONFIG_DIR` and `SM64_LOG` override the other paths `build.sh` wrote down.

### Platform notes

- **Throughput** on an M4 Pro, headless: 1 game runs 2,900 agent steps a second, 8 run 9,200, and 16 run 8,500. Each game keeps more than a core busy, so it peaks around eight. Keep `num_threads` equal to `total_agents`: a thread stepping a game is waiting on a socket, not working.
- **Linux** builds the game host with no renderer by default, so headless training needs no GPU or graphics driver. The picture and the watch targets need the renderer: build with `SM64_RENDERER=1`, which wants `libvulkan-dev`. On a Mac the renderer is built by default.
- **raylib** releases no build for arm64 Linux, so `build.sh` builds it from source there. `RAYLIB` names one that's already somewhere, and `N64BUNDLER` names another N64Bundler checkout.
- **In a container**, give it more shared memory than Docker's default 64 MB (`--shm-size=1g`). Each game shares about 9 MB with the env.
- The env is POSIX (`shm_open`, `posix_spawn`, a socket pair), so macOS and Linux only.


On an M4 Pro the trainer itself is far from the bottleneck: eight games train at roughly 3,500 to 5,000 steps a second, and a step of acting costs 0.22 ms against the 0.18 ms a GPU round trip costs with nothing to do.

## Status: Peach's Secret Slide

`pss-2` is the star every long run here has been after. The slide drops Mario 6,000 units, and the star appears at the bottom if he gets there inside 21 seconds.

| frames | counted at | what found it |
|---|---|---|
| 1,326 | touch | random play from the archive: the first star |
| 936 | spawn | robustifying: slides the whole track, then a long jump into the box |
| 648 | spawn | training with the archive: jumps off the upper track onto the lower one, skipping the middle turns |
| 786 | touch | the 648's route carried through to the touch, which is the goal from here on |
| 770, 768 | touch | a better jump off the upper track, then an archive seeded from the runs on file |
| 750 | touch | the ghost |
| **674** | touch | the long flight: backwards through the last upper-track turn, then a 6,750-unit fall |

The spawn comes about a hundred frames of cutscene before the touch, so the 648 and the 786 are the same run. Later unattended runs have only shaved that to 662, which isn't checked in.

**What's unsolved.** No policy has ever taken the star from the top of the slide in one episode. Every record is a run the archive assembled from starts partway down. Robustifying learns the last few seconds of a demo and then stalls, because the ending only works from the state the demo arrived in. And because `pss-2` pays nothing unless the slide is already fast, an episode from the top has no reward to learn from.

**From a blank slate, with everything (2026-10-14).** `make sm64-speedrun STAR=pss-2` ran for an hour: 10.2M steps, no demos or seeds, with the 290-number observation, floor probes, the star's place, progress in a hurry, speed and heading in the archive's cells, the frontier and the ghost. **It took no star.** The nearest single episodes got within 350 to 500 units of the star's spot, but not inside 21 seconds, so nothing spawned. Three things went wrong, in order:

1. Cells as fine as speed and heading make the archive 20 to 26 thousand cells. Drawing starts by rarity then draws almost evenly over the whole course. The first attempt spent 20 minutes all over it, and its closest approaches got further away. The frontier (`go_explore_near`) fixed that: the average closest approach went from 7,800 units to 1,700 within five minutes.
2. The ways to the bottom it kept ran to 2,700 frames, far outside the 21 seconds the star wants, and replaying them halved the steps a second. Hence `go_explore_frames`.
3. With both fixes, the policy learned to be careful, not fast. Top speed fell from 95 units a frame to 45 to 50 over the hour, and deaths an episode from 1.6 to 0.4. With `respawn` on, a fall gives up whatever progress the rest of the episode would have paid, so slowing down to stay on the track is what that reward pays for. On `pss-2`, careful means no star.

What would most likely change it: pay progress only inside the time the star allows, or make the clock and the hurry bigger than what a fall costs; or seed the archive with one run to the star (`go_explore_seed`), so the ghost has something to race from the first minute. The archive's cells, the frontier and the star's place all carry over to other stars as they are.

**The star's own clock (2026-10-14, second hour).** Following that, `pss-2` got `max_ticks = 750` (the 674 run plus slack), `respawn = 0`, `go_explore_clock = 1`, and the frontier drawn by pace instead of by distance alone. **Again no star**, in 4.6M steps; throughput halved, because most episodes now end in a fall after about 150 steps and every reset reloads the savestate and replays a way. The caution went away at first: average forward speed was 40 units a frame in the first twenty minutes, against 10 to 27 in the hour before. But it slid back to 15 by the end, top speed fell from 76 to 44, and no epoch's average closest approach got under 4,600 units, where the hour before had epochs under 500. With every way the archive keeps limited to 750 frames, it can no longer build a way down the slide in stages; a blank slate has to find a route inside 21 seconds in one go. What it probably needs is the two kept apart: the archive exploring with the whole clock, and only the frontier's starts held to the star's.

**Exploring apart from racing (2026-10-15, two hours).** `explore_ticks = 1500` does that: episodes from the top and from the frontier race on the star's 750 frames (62% of them), and every other archive start explores on 1,500 frames of its own, with the archive keeping ways that long. **No star in 12.2M steps.** It got nearer than the hour before: in the first half hour some log intervals averaged 211 and 378 units from the star, and later ones 1,000 to 2,000, against no better than 4,600. But the same slide set in as in both earlier runs: average forward speed went from 38 units a frame in the first fifteen minutes to 15 in the last, top speed from 74 to 38, and the progress paid an episode from 0.10 to 0.04. With no star ever taken, the only pay for speed is progress, which comes to about a tenth of a point an episode, and the policy drifts away from it.

**The way along, and the star (2026-10-15, two hours).** Two things were wrong with exploring. The archive's cubes stopped at a height of -4,096, and the bottom of the slide, star included, is below that: Go-Explore had never kept a single place at the bottom. And the frontier and progress both measured distance to the star as the crow flies, which ranks the whole lower track behind the upper one. With the cubes down to -8,192 and both measured along the steps taken (`frontier_way`, `progress_way`), **the first star from a blank slate came 26 minutes in**. The fastest went 782 → 742 → 732 → 726 → **718** frames (checked by replay), and from the second half hour on, about half of all episodes took the star. Speed no longer fades once the star pays: average forward speed went 39 → 17 → 41 over the run, and top speed back up to 72. None of the stars came from the top of the slide (`start_perf` 0): every one started partway down, from the archive. That is the next thing to fix. [`demos/peach-slide/README.md`](demos/peach-slide/README.md) describes each record run, and [`NEXT.md`](demos/peach-slide/NEXT.md) beside it has the exact command for the next run and where the remaining frames probably are.

## Findings worth keeping

Most of this was learned on the env's first goal, opening the castle's front door from the grounds. That goal has since been removed, but the reward is shaped the way it is because of it.

- **A goal and a clock alone teach nothing.** 12M steps never opened the door: every episode was worth exactly −1 and entropy sat at its maximum. Novelty is the memory of where Mario has been.
- **Novelty needs both parts.** The fading part alone found the door by 100K steps and lost it by 500K, once the way there was worn out. The per-episode part, which never fades, keeps covering ground worth doing.
- **Novelty still isn't enough.** It swam the moat for 10M steps. The archive is what gets episodes past a long stretch with nothing new in it.
- **Don't pay for falling.** A fall through empty space enters a fresh cube every 500 units. Novelty was paying for it and the archive was restarting episodes mid-fall, so a cube with no floor under Mario is not a place.
- **Deaths in a dangerous course.** In Whomp's Fortress 70% of episodes ended in an early death. `respawn = 1` took episodes from 12 cubes to 60 without making dying cheap.
- **The clock has to fit the task.** The slide is about 1,100 frames at an explorer's pace, so no episode could fit it into the default 900.
- **Muon random-walks without a signal.** Its updates are the same size however weak the gradient is, and the first sparse-reward run blew up at 3.4M steps. `weight_decay = 0.01` holds the weights.
- **No fixed `ent_coef` worked.** At 0.05 the policy stayed uniform, and at 0.005 it collapsed. Hence `target_entropy`, and `ent_coef_damping` once runs score from the first minute.
- **Advantage normalization is what sm64 needs from 4.0.** 5.0 dropped it, and sm64's rewards are small. Without it, robustifying the slide learned nothing in 1.5M steps and entropy drifted *up*. `norm_adv = 1` brings it back, and `make sm64-robustify` sets it.
- **Robustifying wants a gentle learning rate.** At PufferLib's 0.015, approximate KL hit 0.95 in one epoch and the success rate went to zero. At 0.005 it stays near 0.001.
- **A frontier should be a savestate.** Replaying 1,200 frames of demo to start a 330-frame episode ran training at 1,000 steps a second. Loading the game there runs it at 5,000.
- **The stick has to be sent the way the game reads it.** The game takes `atan2s(-stickY, stickX)`, so the stick goes out as `(sin θ, −cos θ)`. The mirror image can't be corrected by any measured camera offset.
- **The picture** was worth nothing measurable in Whomp's Fortress. On the slide it covered about twice the ground per step, where the one thing to know is where the floor runs out, which roughly pays for the half of the throughput it costs.

The full experiment log that this README condenses, with every run, table and dead end, is [puff-rs's README as of commit `2069328`](https://github.com/bigmah/puff-rs/blob/2069328/README.md), from before the envs moved here.

## Bloons TD 3

The first track on easy, in the native Rust port of the game ([bigmah/bloons](https://github.com/bigmah/bloons)), checked out beside this one or wherever `BLOONS` says. `build.sh` builds its simulation into a static library and links it into the env. Each step is a build, an upgrade or a wait on a 20 px grid, then half a second of game; rounds run back to back into freeplay, and `score` is the round the game ends on.

```sh
make train ENV=bloons
make eval ENV=bloons
```

## Bevy envs

`bevy/` is a crate of envs whose physics is [Bevy](https://bevy.org) with [Avian](https://github.com/avianphysics/avian): collisions, joints and friction come from the engine. An env implements `BevyEnv`: `spawn` builds one arena (bodies, colliders, joints) once, `reset` puts it where an episode starts (`teleport`), `act` applies an action, `observe` and `outcome` read it back. `Arenas` puts `[env] arenas` of them, 20 m apart, in one headless Bevy world stepped one physics tick a step, and is one pufferl env with an agent per arena; the vecenv steps a world per thread.

- **bevy_cartpole:** gym's CartPole on a slider joint and a hinge. About 450 of 500 steps in 20M steps, 7 s.
- **bevy_pusher:** top-down, a round agent shoves a box onto a goal in a walled room. About 93% solved in 100M steps, under a minute, on two seeds of three (seed 73, the default, stays at random's 11%).

```sh
make bevy-train BEVY_ENV=bevy_pusher
make bevy-eval BEVY_ENV=bevy_pusher    # watch it in a window: [env] num_agents arenas, view_speed x real time
make bevy-bench
```

The window (`bevy/src/view.rs`) draws every body from its collider, so a new env shows up with no drawing code; things that aren't bodies (a goal, a track) an env adds through `marks`. Bevy has the main thread and eval runs behind it; closing the window ends eval.

Bevy 0.20 needs rustc 1.97 or newer, so the targets build with `cargo +1.99`; the rest of the repo still builds on stable. The crate is a workspace member of its own, so `cargo build --release` at the top does not compile Bevy.

Speed on an M4 Pro (14 cores), physics alone (`make bevy-bench`), with 512 arenas a world: CartPole 1.2M steps/s a thread, 12M on all 14; Pusher 2.6M a thread, 26M on all 14. Training, the trainer's own time included: CartPole 3.1M steps/s, Pusher 1.9M. A world with a single arena manages only about 70K steps/s, so many arenas a world is what makes it fast.

Don't despawn and respawn bodies between episodes: a world doing that slowed by half in 30 seconds (Avian keeps something per despawned body), where moving them back in place holds steady and is faster.

## The platformer

One level: run right, clear the pits and spikes, touch the flag.

- **Actions:** two discrete heads. Move is none, left or right, and the other is none, jump, or drop through a one-way platform.
- **Observations:** a 15×9 window of tiles around the player in three channels (solid, one-way, hazard), plus 8 numbers of player state.
- **Reward:** progress toward the flag (1.0 over the whole run), +1 for the flag, and −0.1 for dying or running out of time. Episodes end at the flag, on death, or after 30 seconds.

Keep the failure penalty small. At −0.5, early deaths taught some seeds to stand still.

