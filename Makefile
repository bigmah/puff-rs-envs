.PHONY: setup trainer build play train eval sm64 sm64-state sm64-train sm64-explore sm64-robustify sm64-speedrun sm64-picture sm64-watch sm64-replay sm64-demo sm64-bench bevy-train bevy-eval bevy-score bevy-bench bevy-test

# Which env to build and train: one in envs/. Each builds into a library of its own
# (build/cenv_<env>.dylib, or .so), so building one leaves the others as they are.
ENV ?= platformer

# puff-rs's trainer, with this repo's envs (src/main.rs). It runs on wgpu: Metal, Vulkan
# or DX12, whatever the machine has.
PUFFERL = ./target/release/pufferl

# One-time, and optional: fetch the submodules and build the trainer (cargo fetches puff-rs)
setup:
	git submodule update --init
	cargo build --release

trainer:
	cargo build --release

build: trainer
	./build.sh $(ENV)

# A/D move, W/Space jump, S drop through platforms, R restart, ESC quit
play: build
	./build/platformer

# Extra flags pass through, e.g. make train ARGS="--train.total-timesteps 10_000_000"
# On the GPU: checkpoints go to checkpoints/<env>/
train: build
	$(PUFFERL) train $(ENV) $(ARGS)

# Watch the most trained checkpoint play one env (ESC to quit)
eval: build
	$(PUFFERL) eval $(ENV) latest $(ARGS)

# --- Super Mario 64 ----------------------------------------------------------
# The game is your own sm64.z64, Super Mario 64 (USA), at the top of this checkout
# or wherever SM64_ROM says. The first build fetches N64Bundler (vendor/n64bundler),
# builds it and recompiles the game with it; N64BUNDLER names another checkout.
#
# Every target is about one star, STAR: <course>-<number> as the game numbers a
# course's stars (pss-2, wf-1, bob-7), or a course alone for any star in it.
# `./build/sm64_tool courses` lists the courses; without STAR it is the star
# sm64.ini names. A star with settings of its own -- a longer clock, respawning --
# has them in envs/sm64/stars/$(STAR).ini, which goes over sm64.ini; any other
# star needs no file. A star keeps its savestate, demos and seeds in
# build/sm64/$(STAR)/, and its checkpoints and logs in checkpoints/$(STAR)/ and
# logs/$(STAR)/, so `latest` is that star's most trained policy.
STAR ?= $(shell sed -n 's/^star *= *\([a-z0-9-]*\).*/\1/p' envs/sm64/sm64.ini)
STAR_INI = envs/sm64/stars/$(STAR).ini
STAR_CONFIG = $(if $(wildcard $(STAR_INI)),--config $(STAR_INI)) --env.star $(STAR)
STAR_RUN = $(STAR_CONFIG) --base.checkpoint-dir checkpoints/$(STAR) --base.log-dir logs/$(STAR)

sm64: trainer
	./build.sh sm64

# Get into the star's course and save the game there. Every episode starts from
# this. Made automatically the first time a star is trained; this remakes it.
sm64-state: sm64
	./build/sm64_tool state $(STAR_CONFIG)

# The star, the clock and novelty, explored with Go-Explore
sm64-train: sm64
	$(PUFFERL) train sm64 $(STAR_RUN) $(ARGS)

# Go-Explore's first phase with no policy: random play from the archive until it
# has the star, keeping the fastest run in build/sm64/$(STAR)/fastest.demo
sm64-explore: sm64
	./build/sm64_tool explore $(or $(GAMES),8) $(or $(SECONDS),600) $(STAR_CONFIG)

# Go-Explore's second phase: train a policy to get the star from the real start,
# working back along that run, with only the star and the clock paying. Every
# star it gets faster than the run becomes the run for the next time.
# Advantages are normalized (norm_adv), which is 4.0's and not 5.0's: without it
# the slide learned nothing in 1.5M steps. The discount is 0.999 rather than
# 5.0's 0.995, at which a star 400 frames off is worth a third. Entropy is bought
# at a fixed price rather than held at sm64.ini's target, and the learning rate
# is a third of PufferLib's 0.015: see the README.
sm64-robustify: sm64
	$(PUFFERL) train sm64 $(STAR_RUN) --env.backward 0.8 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 \
		--train.target-entropy 0 --train.ent-coef 0.005 --train.learning-rate 0.005 \
		--train.norm-adv 1 --train.gamma 0.999 --train.gae-lambda 0.95 --train.minibatch-size 1024 $(ARGS)

# Everything at once, for the fastest run to the star: Go-Explore with cells
# that tell speed and heading apart, most episodes started from the archive and
# most of those along the way to the star (or, before there is one, nearest
# it), no way kept that is longer than the clock, the ghost to race, and --
# with the star's place in its stars/$(STAR).ini -- progress paid in a hurry. The fastest
# runs land in build/sm64/$(STAR)/fastest/ as they are found. Advantages are
# normalized and entropy is held with damping: see NEXT.md in demos/peach-slide.
sm64-speedrun: sm64
	$(PUFFERL) train sm64 $(STAR_RUN) --env.go-explore 0.75 --env.go-explore-star 0.75 --env.go-explore-frames 1 --env.go-explore-clock 1 --env.ghost 0.02 \
		--train.norm-adv 1 --train.learning-rate 0.005 --train.ent-coef 0.007 --train.ent-coef-rate 0.0025 \
		--train.ent-coef-damping 0.5 $(ARGS)

# Train with the policy looking at the game: an 80 by 60 picture of it in the
# observation, through a small convolutional net. About a quarter of the speed.
# Watch one with: make sm64-watch ARGS="--env.picture-width 80 --env.picture-height 60"
sm64-picture: sm64
	$(PUFFERL) train sm64 $(STAR_RUN) --env.picture-width 80 --env.picture-height 60 $(ARGS)

# Watch the star's most trained checkpoint start where the task does and go for
# it, in a window, at the speed a console ran
sm64-watch: sm64
	$(PUFFERL) eval sm64 latest $(STAR_RUN) --env.window 1 --env.go-explore 0 \
		--env.novelty 0 --env.novelty-episode 0 $(ARGS)

# Watch the fastest run to the star on file
sm64-replay: sm64
	./build/sm64_tool replay watch $(STAR_CONFIG)

# The game with no policy at all: hold forward and jump, in a window
sm64-demo: sm64
	./build/sm64_tool watch forward $(STAR_CONFIG)

# How many agent steps a second this machine manages, with N games at once
sm64-bench: sm64
	./build/sm64_tool bench $(or $(GAMES),8) 400 $(STAR_CONFIG)

# --- Bevy envs (bevy/): Bevy + Avian physics, many arenas a world ----------------
# Bevy 0.20 needs rustc 1.97+: these build with the 1.99 toolchain, the rest on stable
BEVY_ENV ?= bevy_pusher
BEVY_CARGO = cargo +1.99

bevy-train:
	$(BEVY_CARGO) run --release -p bevy-envs --bin bevy-envs -- train $(BEVY_ENV) $(ARGS)

# Watch the most trained checkpoint play in a window ([env] num_agents arenas)
bevy-eval:
	$(BEVY_CARGO) run --release -p bevy-envs --bin bevy-envs -- eval $(BEVY_ENV) latest $(ARGS)

# The same checkpoint scored over many games, no window
bevy-score:
	$(BEVY_CARGO) run --release -p bevy-envs --bin bevy-envs -- eval $(BEVY_ENV) latest --headless $(ARGS)

bevy-bench:
	$(BEVY_CARGO) run --release -p bevy-envs --bin bevy-bench

bevy-test:
	$(BEVY_CARGO) test --release -p bevy-envs
