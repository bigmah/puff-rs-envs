# Shoot into the Wild Blue in 656 frames

`star-656.demo` is the fastest run on record here for Whomp's Fortress's third
star, Shoot into the Wild Blue (`wf-3`): the pad inputs that take Mario from the
course savestate (act 3) to the star in 656 frames, 21.9 seconds at the game's
thirty a second, in 328 inputs. It is kept here because the copy the tools use,
`build/sm64/wf-3/fastest.demo`, is overwritten by any run that finds a faster
star. `route-656.txt` is where Mario is every thirty frames on the way.

The star is the one act 3's level script places at (-2500, 1500, -750), on the
ledge on the far side of the fortress's outer wall; the act select numbers it
third (star index 2 in its behavior parameters), not fifth. No cannon: Mario
crosses to the foot of the wall by frame 420, spends about a hundred frames
standing there, then jumps up from the ground below the ledge and takes the star
from the air at frame 656.

## How it was found

`make sm64-speedrun STAR=wf-3`'s arguments (`envs/sm64/stars/wf-3.ini`), started
2026-10-17 at 20:52 with a new policy and an empty archive, for twelve hours:

- 20:52 to 01:18, 36.7M steps: no star. The trainer exited when one game's
  savestate would not load at a reset ("thread 1 is waiting somewhere else than
  it was in the state"); `puf_reset` exits on that rather than retrying.
- 01:18 to 08:14, 60M steps, from that run's last checkpoint with an empty
  archive: the first stars, and the touch rate went from 2.5% to 45% of
  episodes by the end. The fastest folder was all 656 to 660 frames by 06:21,
  and nothing faster came after.
- 08:15 to 08:52, 5.8M steps, from that checkpoint with the archive seeded
  from the runs on file: nothing faster.

As on the slide, none of the stars came from where the task starts:
`start_perf` was 0 throughout, and every one started from the archive.

## Playing it

```sh
./build/sm64_tool replay watch demos/wild-blue/star-656.demo --config envs/sm64/stars/wf-3.ini
./build/sm64_tool replay demos/wild-blue/star-656.demo --config envs/sm64/stars/wf-3.ini
SM64_TRACE=30 ./build/sm64_tool replay demos/wild-blue/star-656.demo --config envs/sm64/stars/wf-3.ini
```
