//! pufferl for the envs in this repo. Each is a PufferLib 5.0 C env in envs/<name>/, which
//! build.sh compiles into build/cenv_<name> and src/cenv.rs loads; the trainer and the
//! vecenv are puff-rs's (github.com/bigmah/puff-rs).
//!
//!     ./target/release/pufferl train sm64 [--config FILE ...] [--section.key=value ...]
//!
//! A Rust env would be an .add line here, as in puff-rs's examples/squared.

mod cenv;

use pufferl::{cli, Envs};

/// Say what is wrong and stop
pub fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

fn main() {
    cli::main(Envs::new().fallback(cenv::find));
}
