//! A PufferLib 5.0 C env as an [`Env`], through build/cenv_ENV.dylib (.so, .dll), which
//! build.sh compiles around the env (cenv.c). The library makes, binds, steps and closes
//! one env at a time; everything else, threads and buffers and logs, is pufferl's vecenv's.

use std::ffi::{c_char, c_double, c_float, c_int, c_void, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use libloading::Library;
use pufferl::env::{Agents, Config, Env, Log, Obs};
use pufferl::{Entry, VecEnv};

use crate::fail;

/// PUF_DICT_MAX_KEY
const KEY_BYTES: usize = 128;
const MAX_LOG_KEYS: usize = 64;

type EnvPtr = *mut c_void;

/// The library's functions, looked up once
struct Api {
    num_agents: unsafe extern "C" fn(EnvPtr) -> c_int,
    new: unsafe extern "C" fn(*mut c_void, c_int) -> EnvPtr,
    bind: unsafe extern "C" fn(EnvPtr, *mut c_void, *const c_float, *mut c_float, *mut c_float, *mut u8, c_int),
    reset: unsafe extern "C" fn(EnvPtr),
    step: unsafe extern "C" fn(EnvPtr),
    log_take: unsafe extern "C" fn(EnvPtr, *mut c_float, c_int),
    log_names: unsafe extern "C" fn(*const c_float, c_int, c_int, *mut c_char, *mut c_double) -> c_int,
    render: unsafe extern "C" fn(EnvPtr),
    window_closed: unsafe extern "C" fn() -> c_int,
    close: unsafe extern "C" fn(EnvPtr),
    config_free: unsafe extern "C" fn(*mut c_void),
    obs_size: unsafe extern "C" fn() -> c_int,
    act_sizes: Vec<usize>,
    obs_bytes: usize,
    log_fields: usize,
    /// The Dict every env read its [env] settings from, which lives as long as they do
    config: *mut c_void,
    _lib: Library,
}

// Envs are made, stepped and closed on whichever thread the vecenv likes, as OpenMP did
// with them in PufferLib's vecenv; each touches only its own Env.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Drop for Api {
    fn drop(&mut self) {
        unsafe { (self.config_free)(self.config) }
    }
}

/// Open the env's library for good. An env may start threads of its own that outlive
/// every env it made, and unloading the library under them is a crash at exit.
/// RTLD_NODELETE keeps it mapped until the process ends, which is when its threads do. The
/// rest are libloading's own flags.
#[cfg(unix)]
unsafe fn open(path: &Path) -> Result<Library, libloading::Error> {
    let flags = libc::RTLD_LAZY | libc::RTLD_LOCAL | libc::RTLD_NODELETE;
    libloading::os::unix::Library::open(Some(path), flags).map(Library::from)
}

#[cfg(not(unix))]
unsafe fn open(path: &Path) -> Result<Library, libloading::Error> {
    Library::new(path)
}

pub fn library_name(env_name: &str) -> String {
    let extension = if cfg!(target_os = "macos") { "dylib" } else if cfg!(windows) { "dll" } else { "so" };
    format!("cenv_{env_name}.{extension}")
}

impl Api {
    fn open(root: &Path, env_name: &str, config: &Config) -> Self {
        let path = root.join("build").join(library_name(env_name));
        if !path.exists() {
            fail(&format!("{} not found: run ./build.sh {env_name}", path.display()));
        }
        // The library is an env someone compiled to be trained on; loading it runs it
        let lib = unsafe { open(&path) }.unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
        macro_rules! get {
            ($name:literal) => {
                *unsafe { lib.get($name) }.unwrap_or_else(|e| fail(&format!("{}: {e}: rebuild it with ./build.sh {env_name}", path.display())))
            };
        }
        let num_atns: unsafe extern "C" fn() -> c_int = get!(b"cenv_num_atns");
        let act_sizes: unsafe extern "C" fn() -> *const c_int = get!(b"cenv_act_sizes");
        let obs_bytes: unsafe extern "C" fn() -> c_int = get!(b"cenv_obs_bytes");
        let log_fields: unsafe extern "C" fn() -> c_int = get!(b"cenv_log_fields");
        let make_config: unsafe extern "C" fn(c_int, *const *const c_char, *const *const c_char) -> *mut c_void = get!(b"cenv_config");

        let keys: Vec<CString> = config.pairs().iter().map(|(k, _)| CString::new(k.as_str()).unwrap()).collect();
        let values: Vec<CString> = config.pairs().iter().map(|(_, v)| CString::new(v.as_str()).unwrap()).collect();
        let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
        let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();

        let act_sizes = unsafe { std::slice::from_raw_parts(act_sizes(), num_atns() as usize) }.iter().map(|&a| a as usize).collect();
        let (obs_bytes, log_fields) = unsafe { (obs_bytes() as usize, log_fields() as usize) };
        let config = unsafe { make_config(keys.len() as c_int, key_ptrs.as_ptr(), value_ptrs.as_ptr()) };
        Self {
            num_agents: get!(b"cenv_num_agents"),
            new: get!(b"cenv_new"),
            bind: get!(b"cenv_bind"),
            reset: get!(b"cenv_reset"),
            step: get!(b"cenv_step"),
            log_take: get!(b"cenv_log_take"),
            log_names: get!(b"cenv_log_names"),
            render: get!(b"cenv_render"),
            window_closed: get!(b"cenv_window_closed"),
            close: get!(b"cenv_close"),
            config_free: get!(b"cenv_config_free"),
            obs_size: get!(b"cenv_obs_size"),
            act_sizes,
            obs_bytes,
            log_fields,
            config,
            _lib: lib,
        }
    }
}

/// The C env called name: envs/NAME/NAME.h, with its config beside it in NAME.ini, built
/// into build/cenv_NAME. None if there is no such env.
pub fn find(root: &Path, name: &str) -> Option<Entry> {
    let dir = root.join("envs").join(name);
    if !dir.join(format!("{name}.h")).exists() {
        return None;
    }
    let ini = dir.join(format!("{name}.ini"));
    let config = std::fs::read_to_string(&ini).unwrap_or_else(|e| fail(&format!("{}: {e}", ini.display())));
    let (root, name) = (root.to_path_buf(), name.to_string());
    Some(Entry::new(config, ini.display().to_string(), move |config, total_agents, num_threads| {
        load(&root, &name, config, total_agents, num_threads)
    }))
}

/// The C env called env_name in a vecenv
pub fn load(root: &Path, env_name: &str, config: &Config, total_agents: usize, num_threads: usize) -> VecEnv {
    let api = Arc::new(Api::open(root, env_name, config));
    match api.obs_bytes {
        1 => VecEnv::from_envs(|i| CEnv::<u8>::new(api.clone(), i), total_agents, num_threads),
        4 => VecEnv::from_envs(|i| CEnv::<f32>::new(api.clone(), i), total_agents, num_threads),
        n => fail(&format!("{env_name}: observations of {n}-byte elements; pufferl reads unsigned char or float")),
    }
}

pub struct CEnv<O> {
    api: Arc<Api>,
    env: EnvPtr,
    num_agents: usize,
    /// Never written: a C env keeps its Log in its own struct, which summarize reads
    log: Log,
    _obs: PhantomData<O>,
}

unsafe impl<O> Send for CEnv<O> {}

impl<O> CEnv<O> {
    fn new(api: Arc<Api>, index: usize) -> Self {
        let env = unsafe { (api.new)(api.config, index as c_int) };
        let num_agents = unsafe { (api.num_agents)(env) } as usize;
        Self { api, env, num_agents, log: Log::default(), _obs: PhantomData }
    }

    fn bind(&mut self, agents: &mut Agents<'_, O>) {
        unsafe {
            (self.api.bind)(
                self.env,
                agents.observations.as_mut_ptr() as *mut c_void,
                agents.actions.as_ptr(),
                agents.rewards.as_mut_ptr(),
                agents.terminals.as_mut_ptr(),
                agents.action_mask.as_mut_ptr(),
                agents.mask_size as c_int,
            );
        }
    }
}

impl<O: Obs> Env for CEnv<O> {
    type Obs = O;

    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn obs_size(&self) -> usize {
        unsafe { (self.api.obs_size)() as usize }
    }

    fn act_sizes(&self) -> Vec<usize> {
        self.api.act_sizes.clone()
    }

    // The env keeps pointers into the buffers, so they are pointed again before every
    // call, at the slices this call has
    fn reset(&mut self, mut agents: Agents<'_, O>) {
        self.bind(&mut agents);
        unsafe { (self.api.reset)(self.env) }
    }

    fn step(&mut self, mut agents: Agents<'_, O>) {
        self.bind(&mut agents);
        unsafe { (self.api.step)(self.env) }
    }

    fn log(&mut self) -> &mut Log {
        &mut self.log
    }

    /// As PufferLib's vecenv does it: the Log structs of the envs that have finished an
    /// episode added up field by field, divided by n, and named by the env's puf_log
    fn summarize(envs: &mut [Self], clear: bool) -> Vec<(String, f64)> {
        let Some(first) = envs.first() else {
            return vec![("n".to_string(), 0.0)];
        };
        let api = first.api.clone();
        let fields = api.log_fields;
        let mut sums = vec![0f64; fields];
        let mut log = vec![0f32; fields];
        for env in envs.iter() {
            unsafe { (api.log_take)(env.env, log.as_mut_ptr(), clear as c_int) };
            if log[fields - 1] != 0.0 {
                for (sum, &field) in sums.iter_mut().zip(&log) {
                    *sum += field as f64;
                }
            }
        }
        let n = sums[fields - 1];
        let mut out: Vec<(String, f64)> = Vec::new();
        if n > 0.0 {
            let averaged: Vec<f32> = sums.iter().map(|&sum| (sum / n) as f32).collect();
            let mut keys = vec![0u8; MAX_LOG_KEYS * KEY_BYTES];
            let mut values = [0f64; MAX_LOG_KEYS];
            let count = unsafe {
                (api.log_names)(averaged.as_ptr(), MAX_LOG_KEYS as c_int, KEY_BYTES as c_int, keys.as_mut_ptr() as *mut c_char, values.as_mut_ptr())
            } as usize;
            for i in 0..count {
                let key = &keys[i * KEY_BYTES..(i + 1) * KEY_BYTES];
                let end = key.iter().position(|&b| b == 0).unwrap_or(KEY_BYTES);
                out.push((String::from_utf8_lossy(&key[..end]).into_owned(), values[i]));
            }
        }
        match out.iter_mut().find(|(key, _)| key == "n") {
            Some(entry) => entry.1 = n,
            None => out.push(("n".to_string(), n)),
        }
        out
    }

    fn render(&mut self) {
        unsafe { (self.api.render)(self.env) }
    }

    fn window_closed(&self) -> bool {
        unsafe { (self.api.window_closed)() != 0 }
    }
}

impl<O> Drop for CEnv<O> {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.env) }
    }
}
