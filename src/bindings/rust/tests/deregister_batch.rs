// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Agent::deregister_memory_from_dlist` at the scale a caller like bdn uses
//! it: up to 256 regions of 8 MiB, each registered on its own, released in
//! one call. A file of its own so its pinned-memory checks see only its own
//! registrations: `VmPin` is per process, and these tests run one at a time.

use nixl_sys::*;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

/// One 8 MiB chunk of a resident blob per registration.
const CHUNK: usize = 8 << 20;
/// The most chunks bdn withdraws in one batch (`MAX_DEREGISTER_BATCH_ENTRIES`).
const BATCH: usize = 256;

/// Serializes the tests: each checks the process's pinned memory.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn agent_with_ucx(name: &str) -> (Agent, OptArgs) {
    let agent = Agent::new(name).expect("Failed to create agent");
    let (_mems, params) = agent
        .get_plugin_params("UCX")
        .expect("Failed to get UCX plugin params");
    let backend = agent
        .create_backend("UCX", &params)
        .expect("Failed to create UCX backend");
    let mut opt_args = OptArgs::new().expect("Failed to create opt args");
    opt_args
        .add_backend(&backend)
        .expect("Failed to add backend");
    (agent, opt_args)
}

fn opt_args_for(agent: &Agent) -> OptArgs {
    let mut opt_args = OptArgs::new().expect("Failed to create opt args");
    opt_args
        .add_backend(&agent.get_backend("UCX").expect("No UCX backend"))
        .expect("Failed to add backend");
    opt_args
}

/// A chunk of memory mapped on its own and unmapped on drop, as a caller
/// like bdn maps each region it registers. UCX keeps a region's memory
/// registration cached after deregistration until it sees the memory
/// unmapped, so a test that wants to see it released must unmap, which a
/// freed `Vec` need not (the allocator may keep it).
struct Chunk {
    base: *mut libc::c_void,
}

impl Chunk {
    fn new() -> Self {
        // SAFETY: a fresh private anonymous mapping, checked below.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                CHUNK,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "mmap failed");
        Self { base }
    }

    fn addr(&self) -> usize {
        self.base as usize
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        // SAFETY: unmaps exactly what `new` mapped, once.
        unsafe { libc::munmap(self.base, CHUNK) };
    }
}

// The mapping is only ever handed to NIXL by address.
unsafe impl Send for Chunk {}

fn chunks(count: usize) -> Vec<Chunk> {
    (0..count).map(|_| Chunk::new()).collect()
}

/// Register each chunk on its own, as a caller registering chunk by chunk
/// does, and return the handles in order.
fn register_each(agent: &Agent, opt_args: &OptArgs, chunks: &[Chunk]) -> Vec<RegistrationHandle> {
    chunks
        .iter()
        .flat_map(|chunk| {
            let mut descs = RegDescList::new(MemType::Dram).unwrap();
            descs.add_desc(chunk.addr(), CHUNK, 0);
            agent
                .register_memory_from_dlist(&descs, Some(opt_args))
                .expect("Failed to register a chunk")
        })
        .collect()
}

/// The pinned memory to expect once a test's registrations are gone: UCX
/// pins some of its own on an agent's first registration, so take one
/// register, deregister and unmap round trip first.
fn baseline_after_warm_up(agent: &Agent, opt_args: &OptArgs) -> Option<u64> {
    let warm = chunks(1);
    drop(register_each(agent, opt_args, &warm));
    drop(warm);
    // Let UCX's release of the warm-up chunk land before measuring.
    thread::sleep(Duration::from_millis(200));
    pinned_kib()
}

/// The process's pinned memory in KiB (`VmPin`), which an RDMA registration
/// raises and its release lowers. `None` where the kernel reports none.
fn pinned_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmPin:"))
        .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse().ok())
}

/// Assert the process's pinned memory comes back down to `baseline`. UCX
/// releases a deregistered region's memory registration when the memory is
/// unmapped, from its own thread, so this waits briefly for it.
fn assert_unpinned_to(baseline: Option<u64>, what: &str) {
    let Some(baseline) = baseline else {
        return;
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let pinned = pinned_kib().unwrap_or(0);
        if pinned <= baseline {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: {pinned} KiB still pinned against a baseline of {baseline} KiB"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_full_batch_of_8mib_chunks_deregisters_in_one_call() {
    let _serial = serial();
    let (agent, opt_args) = agent_with_ucx("dereg_batch_full");
    let initial_md = agent.get_local_md().unwrap().len();
    let baseline = baseline_after_warm_up(&agent, &opt_args);

    let buffers = chunks(BATCH);
    let started = Instant::now();
    let handles = register_each(&agent, &opt_args, &buffers);
    let registered = started.elapsed();
    assert_eq!(handles.len(), BATCH);
    assert!(agent.get_local_md().unwrap().len() > initial_md);
    let pinned = pinned_kib();

    let started = Instant::now();
    agent
        .deregister_memory_from_dlist(handles, Some(&opt_args))
        .expect("Failed to deregister a full batch in one call");
    let deregistered = started.elapsed();
    assert_eq!(agent.get_local_md().unwrap().len(), initial_md);

    // Every chunk is gone, so each registers again.
    drop(register_each(&agent, &opt_args, &buffers));
    assert_eq!(agent.get_local_md().unwrap().len(), initial_md);

    drop(buffers);
    assert_unpinned_to(baseline, "after the batch and its memory went");
    println!(
        "{BATCH} x {} MiB: registered one by one in {registered:?}, \
         deregistered in one call in {deregistered:?}; pinned KiB {baseline:?} -> {pinned:?} -> {:?}",
        CHUNK >> 20,
        pinned_kib()
    );
}

#[test]
fn a_batch_leaves_registrations_it_does_not_name_alone() {
    let _serial = serial();
    let (agent, opt_args) = agent_with_ucx("dereg_batch_subset");
    let initial_md = agent.get_local_md().unwrap().len();
    let baseline = baseline_after_warm_up(&agent, &opt_args);

    let buffers = chunks(BATCH);
    let (withdrawn, kept): (Vec<_>, Vec<_>) = register_each(&agent, &opt_args, &buffers)
        .into_iter()
        .enumerate()
        .partition(|(index, _)| index % 2 == 0);
    let full_md = agent.get_local_md().unwrap().len();

    agent
        .deregister_memory_from_dlist(
            withdrawn.into_iter().map(|(_, handle)| handle).collect(),
            Some(&opt_args),
        )
        .expect("Failed to deregister half of the chunks");
    let half_md = agent.get_local_md().unwrap().len();
    assert!(
        initial_md < half_md && half_md < full_md,
        "{initial_md} < {half_md} < {full_md}"
    );

    // deregisterMem refuses a list naming anything not registered, so this
    // succeeding shows every kept chunk was still registered.
    agent
        .deregister_memory_from_dlist(
            kept.into_iter().map(|(_, handle)| handle).collect(),
            Some(&opt_args),
        )
        .expect("The chunks left out of the first batch were not all still registered");
    assert_eq!(agent.get_local_md().unwrap().len(), initial_md);

    drop(buffers);
    assert_unpinned_to(baseline, "after both halves and their memory went");
}

#[test]
fn a_batch_skips_handles_already_deregistered() {
    let _serial = serial();
    let (agent, opt_args) = agent_with_ucx("dereg_batch_skips");
    let initial_md = agent.get_local_md().unwrap().len();

    let buffers = chunks(8);
    let mut handles = register_each(&agent, &opt_args, &buffers);
    handles[1].deregister().unwrap();
    handles[6].deregister().unwrap();

    agent
        .deregister_memory_from_dlist(handles, Some(&opt_args))
        .expect("Handles already deregistered must not fail the batch");
    assert_eq!(agent.get_local_md().unwrap().len(), initial_md);
}

#[test]
fn a_refused_batch_still_deregisters_each_handle() {
    let _serial = serial();
    let (owner, owner_args) = agent_with_ucx("dereg_batch_owner");
    let (other, other_args) = agent_with_ucx("dereg_batch_other");
    let initial_md = owner.get_local_md().unwrap().len();
    baseline_after_warm_up(&other, &other_args);
    let baseline = baseline_after_warm_up(&owner, &owner_args);

    let buffers = chunks(BATCH);
    let handles = register_each(&owner, &owner_args, &buffers);
    let result = other.deregister_memory_from_dlist(handles, None);
    assert!(
        matches!(result, Err(NixlError::InvalidParam)),
        "got {result:?}"
    );
    // The refused handles were dropped, each deregistering with its own agent.
    assert_eq!(owner.get_local_md().unwrap().len(), initial_md);

    drop(buffers);
    assert_unpinned_to(baseline, "after the refused batch's memory went");
}

#[test]
fn repeated_batches_leave_nothing_behind() {
    let _serial = serial();
    let (agent, opt_args) = agent_with_ucx("dereg_batch_repeated");
    let initial_md = agent.get_local_md().unwrap().len();
    let baseline = baseline_after_warm_up(&agent, &opt_args);

    for round in 0..10 {
        // Fresh memory each round, as each blob a producer writes is.
        let buffers = chunks(64);
        let handles = register_each(&agent, &opt_args, &buffers);
        agent
            .deregister_memory_from_dlist(handles, Some(&opt_args))
            .unwrap_or_else(|error| panic!("round {round}: {error:?}"));
        assert_eq!(
            agent.get_local_md().unwrap().len(),
            initial_md,
            "round {round}"
        );
        drop(buffers);
        assert_unpinned_to(baseline, &format!("round {round}"));
    }
}

#[test]
fn a_batch_deregisters_while_another_thread_registers() {
    let _serial = serial();
    let (agent, opt_args) = agent_with_ucx("dereg_batch_concurrent");
    let initial_md = agent.get_local_md().unwrap().len();
    let baseline = baseline_after_warm_up(&agent, &opt_args);

    // One thread registers and drops chunks one at a time, as a producer
    // committing new data does, while this one withdraws full batches.
    let registering = {
        let agent = agent.clone();
        thread::spawn(move || {
            let opt_args = opt_args_for(&agent);
            let buffers = chunks(32);
            for _ in 0..10 {
                drop(register_each(&agent, &opt_args, &buffers));
            }
        })
    };
    let buffers = chunks(BATCH);
    for round in 0..5 {
        let handles = register_each(&agent, &opt_args, &buffers);
        agent
            .deregister_memory_from_dlist(handles, Some(&opt_args))
            .unwrap_or_else(|error| panic!("round {round}: {error:?}"));
    }
    registering.join().expect("The registering thread panicked");
    assert_eq!(agent.get_local_md().unwrap().len(), initial_md);

    drop(buffers);
    assert_unpinned_to(baseline, "after both threads' memory went");
}
