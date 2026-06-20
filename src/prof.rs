//! Lightweight env-gated stage profiler for performance work.
//!
//! Zero-cost when off: [`enabled`] is a single cached atomic load, and every
//! [`scope`] call compiles to that check plus (when enabled) one `Instant` pair
//! per *stage* - only ever wrapped around coarse, once-per-frame stages, never
//! inner loops, so the measurement overhead is negligible.
//!
//! Enable with `OPUS_PROF=1` in the environment and call [`dump`] at the end of
//! a run (e.g. a benchmark example) to print accumulated per-label wall time,
//! call counts, and the per-call mean, sorted by total time.
//!
//! This is a developer tool, not part of the public API; it exists only under
//! the `std` feature.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

static STATE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = off, 2 = on

/// Whether profiling is active (reads `OPUS_PROF` once, then caches).
#[must_use]
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var_os("OPUS_PROF").is_some_and(|v| v != "0" && !v.is_empty());
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        },
    }
}

thread_local! {
    static ACC: RefCell<HashMap<&'static str, (u128, u64)>> = RefCell::new(HashMap::new());
}

/// An RAII timer; the elapsed time is added to `label`'s total when dropped.
pub struct Guard {
    label: &'static str,
    start: Instant,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos();
        let label = self.label;
        ACC.with(|a| {
            let mut a = a.borrow_mut();
            let e = a.entry(label).or_insert((0, 0));
            e.0 += ns;
            e.1 += 1;
        });
    }
}

/// Begins timing a stage. Returns `None` (no-op) when profiling is disabled.
#[must_use]
pub fn scope(label: &'static str) -> Option<Guard> {
    if enabled() {
        Some(Guard {
            label,
            start: Instant::now(),
        })
    } else {
        None
    }
}

/// Prints the accumulated per-label timings for the current thread, sorted by
/// total time descending. Clears the accumulator afterwards.
pub fn dump() {
    if !enabled() {
        return;
    }
    ACC.with(|a| {
        let mut a = a.borrow_mut();
        let mut rows: Vec<_> = a.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
        rows.sort_by_key(|y| std::cmp::Reverse(y.1));
        eprintln!("\n--- OPUS_PROF (per-thread) ---");
        eprintln!("{:<28} {:>12} {:>10} {:>12}", "stage", "total ms", "calls", "us/call");
        for (label, total_ns, calls) in rows {
            let total_ms = total_ns as f64 / 1e6;
            let us_call = total_ns as f64 / 1e3 / calls as f64;
            eprintln!("{label:<28} {total_ms:>12.2} {calls:>10} {us_call:>12.3}");
        }
        a.clear();
    });
}

/// Wraps a block in a [`scope`] guard bound to `$label`, returning the block's
/// value. Expands to just the block when profiling is compiled out.
#[macro_export]
macro_rules! prof_scope {
    ($label:expr, $body:expr) => {{
        let _g = $crate::prof::scope($label);
        $body
    }};
}
