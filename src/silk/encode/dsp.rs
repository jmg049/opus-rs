//! Shared floating-point analysis kernels for the SILK encoder (RFC 6716
//! §5.2; normative `silk/float/`). These small building blocks are used by
//! more than one analysis stage (noise shaping, pitch analysis): the sine
//! window, autocorrelation, the Schur recursion, reflection→prediction
//! conversion, bandwidth expansion, an energy accumulator, and the LPC
//! analysis filter.

/// Upper bound on the order these helpers handle (`MAX_SHAPE_LPC_ORDER`).
const MAX_ORDER: usize = 24;

/// `silk_apply_sine_window_FLP`: window `px` with a sine (`win_type==1`) or
/// cosine (`win_type==2`) slope of `length` samples (a multiple of 4).
pub(crate) fn apply_sine_window(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    debug_assert!(win_type == 1 || win_type == 2);
    debug_assert!(length & 3 == 0);
    let freq = core::f32::consts::PI / (length as f32 + 1.0);
    let c = 2.0 - freq * freq;
    let (mut s0, mut s1) = if win_type < 2 {
        (0.0f32, freq)
    } else {
        (1.0f32, 0.5 * c)
    };
    let mut k = 0;
    while k < length {
        px_win[k] = px[k] * 0.5 * (s0 + s1);
        px_win[k + 1] = px[k + 1] * s1;
        s0 = c * s1 - s0;
        px_win[k + 2] = px[k + 2] * 0.5 * (s1 + s0);
        px_win[k + 3] = px[k + 3] * s0;
        s1 = c * s0 - s1;
        k += 4;
    }
}

/// `silk_autocorrelation_FLP`: the first `count` autocorrelation taps.
pub(crate) fn autocorrelation(results: &mut [f32], input: &[f32], count: usize) {
    let n = input.len();
    let count = count.min(n);
    for (i, r) in results.iter_mut().enumerate().take(count) {
        let mut acc = 0.0f64;
        for j in 0..n - i {
            acc += f64::from(input[j]) * f64::from(input[j + i]);
        }
        *r = acc as f32;
    }
}

/// `silk_schur_FLP`: reflection coefficients from the autocorrelation,
/// returning the residual energy.
pub(crate) fn schur(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    let mut c = [[0.0f64; 2]; MAX_ORDER + 1];
    for k in 0..=order {
        c[k][0] = f64::from(auto_corr[k]);
        c[k][1] = f64::from(auto_corr[k]);
    }
    for k in 0..order {
        let rc_tmp = -c[k + 1][0] / c[0][1].max(1e-9);
        refl_coef[k] = rc_tmp as f32;
        for n in 0..order - k {
            let ctmp1 = c[n + k + 1][0];
            let ctmp2 = c[n][1];
            c[n + k + 1][0] = ctmp1 + ctmp2 * rc_tmp;
            c[n][1] = ctmp2 + ctmp1 * rc_tmp;
        }
    }
    c[0][1] as f32
}

/// `silk_k2a_FLP`: reflection coefficients to prediction coefficients.
pub(crate) fn k2a(a: &mut [f32], rc: &[f32], order: usize) {
    for k in 0..order {
        let rck = rc[k];
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a[n];
            let tmp2 = a[k - n - 1];
            a[n] = tmp1 + tmp2 * rck;
            a[k - n - 1] = tmp2 + tmp1 * rck;
        }
        a[k] = -rck;
    }
}

/// `silk_bwexpander_FLP`: chirp the AR filter towards the unit circle.
pub(crate) fn bwexpander(ar: &mut [f32], order: usize, chirp: f32) {
    let mut cfac = chirp;
    for v in ar.iter_mut().take(order - 1) {
        *v *= cfac;
        cfac *= chirp;
    }
    ar[order - 1] *= cfac;
}

/// `silk_energy_FLP`: sum of squares in double precision.
pub(crate) fn energy(data: &[f32]) -> f64 {
    data.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

/// `silk_LPC_analysis_filter_FLP`: the LPC prediction residual of `s`
/// (`r[ix] = s[ix] - Σ_j s[ix-1-j]·a[j]`), with the first `order` outputs
/// set to zero (the filter starts from zero state).
pub(crate) fn lpc_analysis_filter_flp(r: &mut [f32], a: &[f32], s: &[f32], length: usize, order: usize) {
    for ix in order..length {
        let mut pred = 0.0f32;
        for (j, &aj) in a.iter().enumerate().take(order) {
            pred += s[ix - 1 - j] * aj;
        }
        r[ix] = s[ix] - pred;
    }
    for v in r.iter_mut().take(order) {
        *v = 0.0;
    }
}
