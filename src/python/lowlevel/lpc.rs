//! Low-level LPC bindings: linear-prediction analysis/synthesis, pitch
//! estimation, and long-term prediction helpers.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::python::numpy_io::{borrow_1d, vec_to_numpy_1d};

/// LPC predictor coefficients with their residual prediction-error energy.
///
/// Parameters
/// ----------
/// coeffs : Sequence[float]
///     Predictor coefficients ``a[1..=order]`` (zero-indexed).
/// prediction_error : float
///     Residual prediction-error energy after Levinson-Durbin.
#[pyclass(module = "opus_native.lowlevel", name = "LpcCoefficients", from_py_object)]
#[derive(Clone)]
pub struct LpcCoefficients {
    pub(crate) inner: crate::lpc::LpcCoefficients,
}

#[pymethods]
impl LpcCoefficients {
    #[new]
    fn new(coeffs: Vec<f32>, prediction_error: f64) -> Self {
        Self {
            inner: crate::lpc::LpcCoefficients {
                coeffs,
                prediction_error,
            },
        }
    }

    /// Predictor coefficients ``a[1..=order]`` as a list of floats.
    #[getter]
    fn coeffs(&self) -> Vec<f32> {
        self.inner.coeffs.clone()
    }

    /// Residual prediction-error energy.
    #[getter]
    fn prediction_error(&self) -> f64 {
        self.inner.prediction_error
    }

    /// LPC order (number of coefficients).
    #[getter]
    fn order(&self) -> usize {
        self.inner.coeffs.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "LpcCoefficients(order={}, prediction_error={})",
            self.inner.coeffs.len(),
            self.inner.prediction_error
        )
    }
}

/// Biased, Hamming-windowed autocorrelation up to ``max_lag`` inclusive.
#[pyfunction]
#[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float32]", max_lag) -> "numpy.typing.NDArray[numpy.float64]")]
pub fn compute_autocorrelation<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, f32>,
    max_lag: usize,
) -> Bound<'py, PyArray1<f64>> {
    let s = borrow_1d(&samples);
    vec_to_numpy_1d(py, crate::lpc::compute_autocorrelation(&s, max_lag))
}

/// Solve the Toeplitz system for LPC coefficients (Levinson-Durbin).
///
/// Returns ``None`` if the recursion is unstable.
#[pyfunction]
#[pyo3(signature = (autocorr: "numpy.typing.NDArray[numpy.float64]", order))]
pub fn levinson_durbin(autocorr: PyReadonlyArray1<'_, f64>, order: usize) -> Option<LpcCoefficients> {
    let a = borrow_1d(&autocorr);
    crate::lpc::levinson_durbin(&a, order).map(|inner| LpcCoefficients { inner })
}

/// Full LPC analysis of ``samples`` at the given ``order``.
#[pyfunction]
#[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float32]", order))]
pub fn lpc_analysis(samples: PyReadonlyArray1<'_, f32>, order: usize) -> LpcCoefficients {
    let s = borrow_1d(&samples);
    LpcCoefficients {
        inner: crate::lpc::lpc_analysis(&s, order),
    }
}

/// The LPC residual (prediction error) of ``samples`` under ``coeffs``.
#[pyfunction]
#[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float32]", coeffs) -> "numpy.typing.NDArray[numpy.float32]")]
pub fn lpc_residual<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
) -> Bound<'py, PyArray1<f32>> {
    let s = borrow_1d(&samples);
    vec_to_numpy_1d(py, crate::lpc::lpc_residual(&s, &coeffs.inner))
}

/// Reconstruct samples from an LPC ``residual`` and ``coeffs``.
#[pyfunction]
#[pyo3(signature = (residual: "numpy.typing.NDArray[numpy.float32]", coeffs) -> "numpy.typing.NDArray[numpy.float32]")]
pub fn lpc_synthesis<'py>(
    py: Python<'py>,
    residual: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
) -> Bound<'py, PyArray1<f32>> {
    let r = borrow_1d(&residual);
    vec_to_numpy_1d(py, crate::lpc::lpc_synthesis(&r, &coeffs.inner))
}

/// Stateful LPC residual: like :func:`lpc_residual` but carrying filter memory
/// across calls. Returns the residual and the updated state.
///
/// Parameters
/// ----------
/// samples : numpy.ndarray
///     1-D ``float32`` input.
/// coeffs : LpcCoefficients
///     Predictor coefficients.
/// state : Sequence[float], optional
///     Filter memory from the previous call (empty to start). Defaults to empty.
#[pyfunction]
#[pyo3(signature = (samples, coeffs, state = Vec::new()) -> "tuple[numpy.typing.NDArray[numpy.float32], list[float]]")]
pub fn lpc_residual_stateful<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
    state: Vec<f32>,
) -> (Bound<'py, PyArray1<f32>>, Vec<f32>) {
    let mut st = state;
    let s = borrow_1d(&samples);
    let out = crate::lpc::lpc_residual_stateful(&s, &coeffs.inner, &mut st);
    (vec_to_numpy_1d(py, out), st)
}

/// Stateful LPC synthesis: like :func:`lpc_synthesis` but carrying filter memory
/// across calls. Returns the reconstructed samples and the updated state.
///
/// Parameters
/// ----------
/// residual : numpy.ndarray
///     1-D ``float32`` residual.
/// coeffs : LpcCoefficients
///     Predictor coefficients.
/// state : Sequence[float], optional
///     Filter memory from the previous call (empty to start). Defaults to empty.
#[pyfunction]
#[pyo3(signature = (residual, coeffs, state = Vec::new()) -> "tuple[numpy.typing.NDArray[numpy.float32], list[float]]")]
pub fn lpc_synthesis_stateful<'py>(
    py: Python<'py>,
    residual: PyReadonlyArray1<'_, f32>,
    coeffs: &LpcCoefficients,
    state: Vec<f32>,
) -> (Bound<'py, PyArray1<f32>>, Vec<f32>) {
    let mut st = state;
    let r = borrow_1d(&residual);
    let out = crate::lpc::lpc_synthesis_stateful(&r, &coeffs.inner, &mut st);
    (vec_to_numpy_1d(py, out), st)
}

/// Estimate the pitch period (in samples) and confidence of ``samples``.
///
/// Returns ``None`` when no confident pitch is found.
#[pyfunction]
#[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float32]", sample_rate) -> "tuple[int, float] | None")]
pub fn estimate_pitch(samples: PyReadonlyArray1<'_, f32>, sample_rate: u32) -> Option<(usize, f32)> {
    let s = borrow_1d(&samples);
    crate::lpc::estimate_pitch(&s, sample_rate)
}

/// Long-term-prediction residual for a given ``lag`` and ``gain``.
#[pyfunction]
#[pyo3(signature = (samples: "numpy.typing.NDArray[numpy.float32]", lag, gain) -> "numpy.typing.NDArray[numpy.float32]")]
pub fn ltp_residual<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, f32>,
    lag: usize,
    gain: f32,
) -> Bound<'py, PyArray1<f32>> {
    let s = borrow_1d(&samples);
    vec_to_numpy_1d(py, crate::lpc::ltp_residual(&s, lag, gain))
}

/// Reconstruct samples from a long-term-prediction ``residual``.
#[pyfunction]
#[pyo3(signature = (residual: "numpy.typing.NDArray[numpy.float32]", lag, gain) -> "numpy.typing.NDArray[numpy.float32]")]
pub fn ltp_synthesis<'py>(
    py: Python<'py>,
    residual: PyReadonlyArray1<'_, f32>,
    lag: usize,
    gain: f32,
) -> Bound<'py, PyArray1<f32>> {
    let r = borrow_1d(&residual);
    vec_to_numpy_1d(py, crate::lpc::ltp_synthesis(&r, lag, gain))
}
