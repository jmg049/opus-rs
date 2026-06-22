//! Zero-copy NumPy conversion helpers shared by the encoder and decoder.
//!
//! The codec already produces an owned, interleaved `Vec<T>` of PCM. We hand
//! that buffer straight to NumPy with [`PyArray1::from_vec`], which *moves* the
//! allocation into the array - no second allocation, no element copy. A
//! contiguous reshape to `(frames, channels)` is then a view over the same
//! memory (C order maps element `[f, c]` to `f * channels + c`, exactly the
//! interleaved layout), so multichannel output costs nothing extra either.

use std::borrow::Cow;

use numpy::{Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Borrow a 1-D NumPy array as a slice, zero-copy when C-contiguous (copied
/// once otherwise). Used by the low-level SILK/LPC entry points.
pub fn borrow_1d<'a, T: Element + Copy>(arr: &'a PyReadonlyArray1<'a, T>) -> Cow<'a, [T]> {
    match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

/// Move an owned `Vec<T>` into a 1-D NumPy array with no copy.
pub fn vec_to_numpy_1d<T: Element>(py: Python<'_>, v: Vec<T>) -> Bound<'_, PyArray1<T>> {
    PyArray1::from_vec(py, v)
}

/// Borrow an interleaved f32 PCM buffer from a NumPy array for encoding.
///
/// Accepts a 1-D interleaved array or a 2-D ``(frames, channels)`` array (the
/// shape [`interleaved_f32_to_numpy`] produces). The borrow is **zero-copy**
/// when the array is C-contiguous - the common case, including anything that
/// came from this library or `numpy.ascontiguousarray`; a non-contiguous array
/// is copied once into an owned buffer.
pub fn borrow_interleaved_f32<'a>(arr: &'a PyReadonlyArrayDyn<'a, f32>, channels: usize) -> PyResult<Cow<'a, [f32]>> {
    let shape = arr.shape();
    let total = match *shape {
        [n] => n,
        [frames, ch] => {
            if ch != channels {
                return Err(PyValueError::new_err(format!(
                    "array has {ch} columns but the codec is configured for {channels} channels"
                )));
            }
            frames * ch
        },
        _ => {
            return Err(PyValueError::new_err(
                "PCM must be a 1-D interleaved array or a 2-D (frames, channels) array",
            ));
        },
    };
    if channels == 0 || total % channels != 0 {
        return Err(PyValueError::new_err(
            "PCM length must be a whole number of frames for the channel count",
        ));
    }
    Ok(match arr.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    })
}

/// Move an interleaved PCM `Vec<f32>` into a NumPy array shaped
/// `(frames, channels)` with no copy. Mono is returned as `(frames, 1)`.
pub fn interleaved_f32_to_numpy<'py>(
    py: Python<'py>,
    pcm: Vec<f32>,
    channels: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    debug_assert!(channels != 0);
    let frames = pcm.len() / channels;
    // from_vec takes ownership: the Vec's buffer becomes the array's buffer.
    let flat = PyArray1::from_vec(py, pcm);
    // Contiguous reshape -> view over the same buffer, no copy.
    flat.reshape([frames, channels])
}

/// Move an interleaved PCM `Vec<i16>` into a NumPy array shaped
/// `(frames, channels)` with no copy. Mono is returned as `(frames, 1)`.
pub fn interleaved_i16_to_numpy<'py>(
    py: Python<'py>,
    pcm: Vec<i16>,
    channels: usize,
) -> PyResult<Bound<'py, PyArray2<i16>>> {
    debug_assert!(channels != 0);
    let frames = pcm.len() / channels;
    let flat = PyArray1::from_vec(py, pcm);
    flat.reshape([frames, channels])
}
