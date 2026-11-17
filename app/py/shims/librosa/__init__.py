"""The parts of librosa 0.11 that bacpipe uses, written with the same libraries librosa itself calls (soundfile to
read, soxr to resample), so that the samples are the ones librosa.load returns.

    load, resample, to_mono, get_duration, util.fix_length, stft, amplitude_to_db

Only what bacpipe calls is here, with librosa's defaults. Files that soundfile cannot read (librosa would fall back
on audioread for those) raise soundfile's error.
"""
import numpy as np
import soundfile as sf
import soxr

from . import util   # noqa: F401
from .util import fix_length

__version__ = "0.11.0+emberiza"


def to_mono(y):
    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))
    return y


def resample(y, *, orig_sr, target_sr, res_type="soxr_hq", fix=True, scale=False, axis=-1, **kwargs):
    if orig_sr == target_sr:
        return y
    if not res_type.startswith("soxr"):
        raise NotImplementedError(f"res_type {res_type!r}: only librosa's default family, soxr, is available here")
    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[axis] * ratio))
    y_hat = np.apply_along_axis(soxr.resample, axis=axis, arr=y, in_rate=orig_sr, out_rate=target_sr, quality=res_type)
    if fix:
        y_hat = fix_length(y_hat, size=n_samples, axis=axis, **kwargs)
    if scale:
        y_hat /= np.sqrt(ratio)
    return np.asarray(y_hat, dtype=y.dtype)


def load(path, *, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type="soxr_hq"):
    with sf.SoundFile(str(path)) as sf_desc:
        sr_native = sf_desc.samplerate
        if offset:
            sf_desc.seek(int(offset * sr_native))
        frame_duration = int(duration * sr_native) if duration is not None else -1
        y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T
    if mono:
        y = to_mono(y)
    if sr is not None:
        y = resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)
    else:
        sr = sr_native
    return y, sr


def get_duration(*, y=None, sr=22050, path=None, **_):
    if path is not None:
        return sf.info(str(path)).duration
    return float(y.shape[-1]) / sr


def stft(y, *, n_fft=2048, hop_length=None, win_length=None, window="hann", center=True, pad_mode="constant"):
    """The short-time Fourier transform with librosa's conventions: a periodic Hann window of win_length centred in
    n_fft, frames centred on their time (the signal is padded by n_fft // 2), and a hop of win_length // 4."""
    if window != "hann":
        raise NotImplementedError("only the Hann window is available here")
    win_length = n_fft if win_length is None else win_length
    hop_length = win_length // 4 if hop_length is None else hop_length
    y = np.asarray(y)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_length) / win_length)
    left = (n_fft - win_length) // 2
    win = np.pad(win, (left, n_fft - win_length - left)).astype(y.dtype if y.dtype.kind == "f" else np.float32)
    if center:
        y = np.pad(y, n_fft // 2, mode=pad_mode)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop_length]
    dtype = np.complex64 if frames.dtype == np.float32 else np.complex128
    return np.fft.rfft(frames * win, axis=-1).T.astype(dtype)


def amplitude_to_db(S, *, ref=1.0, amin=1e-5, top_db=80.0):
    magnitude = np.abs(np.asarray(S))
    ref_value = ref(magnitude) if callable(ref) else np.abs(ref)
    power = np.square(magnitude)
    log_spec = 10.0 * np.log10(np.maximum(amin ** 2, power))
    log_spec -= 10.0 * np.log10(np.maximum(amin ** 2, ref_value ** 2))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec
