"""Split a song into layers.

Two kinds of layers come out of one input file:

- Instrument stems from Demucs (htdemucs_6s): vocals, drums, bass, guitar,
  piano and "other". A neural network predicts a mask on the spectrogram and
  waveform for each instrument.
- Fourier layers computed directly from the short-time Fourier transform:
  harmonic vs percussive (median-filter HPSS) and five frequency bands.

For every layer the pipeline writes an MP3 for playback, plus a compact
log-frequency spectrogram (uint8) that the page uses for its pictures. It also
estimates tempo and key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import av
import lameenc
import numpy as np
from scipy import ndimage, signal

SR = 44100
STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
BANDS = [  # name, low Hz, high Hz
    ("sub", 0, 60),
    ("low", 60, 250),
    ("low-mid", 250, 2000),
    ("high-mid", 2000, 6000),
    ("air", 6000, SR / 2),
]

# Spectrogram pictures: log-spaced frequency rows, ~15 frames per second.
VIS_BINS = 160
VIS_FMIN, VIS_FMAX = 30.0, 16000.0
VIS_NFFT = 4096
VIS_HOP = 2940  # 44100 / 2940 = 15 fps

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Krumhansl-Kessler key profiles.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

Progress = Callable[[str, float], None]


def decode(path: Path) -> tuple[np.ndarray, dict]:
    """Decode any format FFmpeg knows to float32 stereo at 44.1 kHz, shape (2, n)."""
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        meta = {k.lower(): v for k, v in (container.metadata or {}).items()}
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        raise ValueError("No audio found in the file.")
    wav = np.concatenate(chunks, axis=1).astype(np.float32)
    return wav, {"title": meta.get("title"), "artist": meta.get("artist")}


def write_mp3(path: Path, wav: np.ndarray, kbps: int = 192) -> None:
    pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").T.copy()  # interleaved
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(SR)
    enc.set_channels(2)
    enc.set_quality(2)
    path.write_bytes(enc.encode(pcm.tobytes()) + enc.flush())


def separate(wav: np.ndarray, progress: Progress) -> dict[str, np.ndarray]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2)))
    progress("Loading the separation model", 0.0)
    model = get_model("htdemucs_6s")
    model.eval()
    mix = torch.from_numpy(wav)
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    n = mix.shape[1]

    def on_chunk(d: dict) -> None:
        if d.get("state") == "end":
            models = d.get("models", 1) or 1
            done = (d.get("model_idx_in_bag", 0) + min(1.0, (d["segment_offset"] + SR * 8) / n)) / models
            progress("Separating instruments", min(0.99, done))

    progress("Separating instruments", 0.0)
    with torch.no_grad():
        out = apply_model(model, ((mix - mean) / std)[None], device="cpu", shifts=1, split=True,
                          overlap=0.25, progress=False, num_workers=0, callback=on_chunk)[0]
    out = out * std + mean
    found = {name: out[i].numpy().astype(np.float32) for i, name in enumerate(model.sources)}
    return {name: found[name] for name in STEMS if name in found}


def stft(x: np.ndarray, nfft: int = 4096, hop: int = 1024) -> np.ndarray:
    _, _, z = signal.stft(x, fs=SR, nperseg=nfft, noverlap=nfft - hop, boundary="even", padded=True)
    return z


def istft(z: np.ndarray, n: int, nfft: int = 4096, hop: int = 1024) -> np.ndarray:
    _, x = signal.istft(z, fs=SR, nperseg=nfft, noverlap=nfft - hop, boundary=True)
    x = x[..., :n]
    if x.shape[-1] < n:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])])
    return x.astype(np.float32)


def fourier_layers(wav: np.ndarray, progress: Progress) -> dict[str, np.ndarray]:
    """Harmonic/percussive split and frequency bands, using only the STFT."""
    n = wav.shape[1]
    progress("Fourier transform", 0.0)
    z = stft(wav)  # (2, freqs, frames)
    mag = np.abs(z).mean(0)
    # HPSS (FitzGerald 2010): sustained tones are smooth along time, hits are
    # smooth along frequency. Median-filter each way and compare.
    progress("Harmonic / percussive split", 0.2)
    harm = ndimage.median_filter(mag, size=(1, 17), mode="nearest")
    perc = ndimage.median_filter(mag, size=(17, 1), mode="nearest")
    eps = 1e-10
    mh = harm**2 / (harm**2 + perc**2 + eps)
    layers = {
        "harmonic": istft(z * mh, n),
        "percussive": istft(z * (1 - mh), n),
    }
    progress("Frequency bands", 0.7)
    freqs = np.fft.rfftfreq(4096, 1 / SR)
    for name, lo, hi in BANDS:
        # Soft edges (a half-octave crossfade) avoid ringing.
        w = np.ones_like(freqs)
        if lo > 0:
            w *= np.clip((np.log2(np.maximum(freqs, 1) / lo) + 0.25) / 0.5, 0, 1)
        if hi < SR / 2:
            w *= np.clip((np.log2(hi / np.maximum(freqs, 1)) + 0.25) / 0.5, 0, 1)
        layers[name] = istft(z * w[None, :, None], n)
    return layers


def vis_matrix() -> np.ndarray:
    """Weights mapping linear STFT bins to VIS_BINS log-spaced rows."""
    freqs = np.fft.rfftfreq(VIS_NFFT, 1 / SR)
    edges = np.geomspace(VIS_FMIN, VIS_FMAX, VIS_BINS + 2)
    m = np.zeros((VIS_BINS, len(freqs)), np.float32)
    for i in range(VIS_BINS):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        m[i] = np.maximum(0, np.minimum(up, down))
        if m[i].sum() == 0:  # narrow low rows: take the nearest bin
            m[i, np.argmin(np.abs(freqs - c))] = 1
    return m / m.sum(1, keepdims=True)


def vis_power(wav: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Log-frequency power spectrogram, shape (frames, VIS_BINS)."""
    _, _, z = signal.stft(wav.mean(0), fs=SR, nperseg=VIS_NFFT, noverlap=VIS_NFFT - VIS_HOP,
                          boundary="even", padded=True)
    return (m @ (np.abs(z) ** 2)).T.astype(np.float32)


def to_u8(powers: list[np.ndarray]) -> list[np.ndarray]:
    """Shared dB scale across a layer set: 0 = -80 dB below the loudest cell."""
    top = max(float(p.max()) for p in powers) + 1e-12
    return [np.clip((10 * np.log10(p / top + 1e-12) + 80) / 80 * 255, 0, 255).astype(np.uint8) for p in powers]


def tempo(perc: np.ndarray) -> float | None:
    """Tempo from the autocorrelation of the onset envelope of the drums."""
    hop = 512
    _, _, z = signal.stft(perc.mean(0), fs=SR, nperseg=2048, noverlap=2048 - hop)
    flux = np.maximum(0, np.diff(np.log1p(100 * np.abs(z)), axis=1)).sum(0)
    flux = flux - flux.mean()
    if not np.any(flux):
        return None
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    fps = SR / hop
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    ok = (bpm >= 60) & (bpm <= 180)
    # Prefer tempos near 100 BPM, as dance and pop usually are.
    weight = np.exp(-0.5 * (np.log2(np.where(ok, bpm, 1) / 110) / 0.9) ** 2)
    score = np.where(ok, ac * weight, -np.inf)
    lag = int(np.argmax(score))
    return round(float(bpm[lag]), 1)


def key(wav: np.ndarray) -> dict | None:
    """Key from a pitch-class profile and the Krumhansl-Kessler profiles."""
    _, _, z = signal.stft(wav.mean(0), fs=SR, nperseg=8192, noverlap=8192 - 4096)
    p = (np.abs(z) ** 2).sum(1)
    freqs = np.fft.rfftfreq(8192, 1 / SR)
    ok = (freqs > 55) & (freqs < 2000)
    pcs = np.round(12 * np.log2(freqs[ok] / 440) + 69).astype(int) % 12
    chroma = np.bincount(pcs, weights=p[ok], minlength=12)
    if chroma.sum() == 0:
        return None
    best = None
    for mode, prof in (("major", MAJOR), ("minor", MINOR)):
        for t in range(12):
            r = np.corrcoef(chroma, np.roll(prof, t))[0, 1]
            if best is None or r > best[0]:
                best = (r, t, mode)
    return {"tonic": NOTE_NAMES[best[1]], "mode": best[2], "confidence": round(float(best[0]), 2)}


def process(src: Path, out: Path, progress: Progress) -> dict:
    t0 = time.time()
    progress("Decoding", 0.0)
    wav, meta = decode(src)
    n = wav.shape[1]
    duration = n / SR
    if duration > 15 * 60:
        raise ValueError("Songs up to 15 minutes are supported.")

    def stage(lo: float, hi: float) -> Progress:
        return lambda label, f: progress(label, lo + (hi - lo) * f)

    stems = separate(wav, stage(0.02, 0.85))
    four = fourier_layers(wav, stage(0.85, 0.92))

    progress("Encoding layers", 0.92)
    out.mkdir(parents=True, exist_ok=True)
    write_mp3(out / "mix.mp3", wav)
    m = vis_matrix()
    sets = {"stems": stems, "fourier": four}
    layer_info: dict[str, list] = {}
    for set_name, layers in sets.items():
        names = list(layers)
        powers = [vis_power(layers[k], m) for k in names]
        u8 = to_u8(powers)
        frames = u8[0].shape[0]
        (out / f"spec-{set_name}.bin").write_bytes(b"".join(a.tobytes() for a in u8))
        info = []
        for k, p in zip(names, powers):
            write_mp3(out / f"{set_name}-{k}.mp3", layers[k])
            rms = float(np.sqrt(np.mean(layers[k] ** 2)))
            info.append({"name": k, "file": f"{set_name}-{k}.mp3", "level": round(20 * np.log10(rms + 1e-9), 1)})
        layer_info[set_name] = info
    mix_pow = vis_power(wav, m)
    (out / "spec-mix.bin").write_bytes(to_u8([mix_pow])[0].tobytes())

    progress("Tempo and key", 0.98)
    result = {
        "title": meta.get("title"),
        "artist": meta.get("artist"),
        "duration": round(duration, 2),
        "tempo": tempo(stems.get("drums", four["percussive"])),
        "key": key(sum(stems[k] for k in stems if k not in ("drums",))),
        "layers": layer_info,
        "spec": {"bins": VIS_BINS, "fmin": VIS_FMIN, "fmax": VIS_FMAX, "fps": SR / VIS_HOP, "frames": frames},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "result.json").write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    import sys

    # Usage: python -m layers.pipeline <input> <out dir> [--json]
    # With --json, progress is printed as one JSON object per line (for the server).
    as_json = "--json" in sys.argv

    def report(label: str, f: float) -> None:
        if as_json:
            print(json.dumps({"stage": label, "progress": round(f, 4)}), flush=True)
        else:
            print(f"{f * 100:5.1f}%  {label}", flush=True)

    try:
        res = process(Path(sys.argv[1]), Path(sys.argv[2]), report)
    except Exception as e:  # reported to the page as the failure reason
        if as_json:
            print(json.dumps({"error": str(e) or e.__class__.__name__}), flush=True)
            sys.exit(1)
        raise
    if as_json:
        print(json.dumps({"done": True}), flush=True)
    else:
        print(json.dumps({k: v for k, v in res.items() if k != "layers"}, indent=2))
