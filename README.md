# Layers

Give it a song and hear what it's made of. Layers splits a recording into layers you can play, mute and solo on their own:

- **Instruments.** [Demucs](https://github.com/adefossez/demucs) (`htdemucs_6s`) separates vocals, drums, bass, guitar, piano and "other". It is a neural network that works on the spectrogram and the waveform together.
- **Harmonic / percussive.** Fourier only: a median filter on the short-time Fourier transform splits sustained tones (horizontal lines on the spectrogram) from hits (vertical lines). See FitzGerald, *Harmonic/percussive separation using median filtering* (DAFx 2010).
- **Frequency bands.** Fourier only: sub bass, low, low-mid, high-mid and air.

The page shows a spectrogram coloured by layer, a strip for each layer showing where it plays, and an estimate of the tempo and key.

## Layout

| Path | What it is |
| --- | --- |
| `layers/pipeline.py` | Decoding (PyAV), separation, Fourier layers, MP3 encoding, spectrogram data, tempo and key. Also runs as a CLI. |
| `layers/server.py` | Starlette app: upload, a one-at-a-time job queue (each job runs in its own process), and the result files. |
| `web/index.html` | The page: library, player (Web Audio, sample-accurate across layers) and drawings. No build step. |
| `deploy/` | systemd user service. |

## Run

```sh
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m layers.server        # http://127.0.0.1:8020/
```

Or process one file from the command line:

```sh
.venv/bin/python -m layers.pipeline song.mp3 out/
```

On a 6-core laptop CPU a 3-minute song takes about 2.5 minutes and up to 4 GB of RAM. The Demucs weights (about 80 MB) download on first use.

Settings (environment variables): `LAYERS_HOST` (default `127.0.0.1`), `LAYERS_PORT` (default `8020`), `LAYERS_DATA` (default `./data`).

## Use files you own

Layers processes audio files you upload. It does not download from streaming services. Stems of a copyrighted song are for your own listening and study; don't publish them.

## Credits

- Demucs by Alexandre Défossez / Meta AI, MIT license. The pretrained weights are downloaded from Meta's servers on first use.
- Harmonic/percussive separation after Derry FitzGerald (2010). Key estimation uses the Krumhansl–Kessler key profiles.
