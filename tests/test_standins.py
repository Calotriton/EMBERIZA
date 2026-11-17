"""The stand-ins for libraries a browser cannot load do what bacpipe relies on."""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from emberiza import replay, runtime

runtime.install()
import librosa  # noqa: E402
import torch    # noqa: E402


def test_a_real_library_would_win_except_onnxruntime():
    assert "shims" in Path(torch.__file__).parts and sys.modules["onnxruntime"] is replay


def test_tensors_behave_like_pytorchs_where_bacpipe_looks():
    start = torch.tensor([])
    assert start.dtype == np.float32 and start.shape == (0,) and start.size == 0
    rows = torch.softmax(torch.from_numpy(np.random.rand(3, 5).astype(np.float32)), dim=-1)
    joined = torch.cat([start, rows.clone().detach()])                      # bacpipe starts its lists empty
    assert joined.shape == (3, 5) and joined.size(0) == 3 and joined.dim() == 2
    values, indices = torch.max(joined, dim=1)
    assert np.allclose(values, np.asarray(rows).max(1)) and np.array_equal(indices, np.asarray(rows).argmax(1))
    assert torch.tensor([1.5, 2.5]).dtype == np.float32 and torch.tensor([1, 2]).dtype == np.int64
    assert np.allclose(rows.sum(axis=1), 1, atol=1e-6)
    assert joined.unsqueeze(0).shape == (1, 3, 5) and joined.cpu().numpy().__class__ is np.ndarray


def test_probe_files_are_pytorchs_format(tmp_path):
    layer = torch.nn.Linear(4, 3)
    torch.save(layer.state_dict(), tmp_path / "probe.pt")
    back = torch.load(tmp_path / "probe.pt")
    assert list(back) == ["weight", "bias"] and np.array_equal(back["weight"], layer.weight)
    import zipfile
    with zipfile.ZipFile(tmp_path / "probe.pt") as z:
        assert {"archive/data.pkl", "archive/data/0", "archive/data/1", "archive/version"} <= set(z.namelist())
        assert b"torch._utils" in z.read("archive/data.pkl") or b"_rebuild_tensor_v2" in z.read("archive/data.pkl")


def test_a_pickle_with_anything_else_is_refused(tmp_path):
    import pickle
    import zipfile
    with zipfile.ZipFile(tmp_path / "bad.pt", "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(print))
    with pytest.raises(pickle.UnpicklingError):
        torch.load(tmp_path / "bad.pt")


def test_the_linear_probe_learns():
    rng = np.random.RandomState(0)
    x = np.vstack([rng.normal(loc=c, size=(40, 6)) for c in (-2, 0, 2)]).astype(np.float32)
    y = np.repeat([0, 1, 2], 40)
    layer, loss_fn = torch.nn.Linear(6, 3), torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(layer.parameters(), lr=0.05)
    first = None
    for _ in range(60):
        loss = loss_fn(layer(torch.tensor(x)), torch.tensor(y))
        first = first or loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    _, predicted = torch.max(layer(torch.tensor(x)), 1)
    assert loss.item() < first / 3 and (predicted == y).mean() > 0.9


def test_data_loader_batches_in_order():
    frames = torch.tensor(np.arange(22, dtype=np.float32).reshape(11, 2))
    batches = list(torch.utils.data.DataLoader(frames, batch_size=4, shuffle=False))
    assert [len(b) for b in batches] == [4, 4, 3] and np.array_equal(np.concatenate(batches), frames)


def test_replay_gives_results_back_in_order_and_refuses_a_mismatch():
    replay.InferenceSession.signatures["toy.onnx"] = ("in", ["a", "b"])
    session = replay.InferenceSession("somewhere/toy.onnx")
    session.feed({"a": np.arange(10, dtype=np.float32).reshape(5, 2)})
    first = session.run(None, {"in": np.zeros((3, 4), np.float32)})
    second = session.run(["a"], {"in": np.zeros((2, 4), np.float32)})
    assert first[0].tolist() == [[0, 1], [2, 3], [4, 5]] and first[1].shape == (3, 0) and second[0].tolist() == [[6, 7], [8, 9]]
    with pytest.raises(RuntimeError, match="results for 0 are waiting"):
        session.run(None, {"in": np.zeros((1, 4), np.float32)})


def test_load_reads_resamples_and_cuts_like_librosa(tmp_path):
    sr = 48000
    t = np.arange(sr * 2) / sr
    stereo = np.stack([0.4 * np.sin(2 * np.pi * 1000 * t), 0.2 * np.sin(2 * np.pi * 3000 * t)], axis=1)
    sf.write(tmp_path / "tone.wav", stereo, sr, subtype="PCM_16")
    y, got = librosa.load(tmp_path / "tone.wav", sr=32000)
    assert got == 32000 and y.dtype == np.float32 and y.shape == (64000,)          # ceil(96000 * 32000 / 48000)
    spectrum = np.abs(np.fft.rfft(y))
    assert {int(np.argmax(spectrum) / 2), int(np.argsort(spectrum)[-2] / 2)} <= {1000, 3000, 999, 1001, 2999, 3001}
    native, got = librosa.load(tmp_path / "tone.wav", sr=None, mono=False)
    assert got == sr and native.shape == (2, sr * 2)
    part, _ = librosa.load(tmp_path / "tone.wav", sr=None, offset=0.5, duration=0.25)
    assert part.shape == (sr // 4,)
    assert librosa.get_duration(path=tmp_path / "tone.wav") == 2.0
    assert librosa.util.fix_length(np.arange(3.0), size=7, mode="wrap").tolist() == [0, 1, 2, 0, 1, 2, 0]


def test_spectrogram_of_a_tone_peaks_at_the_tone():
    sr = 32000
    y = np.sin(2 * np.pi * 4000 * np.arange(sr) / sr).astype(np.float32)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, win_length=1024)), ref=np.max)
    assert S.shape == (1025, 1 + sr // 256) and S.max() == 0 and S.min() >= -80
    assert abs(np.argmax(S[:, 60]) * sr / 2048 - 4000) < 20
