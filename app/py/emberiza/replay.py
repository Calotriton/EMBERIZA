"""The onnxruntime that bacpipe's model wrappers talk to. The network itself has already run elsewhere, in ONNX
Runtime Web in the browser (or in the real onnxruntime in the tests), because that call cannot be made from inside
running Python there. This session hands the results back in the order bacpipe asks for them.

For each recording, the caller:
  1. takes the audio windows bacpipe prepared;
  2. runs the model on them;
  3. gives the outputs to `InferenceSession.feed(...)`;
  4. lets bacpipe's own code run. Its `session.run(...)` calls then return those outputs, batch by batch.

`run` checks that it is given as many windows as it has results for, so a mismatch stops the run instead of pairing
embeddings with the wrong audio.
"""
import numpy as np


class _Arg:
    def __init__(self, name, type_="tensor(float)", shape=None):
        self.name, self.type, self.shape = name, type_, shape


class SessionOptions:
    graph_optimization_level = None


class GraphOptimizationLevel:
    ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL = 0, 1, 2, 99


def get_available_providers():
    return ["CPUExecutionProvider"]


class InferenceSession:
    # What each model file takes and returns, by file name: read from the files with the real onnxruntime
    # (tests/test_models.py checks it). {file name: (input name, [output names in the model's order])}
    signatures = {}
    sessions = {}

    def __init__(self, path, sess_options=None, providers=None, **_):
        self.path = str(path)
        name = self.path.replace("\\", "/").rsplit("/", 1)[-1]
        if name not in InferenceSession.signatures:
            raise ValueError(f"no input and output names are known for the model file {name}")
        input_name, output_names = InferenceSession.signatures[name]
        self._inputs, self._outputs = [_Arg(input_name)], [_Arg(n) for n in output_names]
        self._pending, self._offset = None, 0
        InferenceSession.sessions[name] = self

    def get_providers(self):
        return ["EmberizaExecutionProvider"]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def feed(self, outputs):
        """outputs: {output name: array with one row per window}, for the windows the next `run` calls will bring."""
        self._pending, self._offset = {k: np.asarray(v) for k, v in outputs.items()}, 0

    def pending_rows(self):
        return 0 if not self._pending else len(next(iter(self._pending.values()))) - self._offset

    def run(self, output_names, input_feed):
        n = len(next(iter(input_feed.values())))
        if self._pending is None or n > self.pending_rows():
            raise RuntimeError(f"the model was asked for {n} windows, but results for {self.pending_rows()} are waiting")
        names = output_names or [o.name for o in self._outputs]
        rows = slice(self._offset, self._offset + n)
        self._offset += n
        empty = np.zeros((n, 0), np.float32)       # outputs nobody computed (bacpipe reads them but does not use them)
        return [self._pending[name][rows] if name in self._pending else empty for name in names]
