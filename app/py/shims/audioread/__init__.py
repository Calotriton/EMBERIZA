"""audioread.audio_open, for the one thing bacpipe asks of it: the duration of a file in seconds.

It reads the duration from the file header with soundfile: the number of frames over the sample rate, which is what
audioread reports for the same file.
"""
import soundfile as sf


class _Info:
    def __init__(self, path):
        info = sf.info(str(path))
        self.samplerate, self.channels = info.samplerate, info.channels
        self.duration = info.frames / info.samplerate

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def audio_open(path, backends=None):
    return _Info(path)


def available_backends():
    return []
