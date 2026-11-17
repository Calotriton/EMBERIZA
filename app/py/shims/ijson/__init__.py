"""ijson.parse, for the one question bacpipe asks with it: which top-level keys a JSON file has."""
import json


def parse(f):
    data = json.load(f)
    if isinstance(data, dict):
        for key in data:
            yield key, "map_key", None
