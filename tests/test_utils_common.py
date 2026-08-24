import base64
import json

import pytest
from box import ConfigBox

from cnnClassifier.utils.common import (
    create_directories,
    decodeImage,
    encodeImageIntoBase64,
    get_size,
    load_json,
    read_yaml,
    save_json,
)


def test_read_yaml_returns_configbox(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("key: value\nnested:\n  inner: 1\n")

    result = read_yaml(yaml_path)

    assert isinstance(result, ConfigBox)
    assert result.key == "value"
    assert result.nested.inner == 1


def test_read_yaml_raises_on_empty_file(tmp_path):
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")

    with pytest.raises(ValueError):
        read_yaml(yaml_path)


def test_create_directories_creates_nested_dirs(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b" / "c"]

    create_directories([str(d) for d in dirs])

    assert all(d.is_dir() for d in dirs)


def test_save_json_then_load_json_roundtrip(tmp_path):
    json_path = tmp_path / "data.json"
    payload = {"accuracy": 0.99, "loss": 0.03}

    save_json(json_path, payload)
    loaded = load_json(json_path)

    assert json.loads(json_path.read_text()) == payload
    assert loaded.accuracy == payload["accuracy"]
    assert loaded.loss == payload["loss"]


def test_get_size_reports_kb(tmp_path):
    file_path = tmp_path / "file.bin"
    file_path.write_bytes(b"0" * 2048)

    assert get_size(file_path) == "~ 2 KB"


def test_decode_then_encode_image_roundtrip(tmp_path):
    original_bytes = b"not-really-an-image-but-good-enough-for-a-roundtrip"
    encoded = base64.b64encode(original_bytes)
    image_path = tmp_path / "inputImage.jpg"

    decodeImage(encoded, str(image_path))

    assert image_path.read_bytes() == original_bytes
    assert encodeImageIntoBase64(str(image_path)) == encoded
