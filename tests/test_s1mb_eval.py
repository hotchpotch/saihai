"""Evaluation selects the Japanese source independently of the English receipt."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from saihai import s1mb_eval


@pytest.mark.parametrize("wrong", [None, "repo_id", "category", "revision"])
def test_japanese_receipt_and_offline_scope(tmp_path, monkeypatch, wrong):
    events = []
    receipt = {
        "repo_id": "hotchpotch/S1MB-japanese-v1",
        "category": "japanese-v1",
        "revision": "pinned",
    }
    if wrong:
        receipt[wrong] = "incorrect"
    sources = tmp_path / "datasets" / "hub-sources"
    sources.mkdir(parents=True)
    (sources / "japanese-v1.json").write_text(json.dumps(receipt))
    (sources.parent / "hub-source.json").write_text('{"revision": "english"}')
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").write_bytes(b"test")

    @contextmanager
    def session(root, **kwargs):
        events.append(kwargs)
        yield

    def category(root, name):
        events.append(name)
        return SimpleNamespace(benchmarks=[])

    monkeypatch.setitem(sys.modules, "s1mb", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "s1mb.data",
        SimpleNamespace(
            DATA_DIR=tmp_path,
            ModelInfo=object,
            Prediction=object,
            load_benchmark=None,
            load_category=category,
            validate_result=None,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "s1mb.dataset_source", SimpleNamespace(dataset_session=session)
    )
    monkeypatch.setitem(sys.modules, "s1mb.runner", SimpleNamespace(evaluate=None))
    monkeypatch.setattr("saihai.model.Saihai.from_pretrained", lambda path: object())
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--export",
            str(export),
            "--run-id",
            "test",
            "--output",
            str(output),
            "--data-revision",
            "pinned",
        ],
    )
    if wrong:
        with pytest.raises(ValueError, match="Installed"):
            s1mb_eval.main()
        assert not output.exists()
    else:
        s1mb_eval.main()
        summary = json.loads(output.read_text())
        assert summary["category"] == "japanese-v1"
        assert summary["data_source"] == receipt
        assert events[1] == "japanese-v1"
    assert events[0] == {"offline": True, "category": "japanese-v1"}
