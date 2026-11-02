"""Verify that the built executable contains the tested advanced source and weights."""
from pathlib import Path
import hashlib
import json
import types

from PyInstaller.archive.readers import CArchiveReader

ROOT = Path(__file__).resolve().parents[1]


def signature(code):
    if isinstance(code, types.CodeType):
        return (code.co_code, tuple(signature(c) for c in code.co_consts), code.co_names,
                code.co_varnames, code.co_freevars, code.co_cellvars, code.co_flags,
                code.co_argcount, code.co_kwonlyargcount)
    return code


def main():
    archive = CArchiveReader(str(ROOT / "dist/AutoGuardAI/AutoGuardAI.exe"))
    assert not (ROOT / "dist/AutoGuardAI/_internal/icuuc.dll").exists(), "Bundled ICU shadows the Windows API used by Qt"
    pyz = archive.open_embedded_archive("PYZ.pyz")
    modules = ["ai.federated_transport", "ai.active_diagnosis", "ai.advanced_service", "ai.graph_network", "ai.federation",
               "ai.multimodal", "ai.probabilistic", "ai.telemetry_learning", "ai.scenarios",
               "ai.consistency", "ai.pipeline", "app.state", "app.main_window",
               "app.pages.advanced_dialog", "app.pages.live_dialog", "app.pages.results_page",
               "database.repositories", "core.data_sources.obd_optional"]
    for name in modules:
        path = ROOT / (name.replace(".", "/") + ".py")
        expected = compile(path.read_bytes(), str(path), "exec", dont_inherit=True, optimize=0)
        if signature(expected) != signature(pyz.extract(name)):
            raise RuntimeError(f"Executable contains stale code: {name}")
    hashes = {}
    for name in ("graph.npz", "forecast.npz", "drivetrain/graph.npz", "drivetrain/forecast.npz"):
        source = ROOT / "models/advanced" / name
        bundled = ROOT / "dist/AutoGuardAI/_internal/models/advanced" / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(bundled.read_bytes()).hexdigest() == digest, name
        hashes[name] = digest
    result = {"ok": True, "verified_modules": modules, "model_sha256": hashes}
    (ROOT / "docs/advanced_bundle_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Verified {len(modules)} compiled modules and both bundled model files.")


if __name__ == "__main__":
    main()
