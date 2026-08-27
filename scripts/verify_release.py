"""Release-evidence verifier for SENTINEL-FL.

Runs deterministic, dependency-light release invariants and writes a SHA-256 evidence
manifest. This complements pytest: it verifies that the *release deliverable* is
internally coherent, reproducible, tamper-evident, and free of obvious committed secrets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import py_compile
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments" / "release_evidence.json"
sys.path.insert(0, str(ROOT))


class VerificationFailure(RuntimeError):
    pass


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise VerificationFailure(
            f"command failed ({' '.join(command)}):\n{result.stdout}\n{result.stderr}"
        )


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compile_all() -> dict:
    files = [
        p
        for p in ROOT.rglob("*.py")
        if not any(part in {".venv", "venv", "build", "dist", "__pycache__"} for part in p.parts)
    ]
    for path in files:
        py_compile.compile(str(path), doraise=True)
    return {"files_compiled": len(files)}


def _check_demo_results() -> dict:
    data = json.loads((ROOT / "experiments" / "demo_results.json").read_text())
    required = {"fedavg", "multikrum", "multikrum+guard"}
    if not required <= set(data):
        raise VerificationFailure(f"demo results missing strategies: {required - set(data)}")
    fed = float(data["fedavg"]["attack_success_rate"])
    defended = float(data["multikrum"]["attack_success_rate"])
    if not (fed > 0.5 and defended < fed):
        raise VerificationFailure(f"demo security invariant failed: fed={fed}, defended={defended}")
    return {"fedavg_asr": fed, "multikrum_asr": defended, "source_only_asr": True}


def _check_remediation() -> dict:
    data = json.loads((ROOT / "experiments" / "remediation_results.json").read_text())
    reports = data.get("reports", [])
    if len(reports) < 3:
        raise VerificationFailure("expected at least three remediation scenarios")
    for report in reports:
        if not report.get("success"):
            raise VerificationFailure(f"remediation failed: {report.get('scenario')}")
        if float(report["asr_after"]) > float(report["asr_threshold"]):
            raise VerificationFailure(f"ASR threshold missed: {report.get('scenario')}")
        if float(report["clean_accuracy_after"]) < float(report["clean_accuracy_before"]) - 0.1:
            raise VerificationFailure(f"clean utility regressed: {report.get('scenario')}")
    return {
        "scenarios": len(reports),
        "worst_asr_after": max(float(x["asr_after"]) for x in reports),
        "worst_clean_accuracy_after": min(float(x["clean_accuracy_after"]) for x in reports),
    }


def _check_attestation() -> dict:
    from ai.remediation.attestation import AttestationLedger

    path = ROOT / "experiments" / "attestation_chain.jsonl"
    ledger = AttestationLedger(path, secret_key="sentinel-fl-demo-key")
    if len(ledger) < 3 or not ledger.verify_chain():
        raise VerificationFailure("attestation chain is absent, short, or invalid")
    return {"certificates": len(ledger), "chain_valid": True, "ledger_sha256": _sha256(path)}


def _check_red_team() -> dict:
    path = ROOT / "experiments" / "red_team" / "red_team_results.json"
    data = json.loads(path.read_text())
    summary = data["summary"]
    if int(summary["scenario_count"]) < 8:
        raise VerificationFailure("red-team matrix has fewer than 8 scenarios")
    if float(summary["remediation_acceptance_rate"]) != 1.0:
        raise VerificationFailure("not all red-team remediations passed")
    if float(summary["remediation_asr"]["worst"]) > 0.10:
        raise VerificationFailure("post-remediation red-team ASR exceeds 0.10")
    return {
        "scenarios": int(summary["scenario_count"]),
        "acceptance_rate": float(summary["remediation_acceptance_rate"]),
        "worst_post_remediation_asr": float(summary["remediation_asr"]["worst"]),
        "worst_defended_asr": float(summary["defended_asr"]["worst"]),
    }


def _check_readme_assets() -> dict:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"(?:src=\"|\]\()([^\"\)]+)", readme)
    local = [x.split("#", 1)[0] for x in links if x and not x.startswith(("http", "#", "mailto:"))]
    missing = [x for x in local if not (ROOT / x).exists()]
    if missing:
        raise VerificationFailure(f"README contains missing local links/assets: {missing}")
    return {"local_links_checked": len(local), "missing": 0}


def _check_obvious_secrets() -> dict:
    patterns = {
        "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    hits: list[str] = []
    suffixes = {".py", ".yml", ".yaml", ".json", ".toml", ".md", ".ts", ".tsx"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in {".git", "node_modules", ".venv", "dist"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits.append(f"{name}: {path.relative_to(ROOT)}")
    if hits:
        raise VerificationFailure("possible committed secrets:\n" + "\n".join(hits))
    return {"patterns_checked": list(patterns), "findings": 0}


def _manifest_files() -> dict[str, str]:
    paths = [
        "README.md",
        "ARCHITECTURE.md",
        "NOVELTY.md",
        "MODEL_CARD.md",
        "SECURITY.md",
        "experiments/demo_results.json",
        "experiments/remediation_results.json",
        "experiments/attestation_chain.jsonl",
        "experiments/red_team/red_team_results.json",
        "experiments/red_team/RED_TEAM_REPORT.md",
        "assets/defense_stack.png",
        "assets/remediation_efficacy.png",
    ]
    missing = [x for x in paths if not (ROOT / x).exists()]
    if missing:
        raise VerificationFailure(f"release manifest files missing: {missing}")
    return {x: _sha256(ROOT / x) for x in paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-demos", action="store_true")
    args = parser.parse_args()

    if not args.skip_demos:
        _run([sys.executable, "scripts/run_demo.py"])
        _run([sys.executable, "scripts/run_remediation_demo.py"])
        # Keep an existing full matrix; generate the quick matrix only when absent.
        red = ROOT / "experiments" / "red_team" / "red_team_results.json"
        if not red.exists():
            _run([sys.executable, "scripts/run_red_team_matrix.py", "--quick"])

    checks: list[tuple[str, Callable[[], dict]]] = [
        ("compile", _compile_all),
        ("demo", _check_demo_results),
        ("remediation", _check_remediation),
        ("attestation", _check_attestation),
        ("red_team", _check_red_team),
        ("readme_assets", _check_readme_assets),
        ("secret_scan", _check_obvious_secrets),
    ]
    results: dict[str, dict] = {}
    for name, check in checks:
        results[name] = check()
        print(f"[PASS] {name}: {results[name]}")

    evidence = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "checks": results,
        "sha256_manifest": _manifest_files(),
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"[PASS] release evidence written to {EVIDENCE}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationFailure as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
