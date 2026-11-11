"""Run the stage 3 circuits on a real IBM quantum processor (optional, needs an IBM Quantum account).

    python scripts/run_ibm_hardware.py submit [--backend ibm_torino] [--shots 4096]
    python scripts/run_ibm_hardware.py collect

The API key is read from the environment variable IBM_QUANTUM_TOKEN or from the file
~/.ibm/quantum_api_key. It is never written to the repository. `submit` records the job in
results/hardware/job.json; `collect` writes results/hardware/ibm_hardware.json once the job is
done. The free Open Plan is enough: the five circuits take a few seconds of machine time.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_extensions  # noqa: E402
from campaign import config, outputs, real_device  # noqa: E402

TARGET = ROOT / "results" / "hardware"
ORDER = ["stage3", "calibration_zeros", "calibration_ones", "correlated", "qaoa_p1"]


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    key_file = Path.home() / ".ibm" / "quantum_api_key"
    if not token and key_file.exists():
        token = key_file.read_text().strip()
    if not token:
        sys.exit("no API key: set IBM_QUANTUM_TOKEN or write the key to ~/.ibm/quantum_api_key")
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)


def submit(args) -> None:
    from qiskit_ibm_runtime import SamplerV2
    given = run_extensions.inputs(ROOT)
    svc = service()
    backend = svc.backend(args.backend) if args.backend else svc.least_busy(min_num_qubits=10, operational=True, simulator=False)
    circuits = real_device.logical_circuits(given["probabilities"], given["threshold"], given["stage2"], given["qaoa_runs"])
    device, facts = real_device.place_all(circuits, backend)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([device[name] for name in ORDER], shots=args.shots)
    record = {
        "job_id": job.job_id(),
        "backend": backend.name,
        "backend_qubits": int(backend.num_qubits),
        "shots_per_circuit": int(args.shots),
        "circuit_order": ORDER,
        "circuits": facts,
        "classical_registers": {name: device[name].cregs[0].name for name in ORDER},
        "transpiler_seed": int(config.TRANSPILE_SEEDS[0]),
        "optimization_level": 3,
        "error_suppression": "none requested (sampler defaults)",
    }
    outputs.write_json(TARGET / "job.json", record)
    print(f"submitted job {record['job_id']} to {backend.name}; run `collect` when it has finished")


def collect(args) -> None:
    record = json.loads((TARGET / "job.json").read_text())
    job = service().job(record["job_id"])
    status = str(job.status())
    if not job.in_final_state():
        print(f"job {record['job_id']} on {record['backend']} is not finished yet: {status}")
        return
    if not job.done():
        sys.exit(f"job {record['job_id']} ended without a result: {status} {job.error_message()}")
    result = job.result()
    counts = {}
    for index, name in enumerate(ORDER):
        register = record["classical_registers"][name]
        raw = getattr(result[index].data, register).get_counts()
        counts[name] = {key: int(raw[key]) for key in sorted(raw)}
    given = run_extensions.inputs(ROOT)
    summary = real_device.summarise(counts, given["probabilities"], given["threshold"], given["stage2"])
    try:
        quantum_seconds = float(job.usage())
    except Exception:                       # usage is informational; never lose the counts over it
        quantum_seconds = None
    out = {
        "job_id": record["job_id"],
        "backend": record["backend"],
        "backend_qubits": record["backend_qubits"],
        "shots_per_circuit": record["shots_per_circuit"],
        "created": str(job.creation_date),
        "quantum_seconds": quantum_seconds,
        "circuits": record["circuits"],
        "ids": given["ids"],
        **summary,
    }
    outputs.write_json(TARGET / "ibm_hardware.json", out)
    s3 = summary["stage3"]
    print(f"{record['backend']}: stage 3 unmitigated {s3['unmitigated']['estimate']:.4f}, "
          f"mitigated {s3['mitigated']['estimate']:.4f}, exact {s3['exact']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--backend", default=None, help="device name; default: the least busy one")
    s.add_argument("--shots", type=int, default=4096)
    sub.add_parser("collect")
    args = parser.parse_args()
    {"submit": submit, "collect": collect}[args.command](args)
