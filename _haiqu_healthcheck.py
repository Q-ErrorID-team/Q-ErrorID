import os
import sys
import time

key = os.environ.get("HAIQU_API_KEY")
if not key:
    print('HAIQU_API_KEY not set in this terminal. Run $env:HAIQU_API_KEY = "..." first.')
    sys.exit(1)

from haiqu.sdk import haiqu

haiqu.login(api_access_key=key, raise_on_error=True)

experiment_id = sys.argv[1] if len(sys.argv) > 1 else None
if experiment_id:
    print(f"activating experiment {experiment_id!r}...", flush=True)
    print(haiqu.init(experiment_id), flush=True)

from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

print("submitting a tiny 1-qubit healthcheck job to aer_simulator...", flush=True)
start = time.perf_counter()
job = haiqu.run(
    circuits=qc,
    shots=16,
    device_id="aer_simulator",
    job_name="Haiqu healthcheck v0.6",
)
print(f"JOB: {job.id}", flush=True)
print(f"STATUS right after submit: {job.retrieve_status()}", flush=True)

for i in range(6):
    time.sleep(10)
    elapsed = time.perf_counter() - start
    status = job.retrieve_status()
    print(f"[+{elapsed:5.1f}s] status={status}", flush=True)
    if str(status).upper().endswith("DONE") or str(status).upper().endswith("FAILED"):
        print("logs:", job.logs, flush=True)
        break
else:
    print("Still not finished after 60s on aer_simulator -- this points to a", flush=True)
    print("general Haiqu execution backlog, not something specific to our circuits.", flush=True)
