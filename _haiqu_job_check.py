import os
import sys

key = os.environ.get("HAIQU_API_KEY")
if not key:
    print('HAIQU_API_KEY not set in this terminal. Run $env:HAIQU_API_KEY = "..." first.')
    sys.exit(1)

from haiqu.sdk import haiqu

haiqu.login(api_access_key=key, raise_on_error=True)

experiment_id = sys.argv[1] if len(sys.argv) > 1 else None
job_id = sys.argv[2] if len(sys.argv) > 2 else None

if experiment_id:
    print(f"activating experiment {experiment_id!r}...", flush=True)
    print(haiqu.init(experiment_id), flush=True)

jobs = haiqu.list_jobs(limit=20, widget=False)
jobs = list(jobs) if jobs is not None else []
print(f"jobs found: {len(jobs)}", flush=True)

for job in jobs:
    if job_id and job.id != job_id:
        continue
    print("-" * 60, flush=True)
    print(f"id={job.id} type={type(job).__name__}", flush=True)
    try:
        status = job.retrieve_status()
        print(f"status={status}", flush=True)
    except Exception as exc:
        print(f"retrieve_status FAILED: {type(exc).__name__}: {exc}", flush=True)
    logs = getattr(job, "logs", None)
    print(f"logs tail: {logs[-2000:] if logs else 'No logs'}", flush=True)
