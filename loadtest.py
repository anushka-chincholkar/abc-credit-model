"""
Tiny concurrent load test for the /decision endpoint.

Usage:
  # start the server with a high rate limit for the test:
  RATE_LIMIT_PER_MIN=100000 .venv/bin/python -m uvicorn api:app --port 8000 &
  .venv/bin/python loadtest.py --url http://127.0.0.1:8000 --n 500 --concurrency 20
"""
from __future__ import annotations
import argparse, asyncio, time, statistics
import httpx

APPLICANT = {"applicant": {
    "Qualifications": "SSC", "Employment_Type": "SEP", "Age": 29, "Pincode": "302001",
    "Gender": "Male", "Product_Code": "SC", "Loan_Amount": 95000, "LTV": 84.0,
    "Resident_Type": "O", "Net_salary": 28000, "Final_Tier": "05 Urban", "Make_Code": "JUPITER",
    "Model_Description": "JUPITER 125 BSVI - DISC", "Model_Variant": "125 CC DISC",
    "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"}}


async def worker(client, url, headers, latencies, errors, sem):
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/decision", json=APPLICANT, headers=headers, timeout=10)
            latencies.append((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                errors.append(r.status_code)
        except Exception as e:
            errors.append(str(e)[:40])


async def main(url, n, concurrency, api_key):
    headers = {"X-API-Key": api_key} if api_key else {}
    latencies, errors = [], []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client, url, headers, latencies, errors, sem) for _ in range(n)])
        wall = time.perf_counter() - t0
    ok = len(latencies)
    print(f"requests={n}  concurrency={concurrency}  ok={ok}  errors={len(errors)}")
    print(f"throughput={n/wall:.0f} req/s  wall={wall:.2f}s")
    if latencies:
        latencies.sort()
        pct = lambda p: latencies[min(len(latencies)-1, int(p/100*len(latencies)))]
        print(f"latency ms: mean={statistics.mean(latencies):.1f}  p50={pct(50):.1f}  "
              f"p95={pct(95):.1f}  p99={pct(99):.1f}  max={latencies[-1]:.1f}")
    if errors:
        print("sample errors:", errors[:5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.n, a.concurrency, a.api_key))
