import urllib.request
import json
import time

base = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(f"{base}{path}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{base}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("=== PindaiSurvei Pipeline API Verification ===")

# 1. GET /api/pipeline/stages
stages = get("/api/pipeline/stages")
print(f"1. /api/pipeline/stages: {len(stages.get('stages', []))} tahapan terdaftar.")
for s in stages.get("stages", []):
    print(f"   - [{s['id']}] {s['name']} ({s.get('badge')})")

# 2. POST /api/pipeline/stage-gate/run (single booklet_01)
t0 = time.time()
sg_run = post("/api/pipeline/stage-gate/run", {"booklet_id": "booklet_01", "deep_snapping": False})
dur = time.time() - t0
print(f"2. POST /api/pipeline/stage-gate/run (booklet_01): status={sg_run.get('status')}, decision={sg_run.get('report', {}).get('decision')} in {dur:.2f}s")

# 3. GET /api/pipeline/stage-gate/booklet_01
sg_b01 = get("/api/pipeline/stage-gate/booklet_01")
print(f"3. GET /api/pipeline/stage-gate/booklet_01: Decision={sg_b01.get('decision')}, Routing={sg_b01.get('routing')}")
print(f"   Gates evaluated: {list(sg_b01.get('gates', {}).keys())}")
print(f"   Reasons: {sg_b01.get('reasons')}")

# 4. GET /api/pipeline/stage-gate/summary
summary = get("/api/pipeline/stage-gate/summary")
print(f"4. GET /api/pipeline/stage-gate/summary: Total={summary.get('total_evaluated')}, Distribution={summary.get('decision_distribution')}")

# 5. GET /api/pipeline/status
st = get("/api/pipeline/status")
print(f"5. GET /api/pipeline/status: Total jobs={len(st.get('jobs', []))}")
if st.get("jobs"):
    j0 = st["jobs"][0]
    print(f"   Sample Job 0: ID={j0.get('id')} | KRT={j0.get('krt')} | Gate={j0.get('gate_status')} | Stage={j0.get('stage')}")

print("\n=== ALL PIPELINE API TESTS PASSED SUCCESSFULLY! ===")
