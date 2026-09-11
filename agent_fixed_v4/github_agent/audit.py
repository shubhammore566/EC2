
"""In-session audit trail. Intentionally excludes credentials and secret values."""
from datetime import datetime, timezone
def record(store, request_id, action_id, tool, status, permission, risk, repository=None, error_code=None):
    store.append({
        "request_id":request_id,"action_id":action_id,"tool":tool,"status":status,
        "permission":permission,"risk":risk,"repository":repository,
        "timestamp":datetime.now(timezone.utc).isoformat(),"error_code":error_code
    })
    del store[:-200]
