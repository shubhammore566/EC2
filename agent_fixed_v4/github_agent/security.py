import re
SENSITIVE_FILES=(".env",".pem",".key",".p12",".pfx","id_rsa")
def is_sensitive_path(path):
    n=path.replace("\\\\","/").rsplit("/",1)[-1].lower(); return n in SENSITIVE_FILES or n.startswith(".env") or n.endswith((".pem",".key",".p12",".pfx"))
def safe_path(path):
    v=str(path).strip().replace("\\\\","/").lstrip("/"); p=[x for x in v.split("/") if x not in ("",".")]
    if not p or any(x==".." for x in p): raise ValueError("Invalid path")
    return "/".join(p)
def safe_repo_name(name):
    v=str(name).strip().strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?",v): raise ValueError("Invalid repository")
    return v
