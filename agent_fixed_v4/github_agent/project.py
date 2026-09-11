import io, zipfile
from .security import is_sensitive_path
MAX_FILE=10*1024*1024
MAX_TOTAL=50*1024*1024
def safe_zip_path(name):
    name=name.replace("\\\\","/").lstrip("/")
    parts=[p for p in name.split("/") if p not in ("",".")]
    return None if not parts or any(p==".." for p in parts) else "/".join(parts)
def load_zip(raw):
    out=[]; total=0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            path=safe_zip_path(info.filename)
            if not path: continue
            data=z.read(info)
            if len(data)>MAX_FILE: raise ValueError(f"{path} exceeds 10 MB")
            total += len(data)
            if total>MAX_TOTAL: raise ValueError("Project exceeds 50 MB")
            out.append({"path":path,"content":data,"sensitive":is_sensitive_path(path)})
    return out
def analyze(files):
    names=[x["path"] for x in files]; low=" ".join(names).lower()
    if "streamlit" in low or any(x.endswith("app.py") for x in names): framework="Streamlit"
    elif "fastapi" in low: framework="FastAPI"
    elif "flask" in low: framework="Flask"
    elif "manage.py" in low: framework="Django"
    elif "package.json" in low: framework="Node.js"
    else: framework="Unknown"
    entry=next((x for x in names if x.endswith("app.py")),None)
    return {"framework":framework,"entrypoint":entry,"file_count":len(files),"has_dockerfile":any(x.lower()=="dockerfile" for x in names),"has_requirements":any(x.lower().endswith("requirements.txt") for x in names),"recommended_platform":"Streamlit Cloud" if framework=="Streamlit" else "Render","sensitive_files":[x for x in names if is_sensitive_path(x)]}
