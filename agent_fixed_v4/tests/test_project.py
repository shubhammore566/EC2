import sys, io, zipfile
sys.path.insert(0, ".")
from github_agent.project import load_zip, analyze

def test_zip_project_analysis():
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w") as z:
        z.writestr("app.py","import streamlit as st")
        z.writestr("requirements.txt","streamlit\\n")
    files=load_zip(buf.getvalue())
    assert analyze(files)["framework"] == "Streamlit"
