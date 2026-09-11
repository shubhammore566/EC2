import sys
sys.path.insert(0, ".")
from github_agent.permission_policy import requires_approval, risk_for
from github_agent.security import safe_path, safe_repo_name, is_sensitive_path

def test_reads_do_not_need_approval():
    assert requires_approval("get_file") is False
    assert risk_for("get_file") == "low"

def test_writes_need_approval():
    assert requires_approval("update_file") is True

def test_destructive_is_high():
    assert risk_for("delete_repository") == "high"

def test_paths_block_traversal():
    assert safe_path("a/b.txt") == "a/b.txt"
    try: safe_path("../secret")
    except ValueError: pass
    else: assert False

def test_sensitive_files():
    assert is_sensitive_path(".env") and is_sensitive_path("certs/key.pem")

def test_repo_names():
    assert safe_repo_name("owner/repo") == "owner/repo"
