"""
GitHub Agent - simple Streamlit app

Run:
    pip install -r requirements.txt
    streamlit run app.py

The app keeps API keys and the GitHub token only in Streamlit session state.
It never writes them to a file.
"""

import io
import json
import html
import hashlib
import os
import re
import tempfile
import time
import zipfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import streamlit as st
from github import Github, GithubException

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    AWS_SDK_READY = True
except ImportError:
    boto3 = None
    BotoCoreError = ClientError = Exception
    AWS_SDK_READY = False


st.set_page_config(page_title="GitHub Agent", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.main-hero{padding:22px 26px;border:1px solid rgba(120,120,160,.25);border-radius:18px;margin-bottom:18px;background:linear-gradient(135deg,rgba(91,76,210,.22),rgba(23,170,150,.16))}
.main-hero h1{margin:0 0 6px 0;font-size:34px}.main-hero p{margin:0;opacity:.78}
</style>
<div class="main-hero"><h1>🤖 GitHub Agent</h1><p>Build → GitHub → Docker → AWS EC2 → Live URL, with approval before every write/deploy action.</p></div>
""", unsafe_allow_html=True)

if not AWS_SDK_READY:
    st.error(
        "AWS deployment module is not installed in this environment. "
        "Run `python -m pip install -r requirements.txt` or start with `run_app.bat`. "
        "The current Agent requires boto3 for AWS EC2 deployment."
    )

PROVIDERS = {
    "OpenAI": {
        "kind": "openai",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        # "chat-latest" is OpenAI's own rolling alias for its current default
        # chat model, so this default doesn't go stale when OpenAI ships a
        # new generation.
        "model": "chat-latest",
        "help": "OpenAI API key",
    },
    "Mistral": {
        "kind": "openai",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "help": "Mistral API key",
    },
    "Anthropic": {
        "kind": "anthropic",
        "model": "claude-sonnet-5",
        "help": "Anthropic API key",
    },
    "Google Gemini": {
        "kind": "gemini",
        "model": "gemini-3.6-flash",
        "help": "Google AI Studio API key",
    },
    "Groq": {
        "kind": "openai",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        # Groq retired the Llama 3.x text models; gpt-oss-20b is their
        # current recommended fast/cheap replacement.
        "model": "openai/gpt-oss-20b",
        "help": "Groq API key",
    },
    "QwenCloud (Qwen / DeepSeek)": {
        "kind": "openai",
        "endpoint": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        # Default to a Qwen model since its free quota is shared across the
        # whole account; DeepSeek models on QwenCloud each have their own
        # separate free-quota bucket that can run out independently (change
        # the model field to e.g. "deepseek-v4-flash" if you prefer that).
        "model": "qwen3.8-flash",
        "help": "QwenCloud API key (from the dashboard's API Keys page, not the model page)",
    },
    "OpenRouter": {
        "kind": "openai",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-chat-latest",
        "help": "OpenRouter API key",
    },
    "Custom OpenAI-compatible": {
        "kind": "custom",
        "model": "gpt-4o-mini",
        "help": "Any OpenAI-compatible endpoint",
    },
}

DEPLOYMENT_PLATFORM_OPTIONS = (
    "Streamlit Cloud",
    "AWS EC2",
    "Azure",
    "Google Cloud",
    "Render",
)
ASK_DEPLOYMENT_PLATFORM = "Ask me on deploy"
DEPLOYMENT_PLATFORM_ALIASES = {
    "streamlit cloud": "Streamlit Cloud",
    "streamlit": "Streamlit Cloud",
    "streamlit community cloud": "Streamlit Cloud",
    "aws": "AWS EC2",
    "amazon web services": "AWS EC2",
    "aws ec2": "AWS EC2",
    "ec2": "AWS EC2",
    "amazon ec2": "AWS EC2",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "google cloud": "Google Cloud",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "render": "Render",
}

# Models we should never auto-pick as a fallback chat model (image/audio/
# embedding/moderation models return valid /models entries but can't do
# chat completions).
NON_CHAT_MODEL_KEYWORDS = (
    "embed",
    "embedding",
    "tts",
    "audio",
    "speech",
    "whisper",
    "image",
    "vision-only",
    "moderation",
    "dall-e",
    "video",
    "veo",
    "imagen",
    "rerank",
)

# When auto-picking a replacement model, prefer small/fast/cheap ones first
# so recovery doesn't silently switch someone onto an expensive model.
PREFERRED_FALLBACK_KEYWORDS = (
    "flash",
    "mini",
    "haiku",
    "small",
    "8b",
    "20b",
    "instant",
    "lite",
    "chat-latest",
    "sonnet",
)

SYSTEM_PROMPT = """
You are a friendly GitHub Agent inside a Streamlit app.

The user may speak in English, Hindi, Hinglish, or another language. Reply in
the same language and style. Understand natural language; do not require
fixed commands, exact keywords, or a rigid syntax — infer intent the way a
sharp human assistant would, including from typos, shorthand, or casual
phrasing (e.g. "deploye kr", "isko github pe daal do", "ye wala push kr do").

You will also receive a separate system message titled "Current app state"
before each request. Always read it and use it to fill in details the user
left implicit — for example if they say "deploy kar do" and the state shows
a recently pushed repository or an already-open repository, use that
repository instead of asking the user to repeat it. Only ask a clarifying
question when the state truly gives no reasonable way to infer what is
meant (e.g. multiple plausible repos and none recently used, or a
genuinely destructive action with real ambiguity about the target).

Return ONLY valid JSON, with this exact shape:
{
  "reply": "short helpful response in the user's language",
  "action": null
}

For a GitHub action, action must be an object with one of these types:
- {"type":"list_repos"}
- {"type":"analyze_project"}
- {"type":"open_repo","repo":"owner/repo"}
- {"type":"create_repo","name":"...","private":false,"description":"..."}
- {"type":"create_repo","name":"...","owner":"optional-org","private":false,"description":"..."}
- {"type":"rename_repo","repo":"owner/repo","new_name":"..."}
- {"type":"create_issue","repo":"owner/repo","title":"...","body":"..."}
- {"type":"list_issues","repo":"owner/repo"}
- {"type":"update_issue","repo":"owner/repo","number":1,"title":"optional","body":"optional","state":"open"}
- {"type":"close_pull_request","repo":"owner/repo","number":1}
- {"type":"read_file","repo":"owner/repo","path":"path/to/file"}
- {"type":"update_file","repo":"owner/repo","path":"...","content":"..."}
- {"type":"delete_file","repo":"owner/repo","path":"..."}
- {"type":"create_branch","repo":"owner/repo","branch":"...","from_branch":"main"}
- {"type":"create_pull_request","repo":"owner/repo","title":"...","head":"...","base":"main","body":"..."}
- {"type":"merge_pull_request","repo":"owner/repo","number":1,"commit_message":"..."}
- {"type":"close_issue","repo":"owner/repo","number":1}
- {"type":"delete_repo","repo":"owner/repo"}
- {"type":"list_releases","repo":"owner/repo"}
- {"type":"create_release","repo":"owner/repo","tag":"v1.0.0","name":"...","body":"..."}
- {"type":"delete_release","repo":"owner/repo","release_id":1}
- {"type":"list_workflows","repo":"owner/repo"}
- {"type":"repo_settings","repo":"owner/repo"}
- {"type":"update_repo_settings","repo":"owner/repo","changes":{"private":true,"description":"...","default_branch":"...","has_issues":true,"has_wiki":true,"has_projects":true,"delete_branch_on_merge":true,"topics":["..."]}}
- {"type":"run_workflow","repo":"owner/repo","workflow":"ci.yml","ref":"main","inputs":{}}
- {"type":"deploy_streamlit","repo":"owner/repo","branch":"main","app_path":"app.py"}
- {"type":"deploy_aws_ec2","repo":"owner/repo","port":"","ref":"main"}
- {"type":"prepare_deployment","repo":"owner/repo","platform":"Azure|Google Cloud|Render","branch":"main"}

For a project upload request use:
{"type":"push_project","repo_name":"","private":false,"commit_message":"..."}

When the user asks you to build, create, or generate a brand-new project from
scratch (they are not uploading an existing project), for example "streamlit
project banavo" or "create a streamlit app that does X", use:
{"type":"generate_project","framework":"streamlit","description":"a detailed spec of exactly what the app should do, based on the whole conversation","repo_name":"optional-repo-name-if-the-user-gave-one"}
The app itself writes the code, creates the files, pushes them to GitHub, and
prepares a deployment link, the same way you would write Streamlit code
yourself. Do not write the project code inside "reply"; only describe the
requested app in "description" and let the app generate and handle the files.

When the user asks to open, show, or go to a repository's settings (in
English, Hindi or Hinglish, e.g. "settings pr jao", "repo settings dikhao"),
use repo_settings with the repository the user means, falling back to the
most recently opened repository if none is named. Only use
update_repo_settings when the user asks to actually change something
(private/public, description, issues/wiki/projects on or off, default
branch, delete-branch-on-merge, topics); only include the fields the user
actually asked to change inside "changes".
Never claim that a GitHub action completed unless the app performs it. The app
always asks for a repository name before project pushing, so do not invent or
assume a repository name.
When the user says open, show, or display a repository, use open_repo instead
of read_file. If the user gives only a short name such as "cancer", pass that
short name; the app resolves it for the connected GitHub account.
For write or destructive actions, return the action but do not claim it is
complete. The app will show a preview and ask the user to confirm.
For deployment requests, use the project analysis to recommend one of these
platforms: Streamlit Cloud, AWS EC2, Azure, Google Cloud, or Render. When the
user says AWS, EC2, Amazon server, or deploy on the EC2 instance, use
`deploy_aws_ec2`. AWS EC2 deployment is a direct Agent-controlled deployment through AWS Systems Manager
(SSM) to the configured EC2 instance. The Agent clones/pulls the GitHub repository,
creates a Streamlit Dockerfile when needed, builds the Docker image on EC2, chooses
an available host port, starts/replaces the container, opens that port in the
instance security group when permitted, and reads the real public IP/hostname and
port from AWS before returning the live URL. Never claim the URL before the AWS
command has completed and the URL has been verified. The Agent must ask for explicit
user approval before deployment. If the user has not clearly chosen a platform, ask
them to choose from the five options. For Streamlit Cloud, prepare the setup link;
do not claim deployment is complete. For Azure, Google Cloud, and Render, prepare
a platform-specific plan unless a matching direct workflow is available.
"""



def emit_action_event(event: str, status: str, title: str, message: str,
                      action: Optional[Dict[str, Any]] = None,
                      step: int = 1, total_steps: int = 1,
                      data: Optional[Dict[str, Any]] = None) -> None:
    """Append a typed live event. Secrets are never placed in event payloads."""
    action = action or {}
    payload = {
        "request_id": st.session_state.get("active_request_id") or uuid.uuid4().hex,
        "action_id": st.session_state.get("active_action_id") or uuid.uuid4().hex,
        "event": event,
        "status": status,
        "step": step,
        "total_steps": total_steps,
        "title": title,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data or {},
    }
    st.session_state.action_events.append(payload)
    st.session_state.action_events = st.session_state.action_events[-100:]

def set_live_view(kind: str, data: Dict[str, Any]) -> None:
    """Update the right-side 'Live GitHub View' panel snapshot.

    This never stores secrets - only display data (names, paths, states,
    urls) that mirrors what a human would see on github.com while doing the
    same action by hand.
    """
    st.session_state.live_view = {
        "kind": kind,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def action_permission(action: Dict[str, Any]) -> Dict[str, str]:
    t = str(action.get("type", ""))
    read = {
        "list_repos": ("Metadata", "Read", "low"),
        "open_repo": ("Metadata", "Read", "low"),
        "read_file": ("Contents", "Read", "low"),
        "list_issues": ("Issues", "Read", "low"),
        "list_releases": ("Metadata", "Read", "low"),
        "list_workflows": ("Actions", "Read", "low"),
        "analyze_project": ("Contents", "Read", "low"),
        "repo_settings": ("Administration", "Read", "low"),
    }
    if t in read:
        scope, access, risk = read[t]
    elif t == "update_repo_settings":
        scope, access, risk = ("Administration", "Write", "high")
    elif t in {"run_workflow", "deploy_aws"}:
        scope, access, risk = ("Actions", "Write", "medium")
    elif t == "deploy_aws_ec2":
        scope, access, risk = ("AWS EC2 + SSM", "Deploy", "high")
    elif t in {"create_issue", "update_issue", "close_issue"}:
        scope, access, risk = ("Issues", "Write", "medium")
    elif t in {"create_pull_request", "close_pull_request"}:
        scope, access, risk = ("Pull requests", "Write", "medium")
    elif t == "merge_pull_request":
        scope, access, risk = ("Pull requests", "Write", "high")
    elif t in {"update_file", "delete_file", "push_project"}:
        scope, access, risk = ("Contents", "Write", "high" if t == "delete_file" else "medium")
    elif t == "create_branch":
        scope, access, risk = ("Contents", "Write", "medium")
    elif t in {"create_repo", "rename_repo", "delete_repo"}:
        scope, access, risk = ("Administration", "Write", "high")
    elif t in {"create_release", "delete_release"}:
        scope, access, risk = ("Contents", "Write", "high" if t == "delete_release" else "medium")
    elif t == "prepare_deployment":
        scope, access, risk = ("Actions", "Write", "medium")
    else:
        scope, access, risk = ("GitHub API", "Write", "high")
    return {"scope": scope, "access": access, "risk": risk}

def approval_description(action: Dict[str, Any]) -> str:
    p = action_permission(action)
    repo = action.get("repo") or action.get("repo_name") or "—"
    affected = action.get("path") or action.get("workflow") or action.get("number") or "repository resource"
    return (
        f"**Action:** `{action.get('type','unknown')}`  \n"
        f"**Repository:** `{repo}`  \n"
        f"**Affected:** `{affected}`  \n"
        f"**Required permission:** **{p['scope']}: {p['access']}**  \n"
        f"**Risk:** **{p['risk'].upper()}**  \n"
        f"**Reason:** GitHub must authorize this operation; the Agent will not bypass token permissions."
    )

def init_state() -> None:
    defaults = {
        "active_request_id": None,
        "active_action_id": None,
        "gh_client": None,
        "gh_user": None,
        "project_files": [],
        "project_zip_name": None,
        "uploaded_signature": None,
        "project_analysis": None,
        "last_push_report": None,
        "pending_push": False,
        "pending_confirmation": None,
        "last_pushed_repo": None,
        "auto_deploy_after_push": False,
        "deployment_platform": ASK_DEPLOYMENT_PLATFORM,
        "ec2_runner_label": os.getenv("EC2_RUNNER_LABEL", "ec2-deployer"),
        "ec2_default_port": "",
        "aws_region": os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
        "aws_ec2_instance_id": os.getenv("EC2_INSTANCE_ID", ""),
        "aws_security_group_id": os.getenv("EC2_SECURITY_GROUP_ID", ""),
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
        "aws_session_token": "",
        "pending_deployment": None,
        "provider": "OpenAI",
        "api_key": "",
        "model": PROVIDERS["OpenAI"]["model"],
        "custom_endpoint": "",
        "private_repo": True,
        "commit_message": "Initial commit via GitHub Agent",
        "action_events": [],
        "action_history": [],
        "live_view": None,
        "current_repo": None,
        "github_oauth_client_id": "",
        "github_oauth_client_secret": "",
        "github_oauth_redirect_uri": "http://localhost:8501",
        "github_oauth_state": uuid.uuid4().hex,
        "github_oauth_error": None,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Namaste! Main aapka GitHub Agent hoon.\n\n"
                    "Aap mujhse English, Hindi, Hinglish ya kisi bhi "
                    "language mein baat kar sakte ho. Sidebar mein AI API "
                    "key, GitHub token aur project ZIP set karke bas batao "
                    "kya karna hai.\n\n"
                    "Right side ke **Live GitHub View** tab mein aap dekhoge "
                    "ki main GitHub par kya kar raha hoon — repo, settings, "
                    "issues, releases, sab kuch real-time."
                ),
            }
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_provider_model() -> None:
    st.session_state.model = PROVIDERS[st.session_state.provider]["model"]


def selected_model() -> str:
    provider = PROVIDERS[st.session_state.provider]
    model = str(st.session_state.model).strip()
    # A common setup mistake is pasting an environment-variable name or API
    # key into the model field. Recover with the provider's known-good default.
    if not model or "API_KEY" in model.upper() or model.startswith(("sk-", "key-")):
        model = provider["model"]
        st.session_state.model = model
    return model


def safe_zip_path(name: str) -> Optional[str]:
    normalized = name.replace("\\", "/").lstrip("/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


MAX_UPLOAD_FILE_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 50 * 1024 * 1024


def collect_zip_files(
    raw_zip: bytes,
    files: List[Dict[str, Any]],
    total_bytes: int,
    prefix: str = "",
    depth: int = 0,
) -> int:
    """Extract a ZIP into memory, including nested ZIPs up to two levels."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = safe_zip_path(info.filename)
            if not path:
                continue
            content = archive.read(info)
            if (
                len(content) > MAX_UPLOAD_FILE_BYTES
                and not path.lower().endswith(".zip")
            ):
                raise ValueError(f"`{path}` 10 MB se badi file hai.")
            full_path = f"{prefix}/{path}".strip("/")
            if path.lower().endswith(".zip") and depth < 2:
                try:
                    total_bytes = collect_zip_files(
                        content,
                        files,
                        total_bytes,
                        prefix=full_path[:-4].rstrip("/"),
                        depth=depth + 1,
                    )
                    continue
                except zipfile.BadZipFile:
                    # A non-archive .zip is still a valid uploaded file.
                    pass
            total_bytes += len(content)
            if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                raise ValueError("Uploaded files ka total size 50 MB se zyada hai.")
            files.append({"path": full_path, "content": content})
    return total_bytes


def load_uploaded_files(uploaded_files: List[Any]) -> None:
    try:
        files: List[Dict[str, Any]] = []
        total_bytes = 0
        upload_names: List[str] = []
        for uploaded_file in uploaded_files:
            raw = uploaded_file.getvalue()
            upload_names.append(uploaded_file.name)
            safe_name = safe_zip_path(uploaded_file.name)
            if not safe_name:
                continue
            is_archive = safe_name.lower().endswith(".zip")
            if len(raw) > MAX_UPLOAD_TOTAL_BYTES:
                raise ValueError(f"`{uploaded_file.name}` 50 MB se badi file hai.")
            if len(raw) > MAX_UPLOAD_FILE_BYTES and not is_archive:
                raise ValueError(f"`{uploaded_file.name}` 10 MB se badi file hai.")
            if is_archive:
                try:
                    total_bytes = collect_zip_files(raw, files, total_bytes)
                    continue
                except zipfile.BadZipFile:
                    pass
            total_bytes += len(raw)
            if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                raise ValueError("Uploaded files ka total size 50 MB se zyada hai.")
            files.append({"path": safe_name, "content": raw})

        if not files:
            raise ValueError("Upload mein readable files nahi mili.")
        st.session_state.project_files = files
        st.session_state.project_zip_name = ", ".join(upload_names)
        st.session_state.uploaded_signature = hashlib.sha256(
            b"".join(
                uploaded_file.getvalue()
                for uploaded_file in uploaded_files
            )
        ).hexdigest()
        st.session_state.project_analysis = analyze_project_files(files)
        st.session_state.last_push_report = None
        st.session_state.pending_push = False
        st.success(
            f"{len(files)} files loaded ({len(upload_names)} upload"
            f"{'s' if len(upload_names) != 1 else ''})"
        )
    except (zipfile.BadZipFile, ValueError) as exc:
        st.error(f"Upload load nahi ho paya: {exc}")
    except Exception as exc:
        st.error(f"Files read karne mein error: {exc}")


def analyze_project_files(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspect project metadata without executing uploaded code."""
    names = [str(item["path"]) for item in files]
    lowered_names = [name.lower() for name in names]
    text_parts: List[str] = []
    for item in files:
        path = str(item["path"]).lower()
        if path.endswith((".py", ".txt", ".toml", ".yaml", ".yml", ".md")):
            text_parts.append(bytes(item["content"])[:200_000].decode("utf-8", errors="ignore").lower())
    combined = "\n".join(text_parts)

    framework = "Unknown"
    if "streamlit" in combined:
        framework = "Streamlit"
    elif "fastapi" in combined:
        framework = "FastAPI"
    elif "flask" in combined:
        framework = "Flask"
    elif "django" in combined or "manage.py" in lowered_names:
        framework = "Django"
    elif "package.json" in lowered_names or "express" in combined:
        framework = "Node.js"

    preferred = {
        "Streamlit": ("app.py", "streamlit_app.py", "main.py"),
        "FastAPI": ("main.py", "app.py", "server.py"),
        "Flask": ("app.py", "main.py", "server.py"),
        "Django": ("manage.py", "wsgi.py", "asgi.py"),
        "Node.js": ("server.js", "index.js", "main.js", "app.js"),
    }
    python_files = [
        item["path"] for item in files if str(item["path"]).lower().endswith(".py")
    ]
    javascript_files = [
        item["path"]
        for item in files
        if str(item["path"]).lower().endswith((".js", ".mjs", ".cjs", ".ts"))
    ]
    entrypoint_files = python_files + javascript_files
    entrypoint = next(
        (
            path
            for candidate in preferred.get(framework, ())
            for path in entrypoint_files
            if path.rsplit("/", 1)[-1].lower() == candidate
        ),
        entrypoint_files[0] if entrypoint_files else None,
    )
    dependency_file = next(
        (
            path
            for path in lowered_names
            if path.rsplit("/", 1)[-1] in {
                "requirements.txt",
                "pyproject.toml",
                "poetry.lock",
                "package.json",
                "pnpm-lock.yaml",
                "yarn.lock",
            }
        ),
        None,
    )
    dockerfile = next(
        (path for path in names if path.rsplit("/", 1)[-1].lower() == "dockerfile"),
        None,
    )
    procfile = next(
        (path for path in names if path.rsplit("/", 1)[-1].lower() == "procfile"),
        None,
    )
    workflow_files = [
        path for path in names if path.lower().startswith(".github/workflows/")
    ]
    has_frontend = any(
        marker in combined
        for marker in ("react", "next", "vite", "vue", "angular")
    ) or "package.json" in lowered_names
    if framework == "Unknown" and has_frontend:
        framework = "Node.js / frontend"

    analysis = {
        "framework": framework,
        "entrypoint": entrypoint,
        "dependency_file": dependency_file,
        "file_count": len(files),
        "dockerfile": dockerfile,
        "procfile": procfile,
        "workflow_files": workflow_files,
        "has_frontend": has_frontend,
        "has_python": bool(python_files),
        "has_node": bool(javascript_files) or "package.json" in lowered_names,
    }
    recommendation = recommend_deployment_platform(analysis)
    analysis["recommended_platform"] = recommendation["platform"]
    analysis["recommendation_reason"] = recommendation["reason"]
    return analysis


def recommend_deployment_platform(analysis: Dict[str, Any]) -> Dict[str, str]:
    """Recommend a deployment target from detected project characteristics."""
    framework = str(analysis.get("framework") or "Unknown").lower()
    has_docker = bool(analysis.get("dockerfile"))
    has_workflows = bool(analysis.get("workflow_files"))

    if "streamlit" in framework:
        return {
            "platform": "Streamlit Cloud",
            "reason": "Streamlit app detected; it is the simplest fit and needs minimal deployment setup.",
        }
    if has_docker:
        return {
            "platform": "Render",
            "reason": "A Dockerfile is present, so Render can run the project with a straightforward web service setup.",
        }
    if any(name in framework for name in ("fastapi", "flask", "django", "node")):
        return {
            "platform": "Render",
            "reason": f"{analysis.get('framework')} web project detected; Render is the quickest low-configuration option.",
        }
    if has_workflows:
        return {
            "platform": "AWS",
            "reason": "Existing GitHub Actions workflows suggest a CI/CD-oriented deployment; AWS is a strong production choice.",
        }
    if analysis.get("has_frontend"):
        return {
            "platform": "Render",
            "reason": "A frontend/Node project was detected; Render provides a simple build-and-deploy path.",
        }
    return {
        "platform": "Render",
        "reason": "The project type is not fully identifiable yet; Render is the safest general-purpose starting point.",
    }


def deployment_platform_prompt(analysis: Optional[Dict[str, Any]]) -> str:
    if not analysis:
        return (
            "Deployment se pehle project ZIP upload karo taaki main files scan karke "
            "best platform suggest kar sakoon."
        )
    recommended = analysis.get("recommended_platform", "Render")
    reason = analysis.get(
        "recommendation_reason",
        "Project ke liye general-purpose deployment option.",
    )
    options = "\n".join(
        f"{index}. **{platform}**"
        for index, platform in enumerate(DEPLOYMENT_PLATFORM_OPTIONS, start=1)
    )
    return (
        f"Project files analyze karne ke baad meri recommendation: **{recommended}** — {reason}\n\n"
        "Aap kaunsa deployment platform use karna chahte ho? Number ya naam reply karo:\n"
        f"{options}\n\n"
        "AWS EC2 chahiye to **AWS** ya **2** reply karo. Uske baad Agent deployment permission maangega."
    )


def gh_analyze_repo(repo_name: str) -> Dict[str, Any]:
    """Read repository metadata for deployment advice without executing code."""
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    files: List[Dict[str, Any]] = []
    relevant_suffixes = (
        ".py",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".md",
    )

    def walk(path: str = "", depth: int = 0) -> None:
        if depth > 3 or len(files) >= 150:
            return
        entries = repo.get_contents(path, ref=repo.default_branch)
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            if len(files) >= 150:
                return
            if entry.type == "dir" and (
                not entry.name.startswith(".") or entry.name == ".github"
            ):
                walk(entry.path, depth + 1)
            elif entry.type == "file":
                content = b""
                if entry.path.lower().endswith(relevant_suffixes) and entry.size <= 250_000:
                    content = entry.decoded_content
                files.append({"path": entry.path, "content": content})

    walk()
    if not files:
        return {
            "framework": "Unknown",
            "entrypoint": None,
            "dependency_file": None,
            "file_count": 0,
        }
    return analyze_project_files(files)


def project_analysis_text(analysis: Optional[Dict[str, Any]]) -> str:
    if not analysis:
        return "Pehle project ZIP upload karo."
    return (
        "**Project analysis complete:**\n"
        f"- Framework: `{analysis.get('framework', 'Unknown')}`\n"
        f"- Entrypoint: `{analysis.get('entrypoint') or 'Not detected'}`\n"
        f"- Dependency file: `{analysis.get('dependency_file') or 'Not detected'}`\n"
        f"- Files scanned: `{analysis.get('file_count', 0)}`\n"
        f"- Dockerfile: `{'Detected' if analysis.get('dockerfile') else 'Not detected'}`\n"
        f"- Recommended deployment: **{analysis.get('recommended_platform', 'Render')}**\n"
        f"- Why: {analysis.get('recommendation_reason', 'Project ke liye general-purpose option.')}"
    )


def github_error(exc: Exception) -> str:
    if isinstance(exc, GithubException):
        data = exc.data if isinstance(exc.data, dict) else {}
        message = str(data.get("message", str(exc)))
        if exc.status in (401, 403) and (
            "resource not accessible" in message.lower()
            or "bad credentials" in message.lower()
            or "permission" in message.lower()
        ):
            return (
                f"{message} (HTTP {exc.status}). GitHub token valid ho sakta hai, "
                "lekin is action ki permission missing hai. Personal repo create ke "
                "liye classic PAT mein `repo` scope use karo; delete ke liye "
                "`delete_repo` bhi chahiye. Fine-grained PAT mein sahi resource "
                "owner select karke Administration/Contents/Issues/Pull requests "
                "permissions do, phir naya token reconnect karo. Organization ki "
                "policy bhi repo creation/deletion rok sakti hai."
            )
        return f"{message} (HTTP {exc.status})" if exc.status else message
    return str(exc)


def gh_push_project(repo_name: str, private: bool, commit_message: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", repo_name):
        return "Repo name invalid hai. Sirf letters, numbers, `.`, `_` aur `-` use karo."
    if not st.session_state.project_files:
        return "Pehle sidebar mein apni project files ya ZIP upload karo."
    if st.session_state.gh_client is None or st.session_state.gh_user is None:
        return "Pehle sidebar se GitHub connect karo."

    client = st.session_state.gh_client
    user = st.session_state.gh_user
    full_name = f"{user.login}/{repo_name}"

    try:
        repo = client.get_repo(full_name)
        created = False
    except GithubException as exc:
        # Only a real 404 means the repository is absent. A 401/403 must not
        # fall through to create_repo because that hides the actual permission
        # problem and produces a misleading "create repository" error.
        if exc.status != 404:
            return f"Existing repository check fail hui: {github_error(exc)}"
        try:
            repo = user.create_repo(repo_name, private=private)
            created = True
        except Exception as create_exc:
            return f"Repository create nahi ho paayi: {github_error(create_exc)}"

    created_count = 0
    updated_count = 0
    created_paths: List[str] = []
    updated_paths: List[str] = []
    failures: List[str] = []
    for project_file in st.session_state.project_files:
        path = project_file["path"]
        content = project_file["content"]
        try:
            try:
                existing = repo.get_contents(path)
                if isinstance(existing, list):
                    raise GithubException(400, {"message": "Path is a directory"})
                repo.update_file(path, commit_message, content, existing.sha)
                updated_count += 1
                updated_paths.append(path)
            except GithubException as exc:
                # Only a 404 means the file is new. Do not turn 401/403,
                # rate-limit, or conflict errors into a misleading create call.
                if exc.status != 404:
                    raise
                repo.create_file(path, commit_message, content)
                created_count += 1
                created_paths.append(path)
        except Exception as exc:
            failures.append(f"{path}: {github_error(exc)}")

    result = (
        f"{'Naya repo bana aur ' if created else 'Existing repo mein '}"
        f"project push ho gaya.\n\n"
        f"**Repository:** [{repo.full_name}]({repo.html_url})\n"
        f"- New files: {created_count}\n"
        f"- Updated files: {updated_count}"
    )
    st.session_state.last_pushed_repo = repo.full_name
    st.session_state.last_push_report = {
        "repo": repo.full_name,
        "created": created_paths,
        "updated": updated_paths,
        "failed": failures,
    }
    if created_paths:
        result += "\n\n**Created files:**\n" + "\n".join(
            f"- ✅ `{path}`" for path in created_paths
        )
    if updated_paths:
        result += "\n\n**Updated files:**\n" + "\n".join(
            f"- 🔄 `{path}`" for path in updated_paths
        )
    if failures:
        result += "\n\n**Failed files:**\n" + "\n".join(
            f"- ❌ `{failure}`" for failure in failures
        )
    return result


def gh_list_repos() -> str:
    user = st.session_state.gh_user
    repos = list(user.get_repos())[:30]
    if not repos:
        set_live_view("repos_list", {"owner": user.login, "repos": []})
        return "Aapke account mein koi repository nahi mili."
    set_live_view("repos_list", {
        "owner": user.login,
        "repos": [
            {
                "full_name": repo.full_name,
                "private": repo.private,
                "description": repo.description or "",
                "stars": repo.stargazers_count,
                "html_url": repo.html_url,
            }
            for repo in repos
        ],
    })
    return "\n".join(
        f"- {repo.full_name}{' (private)' if repo.private else ''}" for repo in repos
    )


def resolve_repo_name(repo_name: str) -> str:
    cleaned = repo_name.strip().strip("`'\"")
    if "/" not in cleaned and st.session_state.gh_user is not None:
        return f"{st.session_state.gh_user.login}/{cleaned}"
    return cleaned


def gh_open_repo(repo_name: str) -> str:
    full_name = resolve_repo_name(repo_name)
    repo = st.session_state.gh_client.get_repo(full_name)
    contents = repo.get_contents("")
    entries: List[Dict[str, Any]] = []
    if isinstance(contents, list):
        for item in sorted(contents, key=lambda entry: (entry.type != "dir", entry.name.lower()))[:40]:
            entries.append({
                "name": item.name,
                "type": item.type,
                "path": item.path,
                "url": item.html_url,
            })
    else:
        entries.append({
            "name": contents.name,
            "type": contents.type,
            "path": contents.path,
            "url": contents.html_url,
        })

    readme_text = ""
    try:
        readme = repo.get_readme()
        readme_text = readme.decoded_content.decode("utf-8", errors="replace")[:3000]
    except GithubException:
        readme_text = ""

    st.session_state.current_repo = repo.full_name
    set_live_view("repo", {
        "full_name": repo.full_name,
        "description": repo.description or "",
        "default_branch": repo.default_branch,
        "private": repo.private,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "open_issues": repo.open_issues_count,
        "html_url": repo.html_url,
        "entries": entries,
        "readme": readme_text,
    })

    items = []
    for entry in entries:
        icon = "📁" if entry["type"] == "dir" else "📄"
        items.append(f"- {icon} [{entry['name']}]({entry['url']})")
    listing = "\n".join(items) or "- Repository empty hai."
    return (
        f"**Repository:** [{repo.full_name}]({repo.html_url})\n\n"
        f"**Description:** {repo.description or 'No description'}\n\n"
        f"**Default branch:** `{repo.default_branch}`\n\n"
        f"**Top-level files/folders:**\n{listing}\n\n"
        f"👉 Right side ke **Live GitHub View** panel mein bhi ye repo screen ki tarah dikh raha hai."
    )


def gh_create_repo(
    name: str,
    private: bool = False,
    description: str = "",
    owner: str = "",
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", name):
        return "Repo name invalid hai."
    target_owner = owner.strip().strip("@")
    if target_owner and target_owner.lower() != st.session_state.gh_user.login.lower():
        organization = st.session_state.gh_client.get_organization(target_owner)
        repo = organization.create_repo(
            name, private=private, description=description or ""
        )
    else:
        repo = st.session_state.gh_user.create_repo(
            name, private=private, description=description or ""
        )
    return f"Repository ban gayi: [{repo.full_name}]({repo.html_url})"


def normalize_repo_name(name: str) -> str:
    """Convert a natural-language repo name into a valid GitHub repo name."""
    cleaned = name.strip().strip("`'\".,:;")
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned


def gh_rename_repo(repo_name: str, new_name: str) -> str:
    normalized_name = normalize_repo_name(new_name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", normalized_name):
        return (
            "Naya repo name invalid hai. Letters, numbers, `.`, `_` aur `-` "
            "use karo."
        )

    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    old_full_name = repo.full_name
    repo.edit(name=normalized_name)
    return (
        f"Repository rename ho gayi: [{repo.full_name}]({repo.html_url})\n\n"
        f"Purana naam: `{old_full_name}`\n"
        f"Naya naam: `{repo.full_name}`"
    )


def gh_list_issues(repo_name: str) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    issues = list(repo.get_issues(state="open"))[:20]
    set_live_view("issues", {
        "full_name": repo.full_name,
        "issues": [
            {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "html_url": issue.html_url,
                "is_pull_request": issue.pull_request is not None,
            }
            for issue in issues
        ],
    })
    if not issues:
        return f"{repo_name} mein koi open issue nahi hai."
    return "\n".join(f"- #{issue.number} {issue.title}" for issue in issues)


def gh_read_file(repo_name: str, path: str) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    content = repo.get_contents(path)
    if isinstance(content, list):
        return "Ye path ek folder hai, file nahi."
    text = content.decoded_content.decode("utf-8", errors="replace")
    set_live_view("file", {
        "full_name": repo.full_name,
        "path": content.path,
        "html_url": content.html_url,
        "content": text[:5000],
        "truncated": len(text) > 5000,
    })
    if len(text) > 5000:
        text = text[:5000] + "\n...(truncated)"
    return f"```text\n{text}\n```"


def gh_create_issue(repo_name: str, title: str, body: str = "") -> str:
    issue = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name)).create_issue(
        title=title,
        body=body or "",
    )
    return f"Issue create ho gaya: [#{issue.number}]({issue.html_url})"


def gh_update_issue(
    repo_name: str,
    number: int,
    title: str = "",
    body: str = "",
    state: str = "",
) -> str:
    issue = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name)).get_issue(number)
    changes: Dict[str, str] = {}
    if title.strip():
        changes["title"] = title.strip()
    if body.strip():
        changes["body"] = body
    if state.strip().lower() in {"open", "closed"}:
        changes["state"] = state.strip().lower()
    if not changes:
        return "Issue update ke liye title, body ya state dena zaroori hai."
    issue.edit(**changes)
    return f"Issue #{number} update ho gaya."


def gh_close_pull_request(repo_name: str, number: int) -> str:
    pull_request = st.session_state.gh_client.get_repo(
        resolve_repo_name(repo_name)
    ).get_pull(number)
    pull_request.edit(state="closed")
    return f"Pull request #{number} close ho gaya."


def gh_update_file(repo_name: str, path: str, content: str, commit_message: str) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    existing = repo.get_contents(path)
    if isinstance(existing, list):
        return "Ye path ek folder hai, file nahi."
    repo.update_file(path, commit_message, content, existing.sha)
    return f"File update ho gayi: [{path}]({existing.html_url})"


def gh_delete_file(repo_name: str, path: str, commit_message: str) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    existing = repo.get_contents(path)
    if isinstance(existing, list):
        return "Ye path ek folder hai, file nahi."
    repo.delete_file(path, commit_message, existing.sha)
    return f"File delete ho gayi: `{path}`"


def gh_create_branch(repo_name: str, branch: str, from_branch: str = "") -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    source = from_branch or repo.default_branch
    sha = repo.get_branch(source).commit.sha
    repo.create_git_ref(ref=f"refs/heads/{branch}", sha=sha)
    return f"Branch `{branch}` create ho gayi, source `{source}` se."


def gh_create_pull_request(
    repo_name: str,
    title: str,
    head: str,
    base: str = "",
    body: str = "",
) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    pull_request = repo.create_pull(
        title=title,
        body=body or "",
        head=head,
        base=base or repo.default_branch,
    )
    return f"Pull request create ho gaya: [#{pull_request.number}]({pull_request.html_url})"


def gh_merge_pull_request(repo_name: str, number: int, commit_message: str = "") -> str:
    pull_request = st.session_state.gh_client.get_repo(
        resolve_repo_name(repo_name)
    ).get_pull(number)
    result = pull_request.merge(commit_message=commit_message or None)
    if not result.merged:
        return f"Pull request merge nahi hua: {result.message}"
    return f"Pull request #{number} merge ho gaya."


def gh_close_issue(repo_name: str, number: int) -> str:
    issue = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name)).get_issue(number)
    issue.edit(state="closed")
    return f"Issue #{number} close ho gaya."


def gh_delete_repo(repo_name: str) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    full_name = repo.full_name
    repo.delete()
    return f"Repository `{full_name}` delete ho gayi."


def gh_list_releases(repo_name: str) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    releases = list(repo.get_releases())[:20]
    set_live_view("releases", {
        "full_name": repo.full_name,
        "releases": [
            {
                "tag_name": release.tag_name,
                "title": release.title or release.name or "",
                "html_url": release.html_url,
                "draft": release.draft,
                "prerelease": release.prerelease,
            }
            for release in releases
        ],
    })
    if not releases:
        return f"{repo_name} mein koi release nahi hai."
    return "\n".join(
        f"- `{release.tag_name}` {release.title or release.name or ''} "
        f"({release.html_url})"
        for release in releases
    )


def gh_create_release(
    repo_name: str,
    tag: str,
    name: str = "",
    body: str = "",
    draft: bool = False,
    prerelease: bool = False,
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}", tag.strip()):
        return "Release tag invalid hai."
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    release = repo.create_git_release(
        tag=tag.strip(),
        name=name.strip() or tag.strip(),
        message=body or "",
        draft=draft,
        prerelease=prerelease,
    )
    return f"Release create ho gaya: [{release.tag_name}]({release.html_url})"


def gh_delete_release(repo_name: str, release_id: int) -> str:
    if release_id <= 0:
        return "Valid release_id dena zaroori hai."
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    release = repo.get_release(release_id)
    tag_name = release.tag_name
    release.delete()
    return f"Release `{tag_name}` delete ho gaya."


def gh_list_workflows(repo_name: str) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    workflows = list(repo.get_workflows())[:30]
    set_live_view("workflows", {
        "full_name": repo.full_name,
        "workflows": [
            {"name": workflow.name, "state": workflow.state, "id": workflow.id}
            for workflow in workflows
        ],
    })
    if not workflows:
        return f"{repo_name} mein koi GitHub Actions workflow nahi mila."
    return "\n".join(
        f"- `{workflow.name}` — {workflow.state} (id: {workflow.id})"
        for workflow in workflows
    )


def gh_repo_settings(repo_name: str) -> str:
    """Read the repository's real GitHub settings, the same as the Settings tab shows."""
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    st.session_state.current_repo = repo.full_name
    settings_data = {
        "full_name": repo.full_name,
        "description": repo.description or "",
        "default_branch": repo.default_branch,
        "private": repo.private,
        "has_issues": repo.has_issues,
        "has_wiki": repo.has_wiki,
        "has_projects": repo.has_projects,
        "has_discussions": getattr(repo, "has_discussions", False),
        "allow_merge_commit": getattr(repo, "allow_merge_commit", True),
        "allow_squash_merge": getattr(repo, "allow_squash_merge", True),
        "allow_rebase_merge": getattr(repo, "allow_rebase_merge", True),
        "delete_branch_on_merge": getattr(repo, "delete_branch_on_merge", False),
        "topics": repo.get_topics() if hasattr(repo, "get_topics") else [],
        "html_url": repo.html_url,
        "settings_url": f"{repo.html_url}/settings",
    }
    set_live_view("settings", settings_data)
    lines = [
        f"**Repository settings:** [{repo.full_name}]({settings_data['settings_url']})",
        f"- Visibility: **{'Private' if settings_data['private'] else 'Public'}**",
        f"- Default branch: `{settings_data['default_branch']}`",
        f"- Description: {settings_data['description'] or '_(none)_'}",
        f"- Issues: {'Enabled' if settings_data['has_issues'] else 'Disabled'}",
        f"- Wiki: {'Enabled' if settings_data['has_wiki'] else 'Disabled'}",
        f"- Projects: {'Enabled' if settings_data['has_projects'] else 'Disabled'}",
        f"- Delete branch on merge: {'Yes' if settings_data['delete_branch_on_merge'] else 'No'}",
        f"- Topics: {', '.join(settings_data['topics']) if settings_data['topics'] else '_(none)_'}",
        "",
        "👉 Right side ke **Live GitHub View** mein yeh Settings page ki tarah bhi dikh raha hai.",
    ]
    return "\n".join(lines)


def gh_update_repo_settings(repo_name: str, changes: Dict[str, Any]) -> str:
    """Apply requested settings changes exactly the way GitHub's Settings page would."""
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    edit_kwargs: Dict[str, Any] = {}
    applied: List[str] = []

    if "private" in changes:
        edit_kwargs["private"] = bool(changes["private"])
        applied.append(f"Visibility -> {'Private' if changes['private'] else 'Public'}")
    if "description" in changes:
        edit_kwargs["description"] = str(changes["description"])
        applied.append("Description update")
    if "default_branch" in changes:
        edit_kwargs["default_branch"] = str(changes["default_branch"])
        applied.append(f"Default branch -> {changes['default_branch']}")
    if "has_issues" in changes:
        edit_kwargs["has_issues"] = bool(changes["has_issues"])
        applied.append(f"Issues -> {'Enabled' if changes['has_issues'] else 'Disabled'}")
    if "has_wiki" in changes:
        edit_kwargs["has_wiki"] = bool(changes["has_wiki"])
        applied.append(f"Wiki -> {'Enabled' if changes['has_wiki'] else 'Disabled'}")
    if "has_projects" in changes:
        edit_kwargs["has_projects"] = bool(changes["has_projects"])
        applied.append(f"Projects -> {'Enabled' if changes['has_projects'] else 'Disabled'}")
    if "delete_branch_on_merge" in changes:
        edit_kwargs["delete_branch_on_merge"] = bool(changes["delete_branch_on_merge"])
        applied.append(
            f"Delete branch on merge -> {'Yes' if changes['delete_branch_on_merge'] else 'No'}"
        )

    if not edit_kwargs and "topics" not in changes:
        return "Settings mein kya change karna hai woh clearly batao (jaise private/public, description, issues on/off)."

    if edit_kwargs:
        repo.edit(**edit_kwargs)
    if "topics" in changes and isinstance(changes["topics"], list):
        repo.replace_topics([str(topic) for topic in changes["topics"]])
        applied.append("Topics update")

    # Refresh the live view with the settings GitHub now actually has.
    gh_repo_settings(repo.full_name)
    return "Settings update ho gayi:\n" + "\n".join(f"- {line}" for line in applied)


def gh_run_workflow(
    repo_name: str,
    workflow: str,
    ref: str = "",
    inputs: Optional[Dict[str, Any]] = None,
) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    workflow_name = workflow.strip()
    if not workflow_name:
        return "Workflow file ya workflow id dena zaroori hai."
    workflow_obj = repo.get_workflow(workflow_name)
    workflow_obj.create_dispatch(ref=ref.strip() or repo.default_branch, inputs=inputs or {})
    return f"Workflow `{workflow_name}` trigger ho gaya."


EC2_DEPLOY_WORKFLOW_PATH = ".github/workflows/deploy-aws-ec2.yml"
EC2_DEPLOY_WORKFLOW_MARKER = "Deploy to AWS EC2 (self-hosted)"

def _load_ec2_workflow_template() -> str:
    path = Path(__file__).resolve().parent / EC2_DEPLOY_WORKFLOW_PATH
    return path.read_text(encoding="utf-8")

def _get_github_token_for_api() -> str:
    client = st.session_state.get("gh_client")
    requester = getattr(client, "_Github__requester", None)
    auth = getattr(requester, "_Requester__auth", None) if requester else None
    return str(getattr(auth, "token", "") or "") if auth else ""

def _github_rest(method: str, url: str, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {}) or {}
    headers["Accept"] = "application/vnd.github+json"
    token = _get_github_token_for_api()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.request(method, url, headers=headers, timeout=30, **kwargs)

def gh_ensure_ec2_workflow(repo: Any, branch: str) -> Tuple[bool, str]:
    content = _load_ec2_workflow_template()
    try:
        existing = repo.get_contents(EC2_DEPLOY_WORKFLOW_PATH, ref=branch)
        if isinstance(existing, list):
            return False, "EC2 deployment workflow path is a directory."
        decoded = existing.decoded_content.decode("utf-8", errors="replace")
        if EC2_DEPLOY_WORKFLOW_MARKER in decoded:
            return True, "Managed AWS EC2 deployment workflow already exists."
        return False, f"`{EC2_DEPLOY_WORKFLOW_PATH}` already exists and is not managed by this Agent, so it was not overwritten."
    except GithubException as exc:
        if exc.status != 404:
            return False, github_error(exc)
    try:
        repo.create_file(EC2_DEPLOY_WORKFLOW_PATH, "Add managed AWS EC2 deployment workflow", content, branch=branch)
        return True, "AWS EC2 deployment workflow added."
    except Exception as exc:
        return False, f"Workflow file add nahi ho paayi: {github_error(exc)}"

def _download_ec2_artifact(artifact: Dict[str, Any]) -> Optional[str]:
    url = artifact.get("archive_download_url")
    if not url:
        return None
    response = _github_rest("GET", url)
    if response.status_code != 200:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for name in z.namelist():
                if name.endswith("deployment-info.txt"):
                    return z.read(name).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, OSError):
        return None
    return None

def _parse_deployment_info(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result

def _aws_session() -> Any:
    if boto3 is None:
        raise RuntimeError(
            "AWS SDK (boto3) is not installed in the Agent runtime. "
            "Run `run_app.bat` on Windows or `python -m pip install -r requirements.txt`."
        )
    region = (st.session_state.get("aws_region") or os.getenv("AWS_REGION") or "us-east-1").strip()
    kwargs: Dict[str, Any] = {"region_name": region}
    access = st.session_state.get("aws_access_key_id", "").strip()
    secret = st.session_state.get("aws_secret_access_key", "").strip()
    token = st.session_state.get("aws_session_token", "").strip()
    if access and secret:
        kwargs.update(aws_access_key_id=access, aws_secret_access_key=secret)
        if token:
            kwargs["aws_session_token"] = token
    return boto3.Session(**kwargs)


def aws_connection_status() -> Dict[str, Any]:
    """Return a safe, non-secret AWS readiness snapshot for the UI."""
    if boto3 is None:
        return {"ready": False, "message": "boto3 missing. Start with run_app.bat or install requirements.txt."}
    try:
        session = _aws_session()
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        ec2 = session.client("ec2")
        instances = _aws_find_ec2_instance(ec2, st.session_state.get("aws_ec2_instance_id", ""))
        ssm = session.client("ssm")
        info = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instances["InstanceId"]]}]
        )
        online = bool(info.get("InstanceInformationList"))
        return {
            "ready": online,
            "account": str(identity.get("Account", "")),
            "instance_id": str(instances.get("InstanceId", "")),
            "instance_state": str(instances.get("State", {}).get("Name", "")),
            "ssm_online": online,
            "message": "AWS + EC2 + SSM ready." if online else "EC2 mil gaya, lekin SSM managed/online nahi hai."
        }
    except Exception as exc:
        return {"ready": False, "message": str(exc)[:1000]}


def _aws_find_ec2_instance(ec2: Any, requested_id: str = "") -> Dict[str, Any]:
    if requested_id.strip():
        response = ec2.describe_instances(InstanceIds=[requested_id.strip()])
        reservations = response.get("Reservations", [])
        instances = [i for r in reservations for i in r.get("Instances", [])]
        if not instances:
            raise RuntimeError(f"EC2 instance `{requested_id}` nahi mila.")
        return instances[0]
    name = os.getenv("EC2_INSTANCE_NAME", "github_App")
    response = ec2.describe_instances(Filters=[
        {"Name": "instance-state-name", "Values": ["running"]},
        {"Name": "tag:Name", "Values": [name]},
    ])
    instances = [i for r in response.get("Reservations", []) for i in r.get("Instances", [])]
    if len(instances) == 1:
        return instances[0]
    if not instances:
        raise RuntimeError("Running EC2 instance auto-detect nahi hua. EC2 Instance ID set karo.")
    raise RuntimeError("Multiple running EC2 instances mile. EC2 Instance ID set karke target choose karo.")


def _aws_security_group_id(instance: Dict[str, Any]) -> str:
    configured = st.session_state.get("aws_security_group_id", "").strip() or os.getenv("EC2_SECURITY_GROUP_ID", "").strip()
    if configured:
        return configured
    groups = instance.get("SecurityGroups", [])
    return str(groups[0].get("GroupId", "")) if groups else ""


def _aws_open_port(ec2: Any, security_group_id: str, port: int) -> str:
    if not security_group_id:
        return "Security group ID nahi mila; port rule auto-add nahi hua."
    try:
        ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[{"IpProtocol": "tcp", "FromPort": port, "ToPort": port,
                           "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "GitHub Agent deployment port"}]}],
        )
        return f"Security group `{security_group_id}` mein TCP {port} allow kiya."
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        if code == "InvalidPermission.Duplicate":
            return f"Security group `{security_group_id}` mein TCP {port} already allowed hai."
        return f"Security-group rule auto-add nahi hua: {exc}"


def _aws_wait_ssm(ssm: Any, command_id: str, instance_id: str, timeout: int = 900) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        try:
            last = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        except (ClientError, BotoCoreError):
            time.sleep(3)
            continue
        status = str(last.get("Status", ""))
        if status in {"Success", "Failed", "TimedOut", "Cancelled", "Cancelling"}:
            return last
        time.sleep(4)
    return last or {"Status": "TimedOut", "StandardErrorContent": "SSM command timeout."}


def gh_deploy_aws_ec2(repo_name: str, runner_label: str = "ec2-deployer", port: str = "", ref: str = "") -> str:
    """Deploy a public GitHub repo directly to EC2 through AWS SSM."""
    full_name = resolve_repo_name(repo_name)
    repo = st.session_state.gh_client.get_repo(full_name)
    branch = ref.strip() or repo.default_branch
    if getattr(repo, "private", False):
        return ("Private GitHub repo detected hai. Safe automatic EC2 clone ke liye "
                "GitHub App installation-token integration configure karo; PAT ko SSM command mein expose nahi kiya jayega.")
    requested_port = port.strip()
    if requested_port and (not requested_port.isdigit() or not 1024 <= int(requested_port) <= 65535):
        return "Port 1024-65535 ke beech valid number hona chahiye; blank chhodo to Agent 8501-8599 mein choose karega."
    session = _aws_session()
    ec2 = session.client("ec2")
    ssm = session.client("ssm")
    instance = _aws_find_ec2_instance(ec2, st.session_state.get("aws_ec2_instance_id", ""))
    instance_id = str(instance.get("InstanceId", ""))
    public_ip = str(instance.get("PublicIpAddress", "") or "")
    public_host = str(instance.get("PublicDnsName", "") or "")
    if not public_ip and not public_host:
        return f"EC2 `{instance_id}` running hai, lekin public IPv4/DNS nahi mila."
    import shlex
    slug = re.sub(r"[^a-z0-9_.-]", "-", full_name.rsplit("/", 1)[-1].lower()).strip("-._") or "app"
    repo_q, branch_q, slug_q, port_q = map(shlex.quote, [f"https://github.com/{full_name}.git", branch, slug, requested_port])
    command = f"""set -euo pipefail
APP_ROOT=/opt/github-agent/apps
REPO_URL={repo_q}
BRANCH={branch_q}
SLUG={slug_q}
APP_DIR=\"$APP_ROOT/$SLUG\"
mkdir -p \"$APP_ROOT\"
if [ -d \"$APP_DIR/.git\" ]; then
  git -C \"$APP_DIR\" fetch --depth 1 origin \"$BRANCH\"
  git -C \"$APP_DIR\" checkout -B \"$BRANCH\" \"origin/$BRANCH\"
  git -C \"$APP_DIR\" reset --hard \"origin/$BRANCH\"
else
  rm -rf \"$APP_DIR\"
  git clone --depth 1 --branch \"$BRANCH\" \"$REPO_URL\" \"$APP_DIR\"
fi
cd \"$APP_DIR\"
if [ ! -f Dockerfile ]; then
  test -f app.py && test -f requirements.txt || {{ echo 'No Dockerfile and no app.py + requirements.txt found.'; exit 2; }}
  cat > Dockerfile <<'DOCKERFILE'
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD [\"streamlit\",\"run\",\"app.py\",\"--server.address=0.0.0.0\",\"--server.port=8501\",\"--server.headless=true\"]
DOCKERFILE
fi
IMAGE=\"agent/$SLUG:latest\"
CONTAINER=\"agent-$SLUG\"
docker build -t \"$IMAGE\" .
PORT={port_q}
if [ -z \"$PORT\" ]; then
  for CANDIDATE in $(seq 8501 8599); do
    if ! docker ps --format '{{{{.Ports}}}}' | grep -Eq \"(^|[,:])$CANDIDATE->|:$CANDIDATE-\"; then PORT=\"$CANDIDATE\"; break; fi
  done
fi
[ -n \"$PORT\" ] || {{ echo 'No free host port in 8501-8599'; exit 3; }}
docker rm -f \"$CONTAINER\" >/dev/null 2>&1 || true
docker run -d --restart unless-stopped --name \"$CONTAINER\" -p \"$PORT:8501\" \"$IMAGE\"
sleep 4
docker ps --format '{{{{.Names}}}}' | grep -qx \"$CONTAINER\"
printf 'PORT=%s\nCONTAINER=%s\nIMAGE=%s\nAPP_DIR=%s\n' \"$PORT\" \"$CONTAINER\" \"$IMAGE\" \"$APP_DIR\"
"""
    emit_action_event("step_started", "running", "Connecting to AWS EC2", f"Target EC2 `{instance_id}` selected for `{full_name}`.", {"instance_id": instance_id, "region": session.region_name}, step=1, total_steps=5)
    try:
        response = ssm.send_command(InstanceIds=[instance_id], DocumentName="AWS-RunShellScript", Parameters={"commands": [command]}, Comment=f"GitHub Agent deploy {full_name}", TimeoutSeconds=900)
    except Exception as exc:
        return f"AWS SSM command start nahi hua: {exc}"
    command_id = str(response["Command"]["CommandId"])
    emit_action_event("step_started", "running", "Deploying on EC2", "GitHub repo clone/pull, Docker build aur container start ho raha hai.", {"command_id": command_id}, step=2, total_steps=5)
    result = _aws_wait_ssm(ssm, command_id, instance_id, timeout=900)
    status = str(result.get("Status", ""))
    if status != "Success":
        err = str(result.get("StandardErrorContent", "") or result.get("StatusDetails", "") or "Unknown SSM error")
        emit_action_event("step_failed", "failed", "EC2 deployment failed", err[:2000], {"command_id": command_id, "status": status}, step=2, total_steps=5)
        return f"❌ EC2 deployment failed (`{status}`).\n\n```text\n{err[:4000]}\n```"
    output = str(result.get("StandardOutputContent", ""))
    values: Dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    deployed_port = int(values.get("PORT", requested_port or "0"))
    sg_id = _aws_security_group_id(instance)
    sg_message = _aws_open_port(ec2, sg_id, deployed_port)
    emit_action_event("step_completed", "completed", "Docker container running", f"Container `{values.get('CONTAINER', slug)}` is running on port {deployed_port}.", {"container": values.get("CONTAINER", ""), "port": deployed_port}, step=3, total_steps=5)
    refreshed = _aws_find_ec2_instance(ec2, instance_id)
    host = str(refreshed.get("PublicIpAddress", "") or refreshed.get("PublicDnsName", "") or public_ip or public_host)
    url = f"http://{host}:{deployed_port}"
    set_live_view("aws_ec2", {"repository": full_name, "instance_id": instance_id, "host": host, "port": str(deployed_port), "container": values.get("CONTAINER", ""), "image": values.get("IMAGE", ""), "url": url, "security_group": sg_id or "unknown", "security_group_result": sg_message, "deployment_mode": "AWS SSM direct"})
    emit_action_event("step_completed", "completed", "AWS deployment complete", f"Live app: {url}", {"url": url, "host": host, "port": deployed_port, "instance_id": instance_id}, step=5, total_steps=5)
    return ("## 🚀 AWS EC2 deployment complete\n\n" f"**Live App:** [{url}]({url})\n\n" f"- **EC2 Instance:** `{instance_id}`\n" f"- **Host:** `{host}`\n" f"- **Port:** `{deployed_port}`\n" f"- **Container:** `{values.get('CONTAINER', 'unknown')}`\n" f"- **Security Group:** `{sg_id or 'unknown'}` — {sg_message}\n\nAgent ne actual AWS response ke baad hi URL return kiya hai.")

def gh_deploy_aws(
    repo_name: str,
    workflow: str = "",
    ref: str = "",
    inputs: Optional[Dict[str, Any]] = None,
) -> str:
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    workflows = list(repo.get_workflows())
    target = workflow.strip()
    if target:
        workflow_obj = repo.get_workflow(target)
    else:
        workflow_obj = next(
            (
                item
                for item in workflows
                if "deploy" in f"{item.name} {item.path}".lower()
                or "aws" in f"{item.name} {item.path}".lower()
            ),
            None,
        )
        if workflow_obj is None:
            return (
                "AWS deployment ke liye repo mein deploy/AWS GitHub Actions "
                "workflow nahi mila. `.github/workflows/deploy.yml` add karke "
                "AWS secrets configure karo, phir dobara deploy bolo."
            )
    workflow_obj.create_dispatch(
        ref=ref.strip() or repo.default_branch,
        inputs=inputs or {},
    )
    return (
        f"AWS deployment workflow `{workflow_obj.name}` trigger ho gaya. "
        f"Status GitHub Actions mein dekho: {repo.html_url}/actions"
    )


def find_streamlit_entrypoints(repo: Any, branch: str) -> List[str]:
    found: List[str] = []

    def walk(path: str = "", depth: int = 0) -> None:
        if depth > 3:
            return
        try:
            entries = repo.get_contents(path, ref=branch)
        except GithubException:
            return
        if not isinstance(entries, list):
            return
        for entry in entries:
            if entry.type == "file" and entry.name.endswith(".py"):
                found.append(entry.path)
            elif entry.type == "dir" and not entry.name.startswith("."):
                walk(entry.path, depth + 1)

    walk()
    return found


def gh_prepare_streamlit_deploy(
    repo_name: str,
    branch: str = "",
    app_path: str = "",
) -> str:
    full_name = resolve_repo_name(repo_name)
    repo = st.session_state.gh_client.get_repo(full_name)
    selected_branch = branch or repo.default_branch
    selected_path = app_path.strip().strip("/")
    if selected_path:
        candidate = repo.get_contents(selected_path, ref=selected_branch)
        if isinstance(candidate, list) or not selected_path.endswith(".py"):
            return f"`{selected_path}` ek valid Streamlit Python entrypoint nahi hai."
    else:
        candidates = find_streamlit_entrypoints(repo, selected_branch)
        preferred = ["app.py", "streamlit_app.py", "main.py"]
        selected_path = next(
            (
                path
                for name in preferred
                for path in candidates
                if path.rsplit("/", 1)[-1] == name
            ),
            sorted(candidates, key=lambda path: (path.count("/"), path))[0]
            if candidates
            else "",
        )
        if not selected_path:
            return "Repository mein koi Streamlit `.py` entrypoint nahi mila."

    deploy_url = (
        "https://share.streamlit.io/deploy"
        f"?repository={quote(full_name)}"
        f"&branch={quote(selected_branch)}"
        f"&mainModule={quote(selected_path)}"
    )
    return (
        "Streamlit Community Cloud deployment setup ready hai.\n\n"
        f"- Repository: `{full_name}`\n"
        f"- Branch: `{selected_branch}`\n"
        f"- App file: `{selected_path}`\n\n"
        f"[Open Streamlit deployment setup]({deploy_url})\n\n"
        "Cloud par sign in karke **Create app** confirm karna hoga. "
        "Private repo ke liye Streamlit account ko repo access dena zaroori hai."
    )


def extract_deployment_platform(text: str) -> Optional[str]:
    """Return a supported platform when the user explicitly names one."""
    value = text.lower().strip()
    for alias, platform in sorted(
        DEPLOYMENT_PLATFORM_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", value):
            return platform
    match = re.fullmatch(r"\s*([1-5])(?:\s*[\).:-])?\s*", value)
    if match:
        return DEPLOYMENT_PLATFORM_OPTIONS[int(match.group(1)) - 1]
    return None


def deployment_action_for(platform: str, repo_name: str) -> Dict[str, Any]:
    if platform == "Streamlit Cloud":
        return {
            "type": "deploy_streamlit",
            "repo": repo_name,
            "branch": "",
            "app_path": "",
            "platform": platform,
        }
    if platform == "AWS EC2":
        return {
            "type": "deploy_aws_ec2",
            "repo": repo_name,
            "port": "",
            "instance_id": st.session_state.get("aws_ec2_instance_id", ""),
            "ref": "",
            "platform": platform,
        }
    return {
        "type": "prepare_deployment",
        "repo": repo_name,
        "platform": platform,
        "branch": "",
    }


def deployment_dashboard_url(platform: str) -> str:
    return {
        "Azure": "https://portal.azure.com/#create/Microsoft.WebApp",
        "Google Cloud": "https://console.cloud.google.com/run",
        "Render": "https://dashboard.render.com/blueprints/new",
    }.get(platform, "")


def gh_prepare_platform_deploy(
    repo_name: str,
    platform: str,
    branch: str = "",
) -> str:
    """Prepare a deployment plan for platforms without a direct API integration."""
    if platform not in DEPLOYMENT_PLATFORM_OPTIONS:
        return "Unsupported deployment platform."
    repo = st.session_state.gh_client.get_repo(resolve_repo_name(repo_name))
    selected_branch = branch or repo.default_branch
    contents = repo.get_contents("", ref=selected_branch)
    names = (
        [item.path for item in contents]
        if isinstance(contents, list)
        else [contents.path]
    )
    workflows = list(repo.get_workflows())
    workflow_terms = {
        "Azure": ("azure",),
        "Google Cloud": ("google", "gcp", "cloud-run", "cloud run"),
        "Render": ("render",),
    }
    matching_workflow = next(
        (
            workflow
            for workflow in workflows
            if any(
                term in f"{workflow.name} {workflow.path}".lower()
                for term in workflow_terms.get(platform, ())
            )
            and "deploy" in f"{workflow.name} {workflow.path}".lower()
        ),
        None,
    )
    analysis = st.session_state.project_analysis or {}
    recommended = analysis.get("recommended_platform")
    reason = analysis.get("recommendation_reason")
    dashboard_url = deployment_dashboard_url(platform)
    lines = [
        f"**{platform} deployment plan ready hai.**",
        f"- Repository: `{repo.full_name}`",
        f"- Branch: `{selected_branch}`",
        f"- Detected top-level files: `{', '.join(names[:12]) or 'none'}`",
    ]
    if recommended and recommended != platform:
        lines.append(
            f"- File analysis recommendation: **{recommended}** — {reason}"
        )
    if matching_workflow:
        matching_workflow.create_dispatch(ref=selected_branch, inputs={})
        lines.extend(
            [
                f"- Matching workflow: `{matching_workflow.name}`",
                f"- Workflow trigger ho gaya. Status: {repo.html_url}/actions",
            ]
        )
        return "\n".join(lines)

    if platform == "Azure":
        next_step = (
            "Azure Web App create karke GitHub deployment source connect karo. "
            "Python apps ke liye startup command aur dependency file verify karo."
        )
    elif platform == "Google Cloud":
        next_step = (
            "Cloud Run service create karo. Dockerfile ho to usi se deploy karo; "
            "warna build/runtime command aur port `8080` configure karo."
        )
    else:
        next_step = (
            "Render mein Web Service/Blueprint choose karo, repository connect karo, "
            "build command aur start command set karke deploy karo."
        )
    lines.extend(
        [
            "- Matching GitHub Actions workflow nahi mila, isliye deployment complete nahi hua.",
            f"- Next step: {next_step}",
            f"[Open {platform} setup]({dashboard_url})",
        ]
    )
    return "\n".join(lines)


def requires_github(action_type: str) -> bool:
    return action_type in {
        "list_repos",
        "open_repo",
        "create_repo",
        "rename_repo",
        "create_issue",
        "list_issues",
        "read_file",
        "update_file",
        "delete_file",
        "create_branch",
        "create_pull_request",
        "merge_pull_request",
        "close_issue",
        "delete_repo",
        "list_releases",
        "create_release",
        "delete_release",
        "list_workflows",
        "run_workflow",
        "deploy_aws",
        "deploy_aws_ec2",
        "push_project",
        "deploy_streamlit",
        "prepare_deployment",
        "repo_settings",
        "update_repo_settings",
    }


def execute_action(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    st.session_state.active_request_id = st.session_state.get("active_request_id") or uuid.uuid4().hex
    st.session_state.active_action_id = uuid.uuid4().hex
    emit_action_event("step_started", "running", "Action started",
                      f"Executing {action_type}.", action)
    if requires_github(action_type) and st.session_state.gh_user is None:
        return "Pehle sidebar se GitHub token connect karo."
    try:
        if action_type == "list_repos":
            return gh_list_repos()
        if action_type == "analyze_project":
            return project_analysis_text(st.session_state.project_analysis)
        if action_type == "open_repo":
            return gh_open_repo(str(action.get("repo", "")))
        if action_type == "create_repo":
            return gh_create_repo(
                str(action.get("name", "")),
                bool(action.get("private", False)),
                str(action.get("description", "")),
                str(action.get("owner", "")),
            )
        if action_type == "rename_repo":
            return gh_rename_repo(
                str(action.get("repo", "")),
                str(action.get("new_name", "")),
            )
        if action_type == "create_issue":
            return gh_create_issue(
                str(action.get("repo", "")),
                str(action.get("title", "")),
                str(action.get("body", "")),
            )
        if action_type == "update_issue":
            return gh_update_issue(
                str(action.get("repo", "")),
                int(action.get("number", 0)),
                str(action.get("title", "")),
                str(action.get("body", "")),
                str(action.get("state", "")),
            )
        if action_type == "close_pull_request":
            return gh_close_pull_request(
                str(action.get("repo", "")),
                int(action.get("number", 0)),
            )
        if action_type == "list_issues":
            return gh_list_issues(str(action.get("repo", "")))
        if action_type == "read_file":
            return gh_read_file(str(action.get("repo", "")), str(action.get("path", "")))
        if action_type == "update_file":
            return gh_update_file(
                str(action.get("repo", "")),
                str(action.get("path", "")),
                str(action.get("content", "")),
                str(action.get("commit_message", "Update file via GitHub Agent")),
            )
        if action_type == "delete_file":
            return gh_delete_file(
                str(action.get("repo", "")),
                str(action.get("path", "")),
                str(action.get("commit_message", "Delete file via GitHub Agent")),
            )
        if action_type == "create_branch":
            return gh_create_branch(
                str(action.get("repo", "")),
                str(action.get("branch", "")),
                str(action.get("from_branch", "")),
            )
        if action_type == "create_pull_request":
            return gh_create_pull_request(
                str(action.get("repo", "")),
                str(action.get("title", "")),
                str(action.get("head", "")),
                str(action.get("base", "")),
                str(action.get("body", "")),
            )
        if action_type == "merge_pull_request":
            return gh_merge_pull_request(
                str(action.get("repo", "")),
                int(action.get("number", 0)),
                str(action.get("commit_message", "")),
            )
        if action_type == "close_issue":
            return gh_close_issue(
                str(action.get("repo", "")),
                int(action.get("number", 0)),
            )
        if action_type == "delete_repo":
            return gh_delete_repo(str(action.get("repo", "")))
        if action_type == "list_releases":
            return gh_list_releases(str(action.get("repo", "")))
        if action_type == "create_release":
            return gh_create_release(
                str(action.get("repo", "")),
                str(action.get("tag", "")),
                str(action.get("name", "")),
                str(action.get("body", "")),
                bool(action.get("draft", False)),
                bool(action.get("prerelease", False)),
            )
        if action_type == "delete_release":
            return gh_delete_release(
                str(action.get("repo", "")),
                int(action.get("release_id", 0)),
            )
        if action_type == "list_workflows":
            return gh_list_workflows(str(action.get("repo", "")))
        if action_type == "repo_settings":
            return gh_repo_settings(str(action.get("repo", "")))
        if action_type == "update_repo_settings":
            changes = action.get("changes", {})
            return gh_update_repo_settings(
                str(action.get("repo", "")),
                changes if isinstance(changes, dict) else {},
            )
        if action_type == "run_workflow":
            inputs = action.get("inputs", {})
            return gh_run_workflow(
                str(action.get("repo", "")),
                str(action.get("workflow", "")),
                str(action.get("ref", "")),
                inputs if isinstance(inputs, dict) else {},
            )
        if action_type == "deploy_aws":
            inputs = action.get("inputs", {})
            return gh_deploy_aws(
                str(action.get("repo", "")),
                str(action.get("workflow", "")),
                str(action.get("ref", "")),
                inputs if isinstance(inputs, dict) else {},
            )
        if action_type == "deploy_aws_ec2":
            return gh_deploy_aws_ec2(
                str(action.get("repo", "")),
                str(action.get("runner_label", "ec2-deployer")),
                str(action.get("port", "")),
                str(action.get("ref", "")),
            )
        if action_type == "push_project":
            return gh_push_project(
                str(action.get("repo_name", "")),
                bool(action.get("private", st.session_state.private_repo)),
                str(action.get("commit_message", st.session_state.commit_message)),
            )
        if action_type == "deploy_streamlit":
            return gh_prepare_streamlit_deploy(
                str(action.get("repo", "")),
                str(action.get("branch", "")),
                str(action.get("app_path", "")),
            )
        if action_type == "prepare_deployment":
            return gh_prepare_platform_deploy(
                str(action.get("repo", "")),
                str(action.get("platform", "")),
                str(action.get("branch", "")),
            )
        return "Is request ke liye koi supported GitHub action nahi mila."
    except Exception as exc:
        return f"GitHub action fail hui: {github_error(exc)}"


CONFIRMATION_ACTIONS = {
    "create_repo",
    "rename_repo",
    "create_issue",
    "update_issue",
    "close_pull_request",
    "update_file",
    "delete_file",
    "create_branch",
    "create_pull_request",
    "merge_pull_request",
    "close_issue",
    "delete_repo",
    "create_release",
    "delete_release",
    "run_workflow",
    "deploy_aws",
    "push_project",
    "deploy_streamlit",
    "prepare_deployment",
    "update_repo_settings",
}


def needs_confirmation(action: Dict[str, Any]) -> bool:
    return action.get("type") in CONFIRMATION_ACTIONS


def confirmation_preview(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    if action_type == "push_project":
        return (
            f"Repository `{action.get('repo_name')}` mein project push karna hai.\n"
            f"- Visibility: {'Private' if action.get('private') else 'Public'}\n"
            f"- Commit message: `{action.get('commit_message')}`\n\n"
            "Commit karne ke liye `yes` likho, cancel karne ke liye `no`."
        )
    if action_type == "deploy_streamlit":
        return (
            f"Streamlit Community Cloud ke liye `{action.get('repo')}` prepare karna hai.\n"
            "Deployment setup link banane ke liye `yes` likho, cancel ke liye `no`."
        )
    if action_type == "deploy_aws_ec2":
        return (
            f"Repository `{action.get('repo')}` ko **AWS EC2** par deploy karna hai.\n"
            f"- EC2: `{action.get('instance_id') or os.getenv('EC2_INSTANCE_ID', 'auto-detect')}`\n"
            "- GitHub repository EC2 par clone/pull hogi\n"
            "- Docker image EC2 par build hogi\n"
            "- Available host port automatically choose hoga\n"
            "- Required security-group port automatically add karne ki koshish hogi\n"
            "- Existing same-name container replace hoga\n"
            "- Deployment complete hone par **actual live URL + host + port** return hoga\n\n"
            "**Permission required:** AWS EC2 deployment + SSM command + security-group port update.\n"
            "Continue karne ke liye `yes` likho, cancel ke liye `no`."
        )
    if action_type == "deploy_aws":
        return (
            f"Repository `{action.get('repo')}` ka AWS GitHub Actions deployment "
            "trigger karna hai. `yes` likho to workflow run hoga; `no` se cancel."
        )
    if action_type == "prepare_deployment":
        return (
            f"Repository `{action.get('repo')}` ko **{action.get('platform')}** "
            "par deploy karne ka plan prepare karna hai.\n"
            "Matching workflow mila to trigger hoga; warna setup instructions milengi. "
            "Continue karne ke liye `yes` likho, cancel ke liye `no`."
        )
    if action_type == "rename_repo":
        return (
            f"Repository `{action.get('repo')}` ko "
            f"`{action.get('new_name')}` naam dena hai.\n\n"
            "GitHub par rename karne ke liye `yes` likho, cancel ke liye `no`."
        )
    if action_type == "update_file":
        content = str(action.get("content", ""))
        preview = content[:500] + ("..." if len(content) > 500 else "")
        return (
            f"File `{action.get('path')}` update karni hai.\n\n"
            f"```text\n{preview}\n```\n"
            "Ye change/commit karne ke liye `yes` likho, cancel ke liye `no`."
        )
    if action_type == "delete_repo":
        target = action.get("repo", "")
        return (
            f"Warning: repository `{target}` permanently delete karni hai.\n"
            "Agar sach mein delete karna hai to `yes delete` likho; cancel ke liye `no`."
        )
    return (
        f"Action `{action_type}` perform karni hai. Confirm karne ke liye `yes` "
        "likho, cancel karne ke liye `no`."
    )


def is_confirmation_yes(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(yes|y|haan|ha|ok|okay|confirm|proceed|kar do|yes delete)\s*",
            text,
            flags=re.IGNORECASE,
        )
    )


def is_confirmation_no(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(no|n|nahi|nahin|cancel|cancel karo|mat karo)\s*",
            text,
            flags=re.IGNORECASE,
        )
    )


def looks_like_push(text: str) -> bool:
    value = text.lower()
    push_words = (
        "push",
        "publish",
        "upload",
        "github pe",
        "github par",
        "github mein",
        "github me",
        "गिटहब",
        "पुश",
    )
    project_words = (
        "project",
        "folder",
        "file",
        "files",
        "repo",
        "repository",
        "project",
        "प्रोजेक्ट",
        "फोल्डर",
    )
    return any(word in value for word in push_words) and (
        any(word in value for word in project_words) or "github" in value
    )


def looks_like_deploy(text: str) -> bool:
    value = text.lower()
    deploy_words = (
        "deploy",
        "deply",
        "deploi",
        "diploy",
        "streamlit cloud",
        "streamlit par",
        "streamlit pe",
        "डिप्लॉय",
    )
    return any(word in value for word in deploy_words)


def looks_like_create_repo(text: str) -> bool:
    value = text.lower()
    return (
        any(word in value for word in ("create", "new", "make", "bana", "बना"))
        and bool(re.search(r"\b(repo|repository)\b", value))
        and not looks_like_push(value)
    )


def looks_like_rename_repo(text: str) -> bool:
    value = text.lower()
    rename_words = (
        "rename",
        "renamed",
        "renarte",
        "naam badal",
        "नाम बदल",
        "rename karo",
        "rename kar",
    )
    has_repo_word = bool(
        re.search(r"\b(repo|repos|repository|repositories)\b", value)
    )
    has_owner_repo_path = bool(
        re.search(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\b", value)
    )
    natural_name_change = bool(
        re.search(
            r"\b(repo|repository)\b.*\b(ka naam|name)\b.*\b(rakh|rakho|badal)\b",
            value,
        )
    )
    return (
        (any(word in value for word in rename_words) and (has_repo_word or has_owner_repo_path))
        or natural_name_change
    )


def looks_like_delete_repo(text: str) -> bool:
    value = text.lower()
    return (
        any(word in value for word in ("delete", "remove", "erase", "हटा", "डिलीट"))
        and bool(re.search(r"\b(repo|repository)\b", value))
    )


def looks_like_list_repos(text: str) -> bool:
    value = text.lower()
    return bool(
        re.search(r"\b(list|show|display|mere|my)\b", value)
        and re.search(r"\b(repos?|repositories)\b", value)
    )


def looks_like_analyze(text: str) -> bool:
    value = text.lower()
    return any(
        phrase in value
        for phrase in (
            "analyze project",
            "analyse project",
            "project analysis",
            "framework detect",
            "framework batao",
            "project check",
            "analyze this",
            "analyse this",
        )
    )


GENERATE_PROJECT_VERBS = (
    "banao",
    "banavo",
    "banade",
    "bana do",
    "bana ke do",
    "banaiye",
    "banaye",
    "bana kar do",
    "banadena",
    "bana dena",
    "बनाओ",
    "create",
    "generate",
    "build",
    "banao naa",
)
GENERATE_PROJECT_NOUNS = ("project", "app", "application", "website", "site", "प्रोजेक्ट")
GENERATE_PROJECT_FRAMEWORKS = ("streamlit",)


def looks_like_generate_project(text: str) -> bool:
    """Detect requests to build a brand-new project from scratch (not an upload)."""
    value = text.lower()
    if not any(framework in value for framework in GENERATE_PROJECT_FRAMEWORKS):
        return False
    if not any(noun in value for noun in GENERATE_PROJECT_NOUNS):
        return False
    if looks_like_push(value):
        return False
    return any(verb in value for verb in GENERATE_PROJECT_VERBS)


def extract_generate_project_framework(text: str) -> str:
    value = text.lower()
    for framework in GENERATE_PROJECT_FRAMEWORKS:
        if framework in value:
            return framework
    return "streamlit"


def looks_like_aws_deploy(text: str) -> bool:
    value = text.lower()
    return "aws" in value and looks_like_deploy(value)


def looks_like_ec2_deploy(text: str) -> bool:
    value = text.lower()
    return looks_like_deploy(value) and any(
        term in value for term in ("aws", "ec2", "amazon web services", "amazon server")
    )


def looks_like_open_repo(text: str) -> bool:
    value = text.lower()
    open_words = (
        "open",
        "show",
        "display",
        "view",
        "khol",
        "kholo",
        "dikhao",
        "दिखाओ",
        "खोलो",
    )
    has_repo_word = bool(re.search(r"\b(repo|repos|repository|repositories)\b", value))
    has_owner_repo_path = bool(re.search(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\b", value))
    return any(word in value for word in open_words) and (has_repo_word or has_owner_repo_path)


def looks_like_settings(text: str) -> bool:
    value = text.lower()
    settings_words = ("setting", "settings", "सेटिंग", "सेटिंग्स", "config")
    action_words = (
        "open", "show", "go to", "goto", "jao", "khol", "kholo", "dikhao",
        "change", "update", "badal", "kar do", "karo", "on", "off",
        "private", "public",
    )
    return any(word in value for word in settings_words) and any(
        word in value for word in action_words
    )


SETTINGS_KEYWORD_PATTERN = re.compile(
    r"\b(settings|setting|config|सेटिंग्स|सेटिंग)\b", flags=re.IGNORECASE
)


def extract_settings_repo_name(text: str) -> Optional[str]:
    """Extract the repo name out of a settings command, ignoring the word
    'settings' itself so it is never mistaken for the repo name."""
    stripped = SETTINGS_KEYWORD_PATTERN.sub(" ", text)
    return extract_open_repo_name(stripped)


def extract_settings_changes(text: str) -> Dict[str, Any]:
    """Detect explicit settings changes from natural Hindi/Hinglish/English text."""
    value = text.lower()
    changes: Dict[str, Any] = {}

    if re.search(r"\bpublic\b", value) and re.search(r"\bpublic\s+(kar|karo|kar do|कर\s*दो)\b|\bmake\s+it\s+public\b|\bset\s+to\s+public\b", value):
        changes["private"] = False
    elif re.search(r"\bpublic\b", value) and not re.search(r"\bprivate\b", value) and any(
        w in value for w in ("kar do", "karo", "kar deejiye", "banao")
    ):
        changes["private"] = False
    if re.search(r"\bprivate\b", value) and any(
        w in value for w in ("kar do", "karo", "kar deejiye", "banao", "make it private", "set to private")
    ):
        changes["private"] = True

    for feature, key in (("issues", "has_issues"), ("wiki", "has_wiki"), ("projects", "has_projects")):
        off_match = re.search(rf"\b{feature}\b[^.]{{0,15}}\b(off|band|disable|hata)\b", value)
        on_match = re.search(rf"\b{feature}\b[^.]{{0,15}}\b(on|chalu|enable|shuru)\b", value)
        if off_match:
            changes[key] = False
        elif on_match:
            changes[key] = True

    description_match = re.search(
        r"description\s*(?:change|update|badal)?(?:\s+karke|\s+to|\s*[:=]\s*)?\s*[\"']?([^\"'\n]{3,150})",
        text,
        flags=re.IGNORECASE,
    )
    if description_match:
        changes["description"] = description_match.group(1).strip()

    branch_match = re.search(
        r"default\s+branch\s*(?:change|update|badal)?(?:\s+karke|\s+to|\s*[:=]\s*)?\s*[\"']?([A-Za-z0-9._/-]{1,80})",
        text,
        flags=re.IGNORECASE,
    )
    if branch_match:
        changes["default_branch"] = branch_match.group(1).strip("`'\" ")

    return changes


def extract_open_repo_name(text: str) -> Optional[str]:
    repo_pattern = r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)"
    explicit_path = re.search(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\b", text)
    if explicit_path:
        return explicit_path.group(0)
    patterns = [
        rf"(?:open|show|display|view|deploy|delete|remove|erase|khol(?:o)?|dikhao|खोलो|दिखाओ|डिप्लॉय|डिलीट)\s+(?:the\s+|my\s+|mera\s+|meri\s+|to\s+)?(?:repo(?:sitory)?\s+)?{repo_pattern}",
        rf"{repo_pattern}\s+(?:ka|ki|का|की)?\s*(?:repo|repository)\s+(?:open|show|display|view|khol|खोल)",
        rf"(?:repo|repository)\s+{repo_pattern}\s+(?:open|show|display|view|khol|खोल)",
    ]
    ignored = {
        "repo",
        "repos",
        "repository",
        "repositories",
        "kro",
        "please",
        "na",
        "this",
        "project",
        "app",
        "to",
        "aws",
        "streamlit",
        "cloud",
    }
    # A local filename (a ZIP upload, a source file, ...) is never itself a
    # GitHub repo name — deployment/opening always targets an actual repo
    # that has been pushed, not the raw uploaded file. Reject these so a
    # phrase like "deploy myproject.zip" doesn't get treated as a real repo.
    non_repo_extensions = (
        ".zip", ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".png",
        ".jpg", ".jpeg", ".exe", ".csv", ".pdf", ".h5", ".pkl", ".ipynb",
    )
    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip("`'\" ")
            if candidate.lower() in ignored:
                continue
            if candidate.lower().endswith(non_repo_extensions):
                continue
            return candidate
    return None


def extract_rename_details(text: str) -> Optional[Dict[str, str]]:
    """Extract source and destination from common English/Hinglish phrasing."""
    repo_pattern = r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)"
    patterns = [
        rf"(?:rename|renamed|renarte)\s+(?:the\s+)?(?:repo(?:sitory)?\s+)?"
        rf"{repo_pattern}(?:\s+repo(?:sitory)?)?\s+"
        rf"(?:to|as|into)\s+(.+)",
        rf"{repo_pattern}(?:\s+repo(?:sitory)?)?\s+"
        rf"(?:ka\s+naam|का\s+नाम|name)\s+(?:is|=|:|rakh(?:o)?|rakho)?\s*(.+)",
        rf"(?:repo(?:sitory)?\s+)?{repo_pattern}\s+"
        rf"(?:ka\s+naam|का\s+नाम|name)\s+(?:change|badal)(?:\s+karke|\s+to)?\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        repo_name = match.group(1).strip("`'\" ")
        new_name = match.group(2).strip("`'\" .,")
        new_name = re.sub(
            r"^(?:change|badal)(?:\s+karke|\s+to)?\s+",
            "",
            new_name,
            flags=re.IGNORECASE,
        )
        new_name = re.sub(
            r"\s+(?:karo|kar\s+do|rakho|please|na|ji)$",
            "",
            new_name,
            flags=re.IGNORECASE,
        )
        normalized_name = normalize_repo_name(new_name)
        if repo_name and normalized_name:
            return {"repo": repo_name, "new_name": normalized_name}
    return None


def extract_repo_name(text: str) -> Optional[str]:
    value = text.strip().strip("`'\" ")
    patterns = [
        r"(?:yes\s+)?(?:new|create|make|bana(?:o)?|बना)\s+(?:a\s+)?repo(?:sitory)?\s+(?:named\s+|name\s+|naam\s+|is\s+)?[`'\"]?([A-Za-z0-9][A-Za-z0-9._-]{0,99})",
        r"(?:repo(?:sitory)?(?:\s+(?:ka|का))?\s+(?:name|naam)|नाम)\s*(?:is|=|:|रखो|rakho)?\s*[`'\"]?([A-Za-z0-9][A-Za-z0-9._-]{0,99})",
        r"(?:use|call it|name it|naam)\s*[`'\"]?([A-Za-z0-9][A-Za-z0-9._-]{0,99})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value):
        return value
    return None


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"reply": cleaned, "action": None}
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else {"reply": cleaned, "action": None}
            except json.JSONDecodeError:
                pass
    return {"reply": text.strip(), "action": None}


def api_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                return str(error.get("message", error))
            return str(error)
    except ValueError:
        pass
    return response.text[:500] or f"HTTP {response.status_code}"


class ApiCallError(Exception):
    """Raised when an AI provider returns a non-2xx response.

    Carries the HTTP status code alongside the provider's error message so
    call_ai() can tell a rate limit apart from a retired/unknown model and
    react accordingly instead of just giving up.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def raise_for_response(response: requests.Response) -> None:
    if not response.ok:
        raise ApiCallError(response.status_code, api_error(response))


def call_openai_compatible(endpoint: str, key: str, model: str, messages: List[Dict[str, str]]) -> str:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    response = requests.post(
        endpoint,
        headers=headers,
        json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1200},
        timeout=90,
    )
    raise_for_response(response)
    return str(response.json()["choices"][0]["message"]["content"])


def call_anthropic(key: str, model: str, messages: List[Dict[str, str]]) -> str:
    system = "\n\n".join(
        message["content"] for message in messages if message["role"] == "system"
    ) or SYSTEM_PROMPT
    api_messages = [message for message in messages if message["role"] != "system"]
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": model, "system": system, "max_tokens": 1200, "messages": api_messages},
        timeout=90,
    )
    raise_for_response(response)
    blocks = response.json().get("content", [])
    return "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")


def call_gemini(key: str, model: str, messages: List[Dict[str, str]]) -> str:
    contents = []
    system_text = "\n\n".join(
        message["content"] for message in messages if message["role"] == "system"
    ) or SYSTEM_PROMPT
    for message in messages:
        if message["role"] == "system":
            continue
        contents.append(
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
        )
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1200},
        },
        timeout=90,
    )
    raise_for_response(response)
    return str(response.json()["candidates"][0]["content"]["parts"][0]["text"])


def models_endpoint_for(endpoint: str) -> str:
    base = re.sub(r"/chat/completions/?$", "", endpoint.strip())
    return base.rstrip("/") + "/models"


def fetch_models_openai_compatible(endpoint: str, key: str) -> List[str]:
    """List model ids from an OpenAI-compatible /v1/models endpoint."""
    try:
        response = requests.get(
            models_endpoint_for(endpoint),
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        if not response.ok:
            return []
        data = response.json().get("data", [])
        return [str(item.get("id")) for item in data if item.get("id")]
    except Exception:
        return []


def fetch_models_anthropic(key: str) -> List[str]:
    try:
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=30,
        )
        if not response.ok:
            return []
        data = response.json().get("data", [])
        return [str(item.get("id")) for item in data if item.get("id")]
    except Exception:
        return []


def fetch_models_gemini(key: str) -> List[str]:
    try:
        response = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key},
            timeout=30,
        )
        if not response.ok:
            return []
        names = []
        for item in response.json().get("models", []):
            if "generateContent" in item.get("supportedGenerationMethods", []):
                name = str(item.get("name", "")).split("/")[-1]
                if name:
                    names.append(name)
        return names
    except Exception:
        return []


def fetch_available_models(provider: Dict[str, Any], key: str, endpoint: str) -> List[str]:
    if provider["kind"] == "anthropic":
        return fetch_models_anthropic(key)
    if provider["kind"] == "gemini":
        return fetch_models_gemini(key)
    if endpoint:
        return fetch_models_openai_compatible(endpoint, key)
    return []


def pick_fallback_model(models: List[str], exclude: set) -> Optional[str]:
    """Pick a reasonable chat model out of a provider's /models listing.

    Prefers small/fast/cheap models (flash, mini, haiku, etc.) so an
    automatic recovery never silently jumps someone onto their most
    expensive available model.
    """
    candidates = [
        model
        for model in models
        if model
        and model not in exclude
        and not any(bad in model.lower() for bad in NON_CHAT_MODEL_KEYWORDS)
    ]
    if not candidates:
        return None
    for keyword in PREFERRED_FALLBACK_KEYWORDS:
        matches = sorted((model for model in candidates if keyword in model.lower()), reverse=True)
        if matches:
            return matches[0]
    return sorted(candidates, reverse=True)[0]


def extract_suggested_model(message: str) -> Optional[str]:
    """Pull a replacement model name out of a provider's own error text.

    Providers increasingly tell you exactly what to switch to, e.g. "please
    update your code to use models/gemini-3.6-flash" or "migrate to
    openai/gpt-oss-20b" -- reuse that instead of guessing.
    """
    patterns = [
        r"use\s+models?/([A-Za-z0-9_.\-]+)",
        r"use\s+`?([A-Za-z0-9_./\-]+)`?\s+instead",
        r"replaced?\s+by\s+`?([A-Za-z0-9_./\-]+)`?",
        r"replacement model(?:\s+id)?[:\s]+`?([A-Za-z0-9_./\-]+)`?",
        r"migrat(?:e|ing) to\s+`?([A-Za-z0-9_./\-]+)`?",
        r"try\s+`?([A-Za-z0-9_./\-]+)`?\s+instead",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(").,`'\"")
            if candidate:
                return candidate
    return None


def extract_retry_delay_seconds(message: str, cap: float = 12.0) -> float:
    """Google's 429 errors usually include an exact suggested wait (e.g.
    'Please retry in 16.67s' or '108.049004ms'). Honor that instead of a
    blind fixed backoff, so a real retry has a better chance of succeeding —
    capped so the UI never blocks for an unreasonably long time."""
    match = re.search(r"retry in\s*([\d.]+)\s*(ms|s)\b", message, flags=re.IGNORECASE)
    if not match:
        return 0.0
    value = float(match.group(1))
    if match.group(2).lower() == "ms":
        value /= 1000.0
    return min(max(value, 0.0), cap)


def is_rate_limited(status_code: Optional[int], message: str) -> bool:
    if status_code == 429:
        return True
    lowered = message.lower()
    return "rate limit" in lowered or "too many requests" in lowered


def is_model_unavailable(status_code: Optional[int], message: str) -> bool:
    lowered = message.lower()
    keywords = (
        "no longer available",
        "not found",
        "does not exist",
        "deprecated",
        "decommissioned",
        "has been retired",
        "unknown model",
        "invalid model",
        "not supported",
    )
    return status_code in (400, 404) or any(keyword in lowered for keyword in keywords)


def dispatch_ai_call(
    provider: Dict[str, Any], key: str, model: str, endpoint: str, messages: List[Dict[str, str]]
) -> str:
    if provider["kind"] == "anthropic":
        return call_anthropic(key, model, messages)
    if provider["kind"] == "gemini":
        return call_gemini(key, model, messages)
    return call_openai_compatible(endpoint, key, model, messages)


MAX_AI_ATTEMPTS = 3


def build_session_context_note() -> str:
    """Summarize what the app already knows so the AI can resolve implicit
    references ("isko deploy kardo", "usi repo mein push karo") instead of
    asking the user to repeat information that is already available."""
    lines: List[str] = []
    gh_user = st.session_state.get("gh_user")
    lines.append(
        f"GitHub connected as: {gh_user}" if gh_user else "GitHub: not connected yet."
    )
    if st.session_state.get("current_repo"):
        lines.append(f"Repository currently open/in view: {st.session_state.current_repo}")
    if st.session_state.get("last_pushed_repo"):
        lines.append(f"Most recently pushed repository: {st.session_state.last_pushed_repo}")
    if st.session_state.get("project_files"):
        lines.append(
            f"A project ZIP is already uploaded ({len(st.session_state.project_files)} files)."
        )
        analysis = st.session_state.get("project_analysis")
        if isinstance(analysis, dict):
            lines.append(
                "Detected framework: {framework}; entrypoint: {entry}; "
                "recommended deploy platform: {platform}.".format(
                    framework=analysis.get("framework", "Unknown"),
                    entry=analysis.get("entrypoint") or "none found",
                    platform=analysis.get("recommended_platform", "unknown"),
                )
            )
    else:
        lines.append("No project ZIP uploaded yet.")
    if st.session_state.get("ec2_runner_label"):
        lines.append(f"AWS EC2 runner label configured: {st.session_state.ec2_runner_label}")
    live = st.session_state.get("live_view")
    if isinstance(live, dict) and live.get("kind") == "aws_ec2":
        live_data = live.get("data", {})
        if live_data.get("url"):
            lines.append(f"Last successful AWS EC2 live URL: {live_data.get('url')}")
    if isinstance(st.session_state.get("pending_deployment"), dict):
        lines.append(
            f"A deployment for {st.session_state.pending_deployment.get('repo')} "
            "is waiting only on a platform choice."
        )
    return "Current app state (use this to fill in missing details instead of " \
        "asking again when it's reasonably clear what the user means):\n- " + \
        "\n- ".join(lines)


def call_ai(user_text: str) -> Dict[str, Any]:
    key = st.session_state.api_key.strip()
    if not key:
        return {
            "reply": (
                "Free-language chat ke liye sidebar mein provider select karke "
                "API key add karo. GitHub token alag se connect karna hoga."
            ),
            "action": None,
        }

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]
    history.append({"role": "user", "content": user_text})
    provider = PROVIDERS[st.session_state.provider]
    model = selected_model()
    endpoint = (
        st.session_state.custom_endpoint.strip()
        if provider["kind"] == "custom"
        else provider.get("endpoint", "")
    )
    if provider["kind"] == "custom" and not endpoint:
        return {"reply": "Custom provider ka endpoint add karo.", "action": None}

    context_note = build_session_context_note()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": context_note},
    ] + history
    attempted_models = set()
    notice = ""

    for attempt in range(MAX_AI_ATTEMPTS):
        attempted_models.add(model)
        try:
            raw = dispatch_ai_call(provider, key, model, endpoint, messages)
            if model != st.session_state.model:
                st.session_state.model = model
            result = parse_json_response(raw)
            if notice and isinstance(result, dict):
                reply = str(result.get("reply") or "")
                result["reply"] = f"{notice}\n\n{reply}".strip()
            return result
        except requests.RequestException as exc:
            return {"reply": f"AI service se connection nahi hua: {exc}", "action": None}
        except ApiCallError as exc:
            is_last_attempt = attempt == MAX_AI_ATTEMPTS - 1
            if not is_last_attempt and is_model_unavailable(exc.status_code, exc.message):
                suggested = extract_suggested_model(exc.message)
                if not suggested or suggested in attempted_models:
                    available = fetch_available_models(provider, key, endpoint)
                    suggested = pick_fallback_model(available, attempted_models)
                if suggested and suggested not in attempted_models:
                    notice = (
                        f"⚠️ Model `{model}` available nahi hai ({exc.message}). "
                        f"Automatically `{suggested}` try kar raha hoon."
                    )
                    model = suggested
                    continue
            if not is_last_attempt and is_rate_limited(exc.status_code, exc.message):
                delay = extract_retry_delay_seconds(exc.message)
                time.sleep(delay if delay > 0 else 2 * (attempt + 1))
                continue
            return {"reply": f"AI request fail hui: {exc.message}", "action": None}
        except Exception as exc:
            return {"reply": f"AI request fail hui: {exc}", "action": None}

    return {
        "reply": "AI request fail hui: kai attempts ke baad bhi koi model kaam nahi kar raha.",
        "action": None,
    }


PROJECT_GENERATION_SYSTEM_PROMPT = """
You are an expert {framework} developer. Write a complete, working {framework}
project based on the request below, the same way you would when asked to
write code directly for someone.

Rules:
- The project must actually run: no placeholder TODOs, no missing imports,
  no undefined functions or variables.
- Always include a working entrypoint file (for Streamlit, "app.py"), a
  "requirements.txt" listing every third-party package actually imported, and
  a short "README.md" explaining what the app does and how to run it.
- Keep the app reasonably self-contained and prefer well-known, commonly
  available packages.
- Reply in the same language as the request for any comments, but code
  identifiers should stay in English as usual.

Return ONLY valid JSON, with this exact shape and nothing else (no markdown
fences, no commentary outside the JSON):
{{
  "files": {{
    "app.py": "full file content as a single string, with real newline characters escaped as \\n",
    "requirements.txt": "...",
    "README.md": "..."
  }}
}}
Every key is a relative file path and every value is the full text content of
that file.
"""


def generate_project_code(description: str, framework: str) -> Dict[str, Any]:
    """Ask the connected AI provider to write a full new project as JSON files."""
    key = st.session_state.api_key.strip()
    if not key:
        return {
            "error": (
                "Naya project generate karne ke liye sidebar mein AI provider "
                "select karke API key add karo."
            )
        }
    provider = PROVIDERS[st.session_state.provider]
    model = selected_model()
    endpoint = (
        st.session_state.custom_endpoint.strip()
        if provider["kind"] == "custom"
        else provider.get("endpoint", "")
    )
    if provider["kind"] == "custom" and not endpoint:
        return {"error": "Custom provider ka endpoint add karo."}

    system = PROJECT_GENERATION_SYSTEM_PROMPT.format(framework=framework)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": description},
    ]
    attempted_models = set()

    for attempt in range(MAX_AI_ATTEMPTS):
        attempted_models.add(model)
        try:
            raw = dispatch_ai_call(provider, key, model, endpoint, messages)
            if model != st.session_state.model:
                st.session_state.model = model
            parsed = parse_json_response(raw)
            files = parsed.get("files") if isinstance(parsed, dict) else None
            if not isinstance(files, dict) or not files:
                return {
                    "error": (
                        "AI ne valid project files generate nahi kiye. Description "
                        "thoda aur clear karke dobara try karo."
                    )
                }
            return {"files": files}
        except requests.RequestException as exc:
            return {"error": f"AI service se connection nahi hua: {exc}"}
        except ApiCallError as exc:
            is_last_attempt = attempt == MAX_AI_ATTEMPTS - 1
            if not is_last_attempt and is_model_unavailable(exc.status_code, exc.message):
                suggested = extract_suggested_model(exc.message)
                if not suggested or suggested in attempted_models:
                    available = fetch_available_models(provider, key, endpoint)
                    suggested = pick_fallback_model(available, attempted_models)
                if suggested and suggested not in attempted_models:
                    model = suggested
                    continue
            if not is_last_attempt and is_rate_limited(exc.status_code, exc.message):
                delay = extract_retry_delay_seconds(exc.message)
                time.sleep(delay if delay > 0 else 2 * (attempt + 1))
                continue
            return {"error": f"AI request fail hui: {exc.message}"}
        except Exception as exc:
            return {"error": f"AI request fail hui: {exc}"}

    return {"error": "AI request fail hui: kai attempts ke baad bhi koi model kaam nahi kar raha."}


def files_dict_to_project_files(files: Dict[str, Any]) -> List[Dict[str, Any]]:
    project_files: List[Dict[str, Any]] = []
    for raw_path, content in files.items():
        path = safe_zip_path(str(raw_path))
        if not path:
            continue
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        project_files.append({"path": path, "content": str(content).encode("utf-8")})
    return project_files


def start_project_generation(description: str, framework: str = "streamlit", repo_name: str = "") -> str:
    """Generate a brand-new project with AI, then hand it to the existing push flow."""
    clean_description = description.strip() or f"A simple, useful {framework} app."
    with st.spinner(f"{framework.capitalize()} project generate ho raha hai..."):
        result = generate_project_code(clean_description, framework)
    if "error" in result:
        return result["error"]

    project_files = files_dict_to_project_files(result["files"])
    if not project_files:
        return "AI se koi valid file generate nahi hui. Description thoda aur clear karke dobara try karo."

    st.session_state.project_files = project_files
    st.session_state.project_zip_name = f"Generated {framework} project"
    st.session_state.uploaded_signature = None
    st.session_state.project_analysis = analyze_project_files(project_files)
    st.session_state.last_push_report = None
    st.session_state.pending_push = False
    st.session_state.auto_deploy_after_push = True

    file_list = "\n".join(f"- 📄 `{item['path']}`" for item in project_files)
    summary = f"**{framework.capitalize()} project ready ho gaya:**\n\n{file_list}"

    if st.session_state.gh_user is None:
        return (
            f"{summary}\n\nAb GitHub par push karne ke liye pehle sidebar se GitHub "
            "token connect karo, phir bolo `push karo`."
        )

    clean_repo_name = normalize_repo_name(repo_name) if repo_name else ""
    if clean_repo_name:
        action = {
            "type": "push_project",
            "repo_name": clean_repo_name,
            "private": st.session_state.private_repo,
            "commit_message": st.session_state.commit_message,
        }
        st.session_state.pending_confirmation = action
        return f"{summary}\n\n{confirmation_preview(action)}"

    st.session_state.pending_push = True
    return f"{summary}\n\nGitHub par push karne ke liye repository ka naam batao."


def request_deployment(repo_name: str, platform: Optional[str] = None) -> str:
    if not st.session_state.project_analysis and st.session_state.gh_client is not None:
        try:
            st.session_state.project_analysis = gh_analyze_repo(repo_name)
        except Exception:
            # Deployment can still continue with the user's explicit platform
            # even when GitHub metadata cannot be read.
            pass
    chosen_platform = platform
    if not chosen_platform:
        selected = st.session_state.deployment_platform
        if selected != ASK_DEPLOYMENT_PLATFORM:
            chosen_platform = selected
    if not chosen_platform:
        st.session_state.pending_deployment = {"repo": repo_name, "suggested_platform": ""}
        return deployment_platform_prompt(st.session_state.project_analysis)

    action = deployment_action_for(chosen_platform, repo_name)
    st.session_state.pending_deployment = None
    st.session_state.pending_confirmation = action
    return confirmation_preview(action)


def extract_deployment_repo_name(text: str) -> Optional[str]:
    """Extract a repo after a platform name, e.g. `deploy Render my-app`."""
    candidate = extract_open_repo_name(text)
    platform = extract_deployment_platform(text)
    if platform:
        without_platform = text
        aliases = [
            alias
            for alias, mapped_platform in DEPLOYMENT_PLATFORM_ALIASES.items()
            if mapped_platform == platform
        ]
        for alias in sorted(aliases, key=len, reverse=True):
            without_platform = re.sub(
                rf"\b{re.escape(alias)}\b",
                "",
                without_platform,
                count=1,
                flags=re.IGNORECASE,
            )
            if without_platform != text:
                break
        cleaned_candidate = extract_open_repo_name(without_platform)
        if cleaned_candidate:
            candidate = cleaned_candidate
    return candidate


def is_ai_quota_or_outage_error(reply_text: str) -> bool:
    lowered = reply_text.lower()
    return lowered.startswith("ai request fail hui") and any(
        marker in lowered
        for marker in (
            "quota", "rate limit", "429", "resource_exhausted",
            "too many requests", "connection nahi hua",
        )
    )


def handle_message(user_text: str) -> str:
    pending_confirmation = st.session_state.pending_confirmation
    if isinstance(pending_confirmation, dict):
        if is_confirmation_yes(user_text):
            st.session_state.pending_confirmation = None
            emit_action_event("approval_received", "approved", "Approval received",
                              "User explicitly approved this action.", pending_confirmation)
            result = execute_action(pending_confirmation)
            emit_action_event("action_completed" if not str(result).lower().startswith(("error", "action failed", "repository", "file")) else "api_response",
                              "completed", "Action finished", "Execution result recorded.", pending_confirmation)
            st.session_state.action_history.append({
                "action_id": st.session_state.active_action_id,
                "type": pending_confirmation.get("type"),
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            should_auto_deploy = (
                pending_confirmation.get("type") == "push_project"
                and st.session_state.auto_deploy_after_push
            )
            st.session_state.auto_deploy_after_push = False
            if should_auto_deploy:
                push_report = st.session_state.last_push_report or {}
                pushed_ok = bool(push_report.get("created") or push_report.get("updated"))
                if pushed_ok and st.session_state.last_pushed_repo:
                    try:
                        deploy_info = request_deployment(
                            st.session_state.last_pushed_repo
                        )
                        result += "\n\n---\n\n" + deploy_info
                    except Exception as exc:
                        result += (
                            "\n\nDeployment option automatically prepare nahi ho paaya: "
                            f"{github_error(exc)}"
                        )
            return result
        if is_confirmation_no(user_text):
            st.session_state.pending_confirmation = None
            st.session_state.auto_deploy_after_push = False
            return "Theek hai, action cancel kar diya."
        return "Please `yes`/`confirm` ya `no`/`cancel` likho."

    pending_deployment = st.session_state.pending_deployment
    if isinstance(pending_deployment, dict):
        if is_confirmation_no(user_text):
            st.session_state.pending_deployment = None
            return "Theek hai, deployment request cancel kar diya."
        platform = extract_deployment_platform(user_text)
        if not platform and is_confirmation_yes(user_text):
            suggested = pending_deployment.get("suggested_platform")
            if suggested in DEPLOYMENT_PLATFORM_OPTIONS:
                platform = suggested
        if not platform:
            return deployment_platform_prompt(st.session_state.project_analysis)
        return request_deployment(str(pending_deployment.get("repo", "")), platform)

    if st.session_state.pending_push:
        if user_text.strip().lower() in {"cancel", "cancel karo", "no", "nahi"}:
            st.session_state.pending_push = False
            return "Theek hai, GitHub push cancel kar diya."
        repo_name = extract_repo_name(user_text)
        if not repo_name:
            return "Repo ka naam clearly batao, jaise `my-portfolio`."
        st.session_state.pending_push = False
        action = {
            "type": "push_project",
            "repo_name": repo_name,
            "private": st.session_state.private_repo,
            "commit_message": st.session_state.commit_message,
        }
        st.session_state.pending_confirmation = action
        emit_action_event("approval_requested", "waiting", "Approval required", "Review the action and approve or deny it.", action)
        return confirmation_preview(action)

    # Prefer real language understanding over keyword-matching whenever an AI
    # key is configured. The rule-based `looks_like_*` checks below exist as a
    # fallback so the app still does something useful without an API key, if
    # the AI call itself fails, or if the AI provider's quota/rate limit is
    # exhausted (free tiers are easy to hit once every message calls the AI).
    quota_notice = ""
    if st.session_state.api_key.strip():
        ai_result = call_ai(user_text)
        action = ai_result.get("action")
        if isinstance(action, dict) and action.get("type"):
            if action.get("type") == "deploy_aws":
                action["type"] = "deploy_aws_ec2"
                action.setdefault("port", "")
                action.setdefault("instance_id", st.session_state.get("aws_ec2_instance_id", ""))
                action.pop("workflow", None)
                action.pop("inputs", None)
            if action.get("type") == "generate_project":
                return start_project_generation(
                    str(action.get("description") or user_text),
                    str(action.get("framework") or "streamlit"),
                    str(action.get("repo_name") or ""),
                )
            if needs_confirmation(action):
                st.session_state.pending_confirmation = action
                emit_action_event("approval_requested", "waiting", "Approval required", "Review the action and approve or deny it.", action)
                return confirmation_preview(action)
            return execute_action(action)
        reply = str(ai_result.get("reply") or "").strip()
        if reply and is_ai_quota_or_outage_error(reply):
            # Don't show the raw quota/outage error for every single message —
            # quietly fall back to rule-based matching below so basic commands
            # (push, deploy, list repos, settings, ...) keep working. Only
            # surface the quota notice if the rule-based fallback also can't
            # figure out what was meant.
            quota_notice = (
                "⚠️ AI (free tier) ka quota abhi khatam hai, thodi der baad "
                "phir try karna — tab tak basic commands (push, deploy, "
                "list repos, settings, analyze) keyword-based tarike se "
                "chalte rahenge.\n\n"
            )
        elif reply:
            return reply
        # Fall through to the rule-based matching below only if the AI
        # returned nothing usable, or its quota/outage error was suppressed.

    return quota_notice + rule_based_reply(user_text)


def rule_based_reply(user_text: str) -> str:
    if looks_like_generate_project(user_text):
        framework = extract_generate_project_framework(user_text)
        repo_hint = extract_repo_name(user_text) or ""
        return start_project_generation(user_text, framework=framework, repo_name=repo_hint)

    if looks_like_analyze(user_text):
        return project_analysis_text(st.session_state.project_analysis)

    if looks_like_create_repo(user_text):
        if st.session_state.gh_user is None:
            return "Pehle sidebar se GitHub token connect karo."
        repo_name = extract_repo_name(user_text)
        if not repo_name:
            return "Naye repository ka naam batao, jaise `create repo my-agent`."
        action = {
            "type": "create_repo",
            "name": repo_name,
            "owner": "",
            "private": st.session_state.private_repo,
            "description": "",
        }
        st.session_state.pending_confirmation = action
        emit_action_event("approval_requested", "waiting", "Approval required", "Review the action and approve or deny it.", action)
        return confirmation_preview(action)

    if looks_like_rename_repo(user_text):
        if st.session_state.gh_user is None:
            return "Pehle sidebar se GitHub token connect karo."
        rename_details = extract_rename_details(user_text)
        if not rename_details:
            return (
                "Rename ke liye source aur naya naam dono batao, jaise "
                "`rename cancer repo to cancer-project`."
            )
        action = {"type": "rename_repo", **rename_details}
        st.session_state.pending_confirmation = action
        emit_action_event("approval_requested", "waiting", "Approval required", "Review the action and approve or deny it.", action)
        return confirmation_preview(action)

    if looks_like_delete_repo(user_text):
        if st.session_state.gh_user is None:
            return "Pehle sidebar se GitHub token connect karo."
        repo_name = extract_open_repo_name(user_text)
        if not repo_name:
            return "Kaunsa repository delete karna hai? Naam batao, jaise `delete repo old-agent`."
        action = {"type": "delete_repo", "repo": repo_name}
        st.session_state.pending_confirmation = action
        emit_action_event("approval_requested", "waiting", "Approval required", "Review the action and approve or deny it.", action)
        return confirmation_preview(action)

    if looks_like_list_repos(user_text):
        if st.session_state.gh_user is None:
            return "Pehle sidebar se GitHub token connect karo."
        try:
            return gh_list_repos()
        except Exception as exc:
            return f"Repositories list nahi ho paayi: {github_error(exc)}"

    if looks_like_deploy(user_text):
        repo_name = (
            extract_deployment_repo_name(user_text)
            or st.session_state.last_pushed_repo
            or st.session_state.current_repo
        )
        if not repo_name:
            return (
                "Deployment se pehle project ko GitHub repository mein push karo, "
                "ya message mein `deploy Render owner/repo` jaisa repository name do."
            )
        return request_deployment(
            repo_name,
            extract_deployment_platform(user_text),
        )

    if looks_like_open_repo(user_text):
        if st.session_state.gh_user is None:
            return "Pehle sidebar se GitHub token connect karo."
        repo_name = extract_open_repo_name(user_text)
        if not repo_name:
            return "Kaunsa repository open karna hai? Naam batao, jaise `cancer`."
        try:
            return gh_open_repo(repo_name)
        except Exception as exc:
            return f"Repository open nahi ho paayi: {github_error(exc)}"

    if looks_like_settings(user_text):
        if st.session_state.gh_user is None:
            return "Pehle sidebar se GitHub token connect karo."
        repo_name = (
            extract_settings_repo_name(user_text)
            or st.session_state.current_repo
            or st.session_state.last_pushed_repo
        )
        if not repo_name:
            return "Kaunse repository ki settings kholni hai? Naam batao, jaise `cancer settings kholo`."
        changes = extract_settings_changes(user_text)
        if changes:
            action = {"type": "update_repo_settings", "repo": repo_name, "changes": changes}
            st.session_state.pending_confirmation = action
            emit_action_event("approval_requested", "waiting", "Approval required", "Review the action and approve or deny it.", action)
            return confirmation_preview(action)
        try:
            return gh_repo_settings(repo_name)
        except Exception as exc:
            return f"Settings load nahi ho paayi: {github_error(exc)}"

    if looks_like_push(user_text):
        if not st.session_state.project_files:
            return "Pehle sidebar ke **Project ZIP Upload** section mein apna project ZIP upload karo."
        if st.session_state.gh_user is None:
            return "Pehle sidebar se GitHub token connect karo."
        st.session_state.pending_push = True
        return "Bilkul. Push karne se pehle batao — GitHub repository ka kya naam rakhoon?"

    return "Main samajh nahi paaya. Thoda differently batao, ya sidebar mein AI API key add karo taaki main free-language commands bhi samajh sakoon."


GITHUB_VIEW_CSS = """
<style>
.ghv-box {background:#0d1117;border:1px solid #30363d;border-radius:8px;
  padding:0;overflow:hidden;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
  color:#c9d1d9;margin-bottom:10px;}
.ghv-header {background:#161b22;border-bottom:1px solid #30363d;padding:12px 16px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;}
.ghv-title {font-size:16px;font-weight:600;color:#58a6ff;}
.ghv-badge {font-size:11px;border:1px solid #30363d;border-radius:2em;padding:1px 8px;
  color:#8b949e;margin-left:6px;}
.ghv-tabs {display:flex;gap:2px;background:#161b22;border-bottom:1px solid #30363d;
  padding:0 12px;flex-wrap:wrap;}
.ghv-tab {padding:9px 12px;font-size:13px;color:#8b949e;border-bottom:2px solid transparent;}
.ghv-tab.active {color:#c9d1d9;border-bottom:2px solid #f78166;font-weight:600;}
.ghv-body {padding:14px 16px;}
.ghv-row {display:flex;justify-content:space-between;align-items:center;
  padding:8px 0;border-bottom:1px solid #21262d;font-size:13px;}
.ghv-row:last-child {border-bottom:none;}
.ghv-name {color:#58a6ff;}
.ghv-meta {color:#8b949e;font-size:12px;}
.ghv-empty {color:#8b949e;font-size:13px;padding:24px;text-align:center;
  border:1px dashed #30363d;border-radius:8px;background:#0d1117;}
.ghv-readme {background:#0d1117;border:1px solid #21262d;border-radius:6px;
  padding:12px;margin-top:10px;font-size:12px;white-space:pre-wrap;
  max-height:220px;overflow-y:auto;color:#c9d1d9;}
.ghv-setting-row {display:flex;justify-content:space-between;align-items:center;
  padding:10px 0;border-bottom:1px solid #21262d;}
.ghv-toggle-on {color:#3fb950;font-weight:600;font-size:12px;}
.ghv-toggle-off {color:#8b949e;font-weight:600;font-size:12px;}
.ghv-code {background:#0d1117;border:1px solid #21262d;border-radius:6px;
  padding:12px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:12px;white-space:pre-wrap;max-height:260px;overflow-y:auto;}
.ghv-issue-open {color:#3fb950;}
.ghv-issue-closed {color:#a371f7;}
</style>
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _ghv_header(full_name: str, right_badges: List[str], active_tab: str) -> str:
    badges_html = "".join(f'<span class="ghv-badge">{_esc(badge)}</span>' for badge in right_badges)
    tabs = ["Code", "Issues", "Pull requests", "Actions", "Releases", "Settings"]
    tabs_html = "".join(
        f'<span class="ghv-tab{" active" if tab == active_tab else ""}">{tab}</span>'
        for tab in tabs
    )
    return (
        '<div class="ghv-box">'
        f'<div class="ghv-header"><span class="ghv-title">📦 {_esc(full_name)}</span>'
        f'<span>{badges_html}</span></div>'
        f'<div class="ghv-tabs">{tabs_html}</div>'
        '<div class="ghv-body">'
    )


_GHV_FOOTER = "</div></div>"


def render_repo_view(data: Dict[str, Any]) -> None:
    badges = ["🔒 Private" if data.get("private") else "🌐 Public"]
    badges.append(f"⭐ {data.get('stars', 0)}")
    badges.append(f"🍴 {data.get('forks', 0)}")
    parts = [_ghv_header(data["full_name"], badges, "Code")]
    if data.get("description"):
        parts.append(f'<div class="ghv-meta" style="margin-bottom:10px;">{_esc(data["description"])}</div>')
    parts.append(f'<div class="ghv-meta">Branch: <b>{_esc(data.get("default_branch", "main"))}</b></div>')
    parts.append('<div style="margin-top:10px;">')
    for entry in data.get("entries", [])[:40]:
        icon = "📁" if entry["type"] == "dir" else "📄"
        parts.append(
            f'<div class="ghv-row"><span>{icon} <span class="ghv-name">{_esc(entry["name"])}</span></span>'
            f'<span class="ghv-meta">{_esc(entry["type"])}</span></div>'
        )
    parts.append("</div>")
    if data.get("readme"):
        parts.append('<div class="ghv-meta" style="margin-top:12px;">📘 README.md preview</div>')
        parts.append(f'<div class="ghv-readme">{_esc(data["readme"])}</div>')
    parts.append(_GHV_FOOTER)
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.link_button("↗ Open on github.com", data.get("html_url", "https://github.com"), use_container_width=True)


def render_settings_view(data: Dict[str, Any]) -> None:
    badges = ["🔒 Private" if data.get("private") else "🌐 Public"]
    parts = [_ghv_header(data["full_name"], badges, "Settings")]
    parts.append('<div class="ghv-meta" style="margin-bottom:6px;">General</div>')

    def toggle_row(label: str, value: bool) -> str:
        cls = "ghv-toggle-on" if value else "ghv-toggle-off"
        state = "● Enabled" if value else "○ Disabled"
        return f'<div class="ghv-setting-row"><span>{_esc(label)}</span><span class="{cls}">{state}</span></div>'

    parts.append(
        f'<div class="ghv-setting-row"><span>Repository name</span>'
        f'<span class="ghv-meta">{_esc(data["full_name"].split("/")[-1])}</span></div>'
    )
    parts.append(
        f'<div class="ghv-setting-row"><span>Description</span>'
        f'<span class="ghv-meta">{_esc(data.get("description") or "(none)")}</span></div>'
    )
    parts.append(
        f'<div class="ghv-setting-row"><span>Default branch</span>'
        f'<span class="ghv-meta">{_esc(data.get("default_branch", "main"))}</span></div>'
    )
    parts.append(
        f'<div class="ghv-setting-row"><span>Visibility</span>'
        f'<span class="ghv-meta">{"Private" if data.get("private") else "Public"}</span></div>'
    )
    parts.append('<div class="ghv-meta" style="margin:14px 0 6px;">Features</div>')
    parts.append(toggle_row("Issues", data.get("has_issues", False)))
    parts.append(toggle_row("Wiki", data.get("has_wiki", False)))
    parts.append(toggle_row("Projects", data.get("has_projects", False)))
    parts.append('<div class="ghv-meta" style="margin:14px 0 6px;">Pull Requests</div>')
    parts.append(toggle_row("Automatically delete head branches", data.get("delete_branch_on_merge", False)))
    if data.get("topics"):
        parts.append('<div class="ghv-meta" style="margin:14px 0 6px;">Topics</div>')
        parts.append(
            "".join(f'<span class="ghv-badge">{_esc(t)}</span>' for t in data["topics"])
        )
    parts.append(_GHV_FOOTER)
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.link_button("↗ Open Settings on github.com", data.get("settings_url", "https://github.com"), use_container_width=True)


def render_issues_view(data: Dict[str, Any]) -> None:
    parts = [_ghv_header(data["full_name"], [f"{len(data.get('issues', []))} open"], "Issues")]
    if not data.get("issues"):
        parts.append('<div class="ghv-meta">No open issues.</div>')
    for issue in data.get("issues", []):
        cls = "ghv-issue-closed" if issue["state"] == "closed" else "ghv-issue-open"
        icon = "🟣" if issue.get("is_pull_request") else "🟢"
        parts.append(
            f'<div class="ghv-row"><span>{icon} <span class="ghv-name">{_esc(issue["title"])}</span></span>'
            f'<span class="{cls}">#{issue["number"]} · {_esc(issue["state"])}</span></div>'
        )
    parts.append(_GHV_FOOTER)
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_releases_view(data: Dict[str, Any]) -> None:
    parts = [_ghv_header(data["full_name"], [f"{len(data.get('releases', []))} releases"], "Releases")]
    if not data.get("releases"):
        parts.append('<div class="ghv-meta">No releases yet.</div>')
    for release in data.get("releases", []):
        tag = "🏷️ " + _esc(release["tag_name"])
        badge = "Draft" if release.get("draft") else ("Pre-release" if release.get("prerelease") else "Latest")
        parts.append(
            f'<div class="ghv-row"><span class="ghv-name">{tag} {_esc(release.get("title", ""))}</span>'
            f'<span class="ghv-meta">{badge}</span></div>'
        )
    parts.append(_GHV_FOOTER)
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_workflows_view(data: Dict[str, Any]) -> None:
    parts = [_ghv_header(data["full_name"], [f"{len(data.get('workflows', []))} workflows"], "Actions")]
    if not data.get("workflows"):
        parts.append('<div class="ghv-meta">No GitHub Actions workflow found.</div>')
    for workflow in data.get("workflows", []):
        parts.append(
            f'<div class="ghv-row"><span class="ghv-name">⚙️ {_esc(workflow["name"])}</span>'
            f'<span class="ghv-meta">{_esc(workflow["state"])}</span></div>'
        )
    parts.append(_GHV_FOOTER)
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_file_view(data: Dict[str, Any]) -> None:
    parts = [_ghv_header(data["full_name"], [], "Code")]
    parts.append(f'<div class="ghv-meta" style="margin-bottom:8px;">📄 {_esc(data.get("path", ""))}</div>')
    content = data.get("content", "")
    if data.get("truncated"):
        content += "\n...(truncated)"
    parts.append(f'<div class="ghv-code">{_esc(content)}</div>')
    parts.append(_GHV_FOOTER)
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.link_button("↗ Open file on github.com", data.get("html_url", "https://github.com"), use_container_width=True)


def render_repos_list_view(data: Dict[str, Any]) -> None:
    parts = [
        '<div class="ghv-box"><div class="ghv-header">'
        f'<span class="ghv-title">👤 {_esc(data.get("owner", ""))} — Repositories</span></div>'
        '<div class="ghv-body">'
    ]
    if not data.get("repos"):
        parts.append('<div class="ghv-meta">No repositories found.</div>')
    for repo in data.get("repos", []):
        badge = "🔒 Private" if repo.get("private") else "🌐 Public"
        parts.append(
            f'<div class="ghv-row"><span class="ghv-name">📦 {_esc(repo["full_name"])}</span>'
            f'<span class="ghv-meta">{badge} · ⭐ {repo.get("stars", 0)}</span></div>'
        )
    parts.append(_GHV_FOOTER)
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_aws_ec2_view(data: Dict[str, Any]) -> None:
    url = data.get("url", "")
    parts = [
        '<div class="ghv-box"><div class="ghv-header"><span class="ghv-title">🚀 AWS EC2 Deployment</span><span><span class="ghv-badge">Docker</span><span class="ghv-badge">Live</span></span></div><div class="ghv-body">',
        f'<div class="ghv-row"><span>Repository</span><span class="ghv-name">{_esc(data.get("repository",""))}</span></div>',
        f'<div class="ghv-row"><span>EC2 Instance</span><span class="ghv-meta">{_esc(data.get("instance_id","unknown"))}</span></div>',
        f'<div class="ghv-row"><span>Host</span><span class="ghv-meta">{_esc(data.get("host","unknown"))}</span></div>',
        f'<div class="ghv-row"><span>Port</span><span class="ghv-meta">{_esc(data.get("port","unknown"))}</span></div>',
        f'<div class="ghv-row"><span>Container</span><span class="ghv-meta">{_esc(data.get("container","unknown"))}</span></div>',
        f'<div class="ghv-row"><span>Deployment Mode</span><span class="ghv-meta">{_esc(data.get("deployment_mode","AWS SSM direct"))}</span></div>',
        f'<div class="ghv-row"><span>Security Group</span><span class="ghv-meta">{_esc(data.get("security_group","unknown"))}</span></div>',
        f'<div style="margin-top:14px;padding:14px;border:1px solid #30363d;border-radius:8px;background:#161b22;"><div class="ghv-meta">LIVE APPLICATION</div><div style="font-size:18px;font-weight:700;margin-top:4px;">{_esc(url)}</div></div>',
        _GHV_FOOTER]
    st.markdown("".join(parts), unsafe_allow_html=True)
    if url:
        st.link_button("🌐 Open Live App", url, use_container_width=True)
    if data.get("workflow_url"):
        st.link_button("⚙️ Open GitHub Actions Run", data["workflow_url"], use_container_width=True)

def render_live_github_view() -> None:
    st.markdown(GITHUB_VIEW_CSS, unsafe_allow_html=True)
    live = st.session_state.live_view
    if not live:
        st.markdown(
            '<div class="ghv-empty">Jab agent koi GitHub action karega — repo kholna, '
            'settings, issues, releases, workflows — yahan bilkul GitHub jaisi screen '
            'live update hoti rahegi.</div>',
            unsafe_allow_html=True,
        )
        return
    kind = live.get("kind")
    data = live.get("data", {})
    renderers = {
        "repo": render_repo_view,
        "settings": render_settings_view,
        "issues": render_issues_view,
        "releases": render_releases_view,
        "workflows": render_workflows_view,
        "file": render_file_view,
        "repos_list": render_repos_list_view,
        "aws_ec2": render_aws_ec2_view,
    }
    renderer = renderers.get(kind)
    if renderer:
        renderer(data)
    else:
        st.json(data)


GITHUB_OAUTH_SCOPE = "repo delete_repo workflow"

# OAuth redirects can establish a fresh Streamlit browser session, AND the
# Streamlit process itself can restart between "Authorize with GitHub" and
# GitHub's redirect back (file-watcher auto-reload, Replit rebuild, etc).
# A plain in-memory dict - even one wrapped in st.cache_resource - does not
# survive a process restart, which is exactly what produces "OAuth session
# expire ho gaya ya state match nahi hua" even right after a fresh click.
# So the short-lived OAuth transaction is persisted to a small file on disk
# instead. The Client Secret never enters the URL, and each entry is
# single-use with a short TTL.
_OAUTH_PENDING_TTL_SECONDS = 10 * 60
_OAUTH_PENDING_PATH = Path(tempfile.gettempdir()) / "github_agent_oauth_pending.json"


def _load_oauth_pending() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_OAUTH_PENDING_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_oauth_pending(store: Dict[str, Dict[str, Any]]) -> None:
    try:
        _OAUTH_PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _OAUTH_PENDING_PATH.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(store, fh)
        os.replace(tmp_path, _OAUTH_PENDING_PATH)
    except OSError:
        pass  # best-effort persistence; falls back to same-session check


def _cleanup_oauth_pending(store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    now = time.time()
    return {
        state: item
        for state, item in store.items()
        if now - float(item.get("created_at", 0)) <= _OAUTH_PENDING_TTL_SECONDS
    }


def _remember_oauth_request(state: str, client_id: str, client_secret: str, redirect_uri: str) -> None:
    store = _cleanup_oauth_pending(_load_oauth_pending())
    store[state] = {
        "created_at": time.time(),
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }
    _save_oauth_pending(store)


def _take_oauth_request(state: str) -> Optional[Dict[str, Any]]:
    store = _cleanup_oauth_pending(_load_oauth_pending())
    item = store.pop(state, None)
    _save_oauth_pending(store)
    if not item:
        return None
    if time.time() - float(item.get("created_at", 0)) > _OAUTH_PENDING_TTL_SECONDS:
        return None
    return item


def github_oauth_authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
    return (
        "https://github.com/login/oauth/authorize"
        f"?client_id={quote(client_id)}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
        f"&scope={quote(GITHUB_OAUTH_SCOPE)}"
        f"&state={quote(state)}"
    )


def exchange_github_oauth_code(client_id: str, client_secret: str, code: str, redirect_uri: str) -> str:
    """Trade the callback ?code=... for a real access token, exactly the way
    github.com does it after the person clicks 'Authorize'."""
    response = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=20,
    )
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(f"GitHub se unexpected response mila (HTTP {response.status_code}).")
    if "error" in payload:
        raise RuntimeError(payload.get("error_description") or payload.get("error"))
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("GitHub ne access token nahi diya. OAuth App ki settings check karo.")
    return str(token)


def connect_github_with_token(token: str) -> None:
    client = Github(token.strip())
    user = client.get_user()
    _ = user.login  # force a call so a bad token fails immediately
    st.session_state.gh_client = client
    st.session_state.gh_user = user


def handle_github_oauth_callback() -> None:
    """Run once per rerun, before the sidebar draws, so an OAuth redirect
    back into the app connects automatically - same effect as the
    'Authorize' screen the person just approved on github.com."""
    params = st.query_params
    code = params.get("code")
    returned_state = params.get("state")
    oauth_error = params.get("error")
    if not code and not oauth_error:
        return

    # GitHub may redirect to a fresh Streamlit session. Therefore do not rely
    # on session_state for the OAuth handshake credentials. Look them up by
    # the one-time random state created before leaving this app.
    request_data = _take_oauth_request(str(returned_state)) if returned_state else None

    # Same-session fallback is useful when the Streamlit websocket survives.
    if request_data is None and returned_state == st.session_state.github_oauth_state:
        client_id = st.session_state.github_oauth_client_id.strip()
        client_secret = st.session_state.github_oauth_client_secret.strip()
        redirect_uri = st.session_state.github_oauth_redirect_uri.strip()
        if client_id and client_secret and redirect_uri:
            request_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
            }

    if not returned_state or request_data is None:
        st.session_state.github_oauth_error = (
            "OAuth session expire ho gaya ya state match nahi hua. "
            "Sidebar mein Client ID/Secret check karke dobara Authorize with GitHub dabao."
        )
        st.query_params.clear()
        return

    if oauth_error:
        description = params.get("error_description") or oauth_error
        st.session_state.github_oauth_error = f"GitHub authorization cancel/fail hua: {description}"
        st.query_params.clear()
        return

    client_id = str(request_data["client_id"]).strip()
    client_secret = str(request_data["client_secret"]).strip()
    redirect_uri = str(request_data["redirect_uri"]).strip()
    st.session_state.github_oauth_client_id = client_id
    st.session_state.github_oauth_client_secret = client_secret
    st.session_state.github_oauth_redirect_uri = redirect_uri

    try:
        token = exchange_github_oauth_code(client_id, client_secret, code, redirect_uri)
        connect_github_with_token(token)
        st.session_state.github_oauth_error = None
    except Exception as exc:
        st.session_state.gh_client = None
        st.session_state.gh_user = None
        st.session_state.github_oauth_error = f"GitHub authorize hua, lekin connect nahi ho paaya: {github_error(exc)}"
    st.query_params.clear()


init_state()
handle_github_oauth_callback()

with st.sidebar:
    st.header("🔑 GitHub Connection")

    if st.session_state.github_oauth_error:
        st.error(st.session_state.github_oauth_error)

    pat_tab, oauth_tab = st.tabs(["🔤 Token", "🔓 Authorize with GitHub"])

    with pat_tab:
        token = st.text_input(
            "GitHub Personal Access Token",
            type="password",
            key="github_token_input",
            help="Ye GitHub token hai, AI provider API key nahi. Action-specific permissions zaroori hain.",
        )
        connect_col, disconnect_col = st.columns(2)
        with connect_col:
            if st.button("Connect", use_container_width=True):
                if not token.strip():
                    st.error("Pehle GitHub token daalo.")
                else:
                    try:
                        connect_github_with_token(token)
                        st.success(f"Connected as {st.session_state.gh_user.login}")
                    except Exception as exc:
                        st.session_state.gh_client = None
                        st.session_state.gh_user = None
                        st.error(f"GitHub connection fail hui: {github_error(exc)}")
        with disconnect_col:
            if st.button("Disconnect", use_container_width=True):
                st.session_state.gh_client = None
                st.session_state.gh_user = None
                st.info("GitHub disconnect ho gaya.")

    with oauth_tab:
        st.caption(
            "Yahan se ChatGPT jaisa 'Authorize application' screen aayega - "
            "token copy-paste karne ki zaroorat nahi."
        )
        st.text_input("OAuth Client ID", key="github_oauth_client_id")
        st.text_input("OAuth Client Secret", type="password", key="github_oauth_client_secret")
        st.text_input(
            "Authorization callback URL",
            key="github_oauth_redirect_uri",
            help="Yahi URL GitHub OAuth App ki 'Authorization callback URL' field mein bhi exactly daalo.",
        )
        with st.expander("Pehli baar setup kaise karein"):
            st.markdown(
                "1. [github.com/settings/developers](https://github.com/settings/developers) "
                "kholo -> **OAuth Apps** -> **New OAuth App**.\n"
                "2. **Homepage URL** aur **Authorization callback URL** dono mein "
                "wahi URL do jahan ye app chal rahi hai (local par "
                "`http://localhost:8501`).\n"
                "3. App create hone ke baad **Client ID** milega, aur **Generate a "
                "new client secret** se **Client Secret** milega.\n"
                "4. Dono values upar wale fields mein paste karo, phir "
                "**Authorize with GitHub** button dabao - GitHub ka apna consent "
                "screen khulega (bilkul screenshot jaisa), Authorize dabate hi ye "
                "app automatically connect ho jayegi.\n\n"
                "Client Secret sirf is browser session mein rehta hai, kahin save "
                "nahi hota."
            )
        client_id = st.session_state.github_oauth_client_id.strip()
        redirect_uri = st.session_state.github_oauth_redirect_uri.strip()
        if client_id and redirect_uri:
            oauth_state = st.session_state.github_oauth_state
            _remember_oauth_request(
                oauth_state,
                client_id,
                st.session_state.github_oauth_client_secret.strip(),
                redirect_uri,
            )
            authorize_url = github_oauth_authorize_url(
                client_id, redirect_uri, oauth_state
            )
            st.link_button("🔓 Authorize with GitHub", authorize_url, use_container_width=True)
        else:
            st.info("Authorize button ke liye pehle Client ID aur callback URL bharo.")
        if st.session_state.gh_user and st.button("Disconnect", key="oauth_disconnect", use_container_width=True):
            st.session_state.gh_client = None
            st.session_state.gh_user = None
            st.info("GitHub disconnect ho gaya.")

    if st.session_state.gh_user:
        st.success(f"Connected: {st.session_state.gh_user.login}")
        st.caption(
            "Connected ka matlab token valid hai; repo create/delete ya workflow "
            "jaise actions ke liye alag permission phir bhi required ho sakti hai."
        )
    else:
        st.warning("GitHub connected nahi hai")

    with st.expander("Create/delete permission fix"):
        st.markdown(
            "Agar `Resource not accessible by personal access token` aaye, "
            "GitHub token mein action ki permission missing hai.\n\n"
            "- Classic PAT: private/repository actions ke liye `repo`; repository "
            "delete ke liye `delete_repo`.\n"
            "- Fine-grained PAT: correct account/organization ko resource owner "
            "banao aur Administration, Contents, Issues, Pull requests ko "
            "Read and write access do. Workflow run ke liye Actions ko bhi "
            "Read and write access chahiye.\n"
            "- Organization token policy repo creation/deletion ko block kar sakti hai.\n\n"
            "[GitHub token settings kholo](https://github.com/settings/personal-access-tokens/new)"
        )

    st.divider()
    st.header("🧠 AI API Key")
    st.caption("Provider choose karo. Key sirf is browser session mein rahegi.")
    st.selectbox(
        "AI provider",
        list(PROVIDERS.keys()),
        key="provider",
        on_change=sync_provider_model,
    )
    provider_info = PROVIDERS[st.session_state.provider]
    st.text_input(
        "Model name (not API key)",
        key="model",
        help=f"Provider default: {provider_info['model']}. Provider change hone par ye automatically update hota hai.",
    )
    if provider_info["kind"] == "custom":
        st.text_input(
            "Custom OpenAI-compatible endpoint",
            key="custom_endpoint",
            placeholder="https://your-host/v1/chat/completions",
        )
    st.text_input(
        f"{st.session_state.provider} API key",
        type="password",
        key="api_key",
        help=provider_info["help"],
    )
    if st.session_state.api_key:
        st.success("AI API key set hai")
    else:
        st.info("API key add karne ke baad free-language chat chalegi")

    if st.button("🔄 Auto-detect working model", use_container_width=True):
        if not st.session_state.api_key.strip():
            st.error("Pehle API key daalo.")
        else:
            detect_endpoint = (
                st.session_state.custom_endpoint.strip()
                if provider_info["kind"] == "custom"
                else provider_info.get("endpoint", "")
            )
            if provider_info["kind"] == "custom" and not detect_endpoint:
                st.error("Pehle custom endpoint daalo.")
            else:
                with st.spinner("Available models check kar raha hoon..."):
                    available = fetch_available_models(
                        provider_info, st.session_state.api_key.strip(), detect_endpoint
                    )
                picked = pick_fallback_model(available, set())
                if picked:
                    st.session_state.model = picked
                    st.success(f"Model set kiya: `{picked}`")
                else:
                    st.error(
                        "Koi model list nahi mil paayi. Key/endpoint check karo, "
                        "ya current model field mein khud naam daalo."
                    )

    st.divider()
    st.header("📁 Project Upload")
    st.caption("Koi bhi file select karo; ZIP files automatically extract hongi.")
    uploaded_files = st.file_uploader(
        "Choose project files or ZIP",
        type=None,
        accept_multiple_files=True,
        help="Individual files, multiple files, ya ZIP upload kar sakte ho.",
    )
    if uploaded_files:
        upload_signature = hashlib.sha256(
            b"".join(uploaded_file.getvalue() for uploaded_file in uploaded_files)
        ).hexdigest()
        if upload_signature != st.session_state.uploaded_signature:
            load_uploaded_files(uploaded_files)
    if st.session_state.project_files:
        st.success(f"{len(st.session_state.project_files)} files ready")
        st.markdown(project_analysis_text(st.session_state.project_analysis))
        with st.expander("File list dekho"):
            st.text("\n".join(project_file["path"] for project_file in st.session_state.project_files))
        risky_files = [
            project_file["path"]
            for project_file in st.session_state.project_files
            if project_file["path"].lower().endswith(
                (".env", ".pem", ".key", ".p12", ".pfx")
            )
        ]
        if risky_files:
            st.warning(
                "Sensitive-looking files bhi loaded hain. Push se pehle review "
                "karo: " + ", ".join(risky_files)
            )
        if st.session_state.last_push_report:
            report = st.session_state.last_push_report
            with st.expander("Last GitHub changes — all files"):
                st.markdown(f"**Repository:** `{report['repo']}`")
                for label, key, icon in (
                    ("Created", "created", "✅"),
                    ("Updated", "updated", "🔄"),
                    ("Failed", "failed", "❌"),
                ):
                    paths = report.get(key, [])
                    st.markdown(f"**{label} ({len(paths)})**")
                    if paths:
                        st.text("\n".join(f"{icon} {path}" for path in paths))
                    else:
                        st.caption("None")
        if st.button("Clear uploaded project", use_container_width=True):
            st.session_state.project_files = []
            st.session_state.project_zip_name = None
            st.session_state.uploaded_signature = None
            st.session_state.project_analysis = None
            st.session_state.last_push_report = None
            st.session_state.pending_push = False
            st.session_state.pending_deployment = None
            st.rerun()

    st.divider()
    st.header("🚀 Deployment Control Center")
    st.caption(
        "Agent project analyze karega, platform puchega, deployment se pehle approval lega, "
        "AWS SSM se EC2 par Docker deploy karega aur actual live URL + host + port return karega."
    )
    st.selectbox(
        "Select Deployment Platform",
        [ASK_DEPLOYMENT_PLATFORM, *DEPLOYMENT_PLATFORM_OPTIONS],
        key="deployment_platform",
    )
    if st.session_state.deployment_platform == "AWS EC2":
        st.text_input("AWS Region", key="aws_region", help="Example: us-east-1")
        st.text_input("EC2 Instance ID (optional)", key="aws_ec2_instance_id", placeholder="i-0123456789abcdef0", help="Blank chhodo to running instance with Name tag github_App auto-detect hoga.")
        st.text_input("Security Group ID (optional)", key="aws_security_group_id", placeholder="sg-0123456789abcdef0", help="Blank chhodo to instance ka first security group auto-detect hoga.")
        st.text_input(
            "AWS Access Key ID",
            type="password",
            key="aws_access_key_id",
            help=(
                "Agent apne local computer par chal raha hai to yahan zaroori hai. "
                "Agar Agent khud kisi AWS server (EC2/App Runner) par IAM role ke sath chal raha hai, tabhi blank chhodo. "
                "Key sirf is browser session mein rahegi, kahin save/log nahi hoti."
            ),
        )
        st.text_input(
            "AWS Secret Access Key",
            type="password",
            key="aws_secret_access_key",
            help=(
                "Agent apne local computer par chal raha hai to yahan zaroori hai. "
                "Agar Agent khud kisi AWS server (EC2/App Runner) par IAM role ke sath chal raha hai, tabhi blank chhodo. "
                "Secret sirf is browser session mein rahega, kahin save/log nahi hota."
            ),
        )
        st.text_input("AWS Session Token (optional)", type="password", key="aws_session_token")
        if st.button("🔎 Test AWS / EC2 / SSM", use_container_width=True):
            status = aws_connection_status()
            if status.get("ready"):
                st.success(
                    f"AWS ready · EC2 {status.get('instance_id')} · SSM online · account {status.get('account')}"
                )
            else:
                st.error(f"AWS readiness check failed: {status.get('message')}")
        with st.expander("🛠️ One-time AWS setup (recommended)"):
            st.markdown(
                "**Agent ko AWS permissions chahiye:** `ec2:DescribeInstances`, `ec2:DescribeSecurityGroups`, `ec2:AuthorizeSecurityGroupIngress`, `ssm:SendCommand`, `ssm:GetCommandInvocation`.\n\n"
                "**Target EC2:** SSM Agent installed/running hona chahiye aur EC2 instance profile mein `AmazonSSMManagedInstanceCore` hona chahiye.\n\n"
                "**Best practice:** AWS keys UI mein na daalo; agent host par IAM role/environment credentials use karo.\n\n"
                "AWS mode GitHub Actions ko trigger nahi karta. Approval ke baad Agent AWS SSM se EC2 par repo clone/pull → Docker build → free port selection → security-group rule → container run → real URL/host/port automatically return karega."
            )
    if st.session_state.project_analysis:
        recommended = st.session_state.project_analysis.get(
            "recommended_platform", "Render"
        )
        st.info(f"File analysis recommendation: **{recommended}**")

    st.divider()
    st.header("🚀 Publish settings")
    st.checkbox("Keep repository private", key="private_repo")
    st.text_input("Commit message", key="commit_message")

st.caption(
    "English, Hindi, Hinglish ya kisi bhi language mein naturally chat karo. "
    "Project upload karke GitHub par publish karne ko bhi bol sakte ho."
)

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.metric("GitHub", "Connected" if st.session_state.gh_user else "Not connected")
with status_col2:
    st.metric("AI", st.session_state.provider if st.session_state.api_key else "API key needed")
with status_col3:
    st.metric("Project", f"{len(st.session_state.project_files)} files" if st.session_state.project_files else "Not uploaded")

chat_col, action_col = st.columns([1, 1], gap="large")

with chat_col:
    st.subheader("💬 Agent Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    user_text = st.chat_input("English / Hindi / Hinglish command...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            with st.spinner("Request analyze ho rahi hai..."):
                answer = handle_message(user_text)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

with action_col:
    st.subheader("⚡ Live Action Workspace")
    ghv_tab, timeline_tab = st.tabs(["🖥️ Live GitHub View", "⚡ Action Timeline"])

    with ghv_tab:
        st.caption(
            "Agent GitHub par jo bhi action karta hai — repo open, settings, "
            "issues, releases, workflows — yahan real GitHub jaisi screen "
            "turant update hoti hai."
        )
        render_live_github_view()

    with timeline_tab:
        pending = st.session_state.pending_confirmation
        if pending:
            st.warning("User approval required before execution.")
            st.markdown(approval_description(pending))
            st.caption("Current token status: GitHub will enforce the actual permission at execution time.")
            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Approve", key="live_approve", use_container_width=True):
                st.session_state.pending_confirmation = None
                emit_action_event("approval_received", "approved", "Approval received",
                                  "User approved the action.", pending)
                with st.spinner("Executing approved action..."):
                    result = execute_action(pending)
                st.session_state.action_history.append({
                    "action_id": st.session_state.active_action_id,
                    "type": pending.get("type"),
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                st.session_state.messages.append({"role":"assistant","content":str(result)})
                st.rerun()
            if b2.button("⛔ Deny", key="live_deny", use_container_width=True):
                st.session_state.pending_confirmation = None
                emit_action_event("approval_denied", "denied", "Approval denied",
                                  "User denied the action.", pending)
                st.session_state.action_history.append({
                    "action_id": st.session_state.active_action_id,
                    "type": pending.get("type"),
                    "status": "denied",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                st.rerun()
            if b3.button("✖ Cancel Action", key="live_cancel", use_container_width=True):
                st.session_state.pending_confirmation = None
                emit_action_event("action_cancelled", "cancelled", "Action cancelled",
                                  "User cancelled the action.", pending)
                st.rerun()
        else:
            st.info("No action is waiting for approval.")

        events = st.session_state.action_events[-30:]
        if events:
            completed = sum(1 for e in events if e["status"] == "completed")
            st.progress(min(1.0, completed / max(1, len(events))))
            st.markdown("**Step timeline**")
            for e in reversed(events):
                icon = {"completed":"🟢","running":"🔵","waiting":"🟡",
                        "failed":"🔴","denied":"⛔","cancelled":"⚪"}.get(e["status"],"•")
                st.markdown(f"{icon} **{e['title']}** — {e['message']}")
                if e.get("data"):
                    with st.expander("Event details", expanded=False):
                        st.json(e["data"])

        if st.session_state.action_history:
            with st.expander("Action history", expanded=False):
                st.json(st.session_state.action_history[-20:])
        if st.button("🔄 Refresh / Verify", use_container_width=True):
            st.rerun()
