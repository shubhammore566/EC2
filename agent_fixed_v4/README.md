# GitHub Agent — production-oriented Streamlit application

This project upgrades the original Streamlit GitHub DevOps Agent with an explicit approval gate, live action workspace, typed event records, security helpers, audit history, project ZIP analysis, and the existing GitHub/AI/deployment workflows.

## Main workflow

1. Connect GitHub either way:
   - **Token tab:** paste a GitHub Personal Access Token, or
   - **Authorize with GitHub tab:** click **Authorize with GitHub** and approve
     on GitHub's own consent screen (the same "Authorize application" screen
     GitHub shows for any OAuth app) - no token copy-pasting needed. See
     "OAuth login setup" below for the one-time GitHub OAuth App you create.
2. Enter a natural-language GitHub request in English, Hindi or Hinglish.
3. The agent plans the action and maps it to a GitHub permission.
4. Read-only actions may run directly.
5. Write, destructive and sensitive actions create an approval card in the **Live Action Workspace**.
6. Only an explicit **Approve** executes the pending action.
7. The right panel records approval, execution and result events.
8. GitHub remains the final authority for the actual token permission.
9. Errors are surfaced without exposing the token.

## Security

- GitHub token and AI key are kept only in Streamlit session state.
- No credential is written to `.env`, source, audit history, or logs.
- Sensitive-looking project files are detected before publishing.
- Repository paths are checked for traversal.
- Destructive operations use a high-risk approval classification.
- No permission upgrade or bypass is attempted.
- Never paste real secrets into chat or source files.

> Important: Streamlit session state is not a durable encrypted production credential vault. For a multi-user production deployment, put OAuth/GitHub App credentials and encrypted server-side sessions behind a proper authentication/session service.

## OAuth login setup

The "Authorize with GitHub" tab uses a standard GitHub OAuth App, so people
connect the same way they would for any other GitHub-integrated app:

1. Go to <https://github.com/settings/developers> -> **OAuth Apps** ->
   **New OAuth App**.
2. Set **Homepage URL** and **Authorization callback URL** to the URL this
   app runs at (e.g. `http://localhost:8501` for local `streamlit run`, or
   the deployed app's URL).
3. Copy the generated **Client ID**, and click **Generate a new client
   secret** for the **Client Secret**.
4. Paste both into the sidebar's "Authorize with GitHub" tab, matching the
   callback URL exactly.
5. Click **Authorize with GitHub**. GitHub shows its own consent screen;
   approving it redirects back into the app with a one-time code, which the
   app immediately exchanges for an access token and connects - the Client
   Secret and the resulting token both stay in the Streamlit session only.

## Live GitHub View

The right-hand panel has two tabs: **Live GitHub View** and **Action Timeline**.

Every time the agent reads or changes something on GitHub — opening a
repository, listing issues/releases/workflows, reading a file, or opening a
repository's Settings — the **Live GitHub View** tab re-renders a
GitHub-styled screen (dark theme, tabs, badges, README preview, settings
toggles) built from the real data the GitHub API just returned, so the
person watching the chat can see "what happened on GitHub" without leaving
the app.

Note: github.com itself blocks being embedded in another site's `<iframe>`
(it sends `X-Frame-Options: deny`), so this panel is a live, data-driven
recreation of the relevant GitHub screen rather than a literal embedded
browser tab. A "🔗 Open on github.com" button is included so the person can
jump to the real page in one click.

You can ask for a repository's settings directly, e.g. `cancer repo settings
kholo` / `open settings for owner/repo`, and change them the same way, e.g.
`cancer ko private kar do`, `issues off kar do`, `description change karke
"New description"`. Settings changes go through the same approval gate as
every other write action.

## Live events

The app records typed events including:
`approval_requested`, `approval_received`, `approval_denied`, `step_started`,
`api_response`, `step_completed`, `action_failed`, and `action_cancelled`.

The UI refreshes without losing the current session state. A true cross-process WebSocket/SSE broker is not included because the existing application is Streamlit; for horizontally scaled production deployments, replace the in-process event list with Redis/pub-sub plus SSE/WebSocket.

## Supported GitHub capability map

The original application includes common repository, file, branch, issue, pull-request, release, workflow and deployment actions. `github_agent/tool_manifest.json` contains the complete typed capability contract requested for the broader GitHub tool surface, including metadata, contents, issues, pull requests, Actions, workflows, secrets/variables, environments, collaborators, webhooks, repository settings, topics, visibility and a generic REST operation.

GitHub API support varies by token type and organization policy. The app does not assume that a valid token has every permission.

## Setup

### Windows

Double-click `run_app.bat`, or:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

### Environment

Copy `.env.example` for deployment documentation. The current UI intentionally accepts credentials through password inputs rather than reading or persisting a local `.env`.

Required production concepts:
- `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` if OAuth is added
- `GITHUB_APP_ID` / `GITHUB_PRIVATE_KEY` if a GitHub App is used
- `AI_PROVIDER` / `AI_API_KEY`
- `DATABASE_URL`
- `SESSION_SECRET`
- `ENCRYPTION_KEY`
- `PUBLIC_APP_URL`

Do not put real values in Git.

## Testing

Run:

```bash
pytest -q
python -m py_compile app.py github_agent/*.py
```

The included tests cover ZIP analysis, path safety, sensitive-file detection, permission approval classification and destructive-action risk.

## Deployment

The original project keeps the existing deployment preparation flows for Streamlit Cloud, AWS, Azure, Google Cloud and Render. Streamlit Cloud still requires the user to sign in and complete the Create app step.

### AWS deployment (two supported paths)

This repo now includes a `Dockerfile`, `.dockerignore`, `apprunner.yaml`, and
`.github/workflows/deploy-aws.yml` — none of these existed before, and AWS
cannot deploy the app without at least one of them. Pick **one** path:

**Path A — App Runner connected directly to GitHub (no Docker/ECR, simplest)**
1. In the AWS Console: App Runner → Create service → Source: **Source code repository** → connect this GitHub repo/branch (creates a CodeStar connection).
2. App Runner auto-detects `apprunner.yaml` in the repo root and uses it to build/run the app.
3. Every push to the connected branch redeploys automatically — no GitHub secrets needed for this path.

**Path B — GitHub Actions → ECR → App Runner (CI/CD, matches this app's "deploy_aws" chat command)**
1. Create an ECR repository and an App Runner service that pulls from that ECR image (first image can be a placeholder).
2. Create an IAM user with `AmazonEC2ContainerRegistryFullAccess` and `AWSAppRunnerFullAccess` (or a tighter custom policy), and generate an access key.
3. Add these **GitHub repo secrets** (Settings → Secrets and variables → Actions):
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (e.g. `ap-south-1`)
   - `ECR_REPOSITORY` (repo name only, not the full URI)
   - `APPRUNNER_SERVICE_ARN`
4. Push to `main`, or in this app say "AWS pe deploy karo" / trigger the `deploy-aws.yml` workflow manually — it builds the Docker image, pushes to ECR, and redeploys the App Runner service. This is exactly the workflow the app's `deploy_aws` chat action looks for and dispatches.

Prefer ECS Fargate or Elastic Beanstalk instead? The same `Dockerfile` works there too; only the last two steps of `deploy-aws.yml` (App Runner-specific) need swapping for the equivalent ECS/EB deploy action.

## Known limitations

- The current app is Streamlit, so its event bus is session-local rather than a distributed WebSocket/SSE service.
- GitHub token permission introspection is limited by GitHub API/token behavior; actual API calls remain authoritative.
- GitHub secret values are never readable through the normal repository secret API; only secret metadata is exposed.
- Not every item in the broad typed manifest has a UI command handler yet; the existing app's concrete handlers remain the executable surface.

### OAuth redirect/session fix

The OAuth callback stores the short-lived Client ID/Client Secret/redirect URI in a Streamlit process-scoped resource cache under the one-time OAuth `state` value. This is needed because a GitHub redirect can create a fresh Streamlit browser session, while a normal module-level dictionary is reset when the Streamlit script reruns. Secrets are never placed in the OAuth URL and pending entries expire after 10 minutes and are single-use.


## Automatic AWS EC2 deployment (direct SSM)

The Agent can now deploy a GitHub repository directly to a target EC2 instance without requiring a self-hosted GitHub Actions runner. After the user explicitly approves the deployment, it: selects the target EC2 instance, clones/pulls a public repository, creates a Streamlit Dockerfile if needed, builds the Docker image, selects a free 8501-8599 host port, runs the container, adds the port to the instance security group when the AWS permission is available, and returns the real public host/IP + port + live URL.

### One-time AWS setup

1. Target EC2 must be running and have SSM Agent connected to Systems Manager. Attach the AWS managed policy `AmazonSSMManagedInstanceCore` to the EC2 instance role.
2. Give the Agent host (Render/server/local AWS profile) the permissions in `aws/agent-deployer-policy.json`. Prefer an IAM role or environment credentials over browser-entered keys.
3. In the Agent sidebar, select **AWS EC2**, set the region (for example `us-east-1`), and optionally set the exact EC2 Instance ID. If blank, the Agent auto-detects a single running instance with Name tag `github_App`.
4. The target repository should be public for the current safe clone path. Private repository deployment is intentionally blocked until a short-lived GitHub App token flow is configured; a PAT is never placed in an SSM shell command.
5. The Agent can automatically add the selected TCP port to the target instance security group. If you do not grant `ec2:AuthorizeSecurityGroupIngress`, add the port manually instead.

### Example environment

```text
AWS_REGION=us-east-1
EC2_INSTANCE_ID=i-xxxxxxxxxxxxxxxxx
EC2_SECURITY_GROUP_ID=sg-xxxxxxxxxxxxxxxxx
```

If the Agent runs on an AWS host with an IAM role, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` are not required.
