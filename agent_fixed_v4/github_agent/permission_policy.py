
"""Central permission policy used by the UI and future tool backends."""
READ_ONLY_TOOLS = {
    "list_authenticated_user","get_current_user","list_repositories","get_repository",
    "list_branches","get_branch","list_commits","get_commit","get_file","list_directory",
    "show_file_diff","list_pull_requests","get_pull_request","list_issues","get_issue",
    "list_releases","get_release","list_workflows","get_workflow_details","get_workflow_runs",
    "get_workflow_logs","get_job_logs","list_repository_secrets","list_repository_variables",
    "list_environments","list_collaborators","list_webhooks",
}
HIGH_RISK_TOOLS = {
    "delete_repository","delete_branch","force_push","merge_pull_request",
    "delete_file","delete_release","delete_tag","secrets_write","collaborator_write",
    "webhook_write","settings_write","workflow_file_write","environment_write",
}
def requires_approval(tool: str) -> bool:
    return tool not in READ_ONLY_TOOLS
def risk_for(tool: str) -> str:
    if tool in HIGH_RISK_TOOLS: return "high"
    if tool in READ_ONLY_TOOLS: return "low"
    return "medium"
