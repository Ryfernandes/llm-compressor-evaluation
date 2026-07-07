from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["requests"]
)
def upload_to_github(
    session_id: str,
    save_path: str = "/tier2/evaluations",
    owner: str = "Ryfernandes",
    repo: str = "llm-evaluation-pipeline-results",
    base_branch: str = "main",
    repo_results_dir: str = "results"
) -> str:
    """Upload collated results JSON to GitHub as a PR."""
    import base64
    import json
    import os
    import time
    import uuid
    from pathlib import Path
    from urllib.parse import quote
    import requests

    # Read GitHub token from environment variable
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set")

    def github_request(method, url, token, json_data=None):
        """Helper function for GitHub API requests."""
        resp = requests.request(
            method,
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=json_data,
            timeout=30,
        )

        if not resp.ok:
            raise RuntimeError(
                f"GitHub API request failed: {method} {url}\n"
                f"Status: {resp.status_code}\n"
                f"Response: {resp.text}"
            )

        if resp.status_code == 204:
            return {}

        return resp.json()

    GITHUB_API = "https://api.github.com"

    # Find the collated results file
    collated_path = Path(save_path) / "sessions" / session_id / "collated" / "collated_results.json"

    if not collated_path.is_file():
        raise FileNotFoundError(f"Collated results file not found: {collated_path}")

    # Load and parse collated results
    with open(collated_path, "r") as f:
        collated_data = json.load(f)

    metadata = collated_data.get("metadata", {})
    model_id = metadata.get("model_id", "unknown-model")

    # Create a timestamped filename for the upload
    filename = f"collated-{session_id}.json"

    # GitHub repository paths should not start with "/"
    repo_results_dir = repo_results_dir.strip("/")
    repo_file_path = f"{repo_results_dir}/{filename}"

    print(f"Uploading {collated_path} to {owner}/{repo}:{repo_file_path}")

    # 1. Get the current commit SHA of base branch
    ref_path = quote(f"heads/{base_branch}", safe="/")
    base_ref = github_request(
        "GET",
        f"{GITHUB_API}/repos/{owner}/{repo}/git/ref/{ref_path}",
        github_token,
    )
    base_sha = base_ref["object"]["sha"]

    # 2. Create a unique branch from base
    unique_suffix = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    branch_name = f"add-result-{session_id}-{unique_suffix}"

    github_request(
        "POST",
        f"{GITHUB_API}/repos/{owner}/{repo}/git/refs",
        github_token,
        json_data={
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha,
        },
    )

    # 3. Commit the file to the new branch
    file_bytes = collated_path.read_bytes()
    encoded_content = base64.b64encode(file_bytes).decode("ascii")

    commit_message = f"Add evaluation results for session {session_id}"

    contents_path = quote(repo_file_path, safe="/")
    github_request(
        "PUT",
        f"{GITHUB_API}/repos/{owner}/{repo}/contents/{contents_path}",
        github_token,
        json_data={
            "message": commit_message,
            "content": encoded_content,
            "branch": branch_name,
        },
    )

    # 4. Format PR body with aggregate stats
    def escape_markdown_table_cell(text):
        """Escape special characters in markdown table cells."""
        if isinstance(text, str):
            return text.replace("|", "\\|")
        return text

    def format_server_info(server_info):
        """Format server information as a markdown section."""
        lines = ["### Server Configuration\n"]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        if "vllm_version" in server_info:
            lines.append(f"| vLLM Version | {server_info['vllm_version']} |")
        if "model_size" in server_info:
            lines.append(f"| Model Size (GB) | {server_info['model_size']:.2f} |")
        if "model_load_time" in server_info:
            lines.append(f"| Model Load Time (s) | {server_info['model_load_time']:.2f} |")
        if "kv_cache_size" in server_info:
            lines.append(f"| KV Cache Size (GB) | {server_info['kv_cache_size']:.2f} |")
        if "kv_cache_tokens" in server_info:
            lines.append(f"| KV Cache Tokens | {server_info['kv_cache_tokens']:,} |")
        if "recommended_concurrency" in server_info:
            lines.append(f"| Recommended Concurrency | {server_info['recommended_concurrency']:.2f} |")
        if "max_model_len" in server_info:
            lines.append(f"| Max Model Length | {server_info['max_model_len']:,} |")

        return "\n".join(lines)

    def format_results_table(results):
        """Format results as a markdown table."""
        if not results:
            return "No results available.\n"

        lines = ["| Task | Metric | Mean | Std Dev | Repetitions |"]
        lines.append("|------|--------|------|---------|-------------|")

        for result in sorted(results, key=lambda x: x["task_name"]):
            task_name = escape_markdown_table_cell(result["task_name"])
            aggregate_stats = result.get("aggregate_stats", {})
            metrics = aggregate_stats.get("metrics", {})
            num_reps = result.get("num_repetitions", 0)

            for metric_name, metric_data in metrics.items():
                metric_name_escaped = escape_markdown_table_cell(metric_name)
                value_stats = metric_data.get("value", {})
                mean = value_stats.get("mean")
                std = value_stats.get("std")

                mean_str = f"{mean:.4f}" if mean is not None else "N/A"
                std_str = f"{std:.4f}" if std is not None else "N/A"

                lines.append(f"| {task_name} | {metric_name_escaped} | {mean_str} | {std_str} | {num_reps} |")

        return "\n".join(lines)

    def detect_concurrency_issues(results):
        """Detect potential concurrency issues in evaluation results."""
        issues = []

        for result in results:
            task_name = result["task_name"]
            task_issues = []

            vllm_metrics = result.get("vllm_metrics_aggregate", {})
            if vllm_metrics:
                preemptions = vllm_metrics.get("preemptions_total", 0)
                if preemptions > 0:
                    task_issues.append(f"preemptions: {int(preemptions)}")

                avg_queue_time = vllm_metrics.get("queue_time_avg_seconds", 0)
                if avg_queue_time > 2.0:
                    task_issues.append(f"avg queue time: {avg_queue_time:.2f}s")

            log_stats = result.get("log_statistics_aggregate", {})
            if log_stats:
                kv_cache = log_stats.get("kv_cache_usage_pct", {})
                max_kv_cache = kv_cache.get("max", 0)
                if max_kv_cache > 85:
                    task_issues.append(f"max KV cache: {max_kv_cache:.1f}%")

            if task_issues:
                issues.append({
                    "task": task_name,
                    "issues": task_issues
                })

        return issues

    def format_concurrency_issues(issues):
        """Format concurrency issues as a markdown section."""
        if not issues:
            return ""

        lines = ["### Potential Concurrency Issues\n"]
        lines.append("The following tasks may have experienced resource contention:\n")
        lines.append("| Task | Issues |")
        lines.append("|------|--------|")

        for issue in issues:
            task_name = escape_markdown_table_cell(issue["task"])
            issues_str = ", ".join(issue["issues"])
            lines.append(f"| {task_name} | {issues_str} |")

        return "\n".join(lines)

    # Extract data from collated results
    server_info = collated_data.get("server", {})
    results = collated_data.get("results", [])

    pr_body_lines = [
        f"## Evaluation Results: {model_id}",
        "",
        f"**Session ID:** `{session_id}`",
        f"**Model:** `{model_id}`",
        "",
    ]

    # Add server info if available
    if server_info:
        pr_body_lines.append(format_server_info(server_info))
        pr_body_lines.append("")

    # Add metadata info
    if metadata:
        pr_body_lines.append("### Evaluation Metadata\n")
        if "parse_datetime" in metadata:
            pr_body_lines.append(f"**Parsed:** {metadata['parse_datetime']}")
        if "total_results_parsed" in metadata:
            pr_body_lines.append(f"**Total Results:** {metadata['total_results_parsed']}")
        if "unique_tasks" in metadata:
            pr_body_lines.append(f"**Tasks:** {', '.join(metadata['unique_tasks'])}")
        pr_body_lines.append("")

    # Add results table
    if results:
        pr_body_lines.append("### Task Results\n")
        pr_body_lines.append(format_results_table(results))
        pr_body_lines.append("")

    # Add concurrency issues section
    concurrency_issues = detect_concurrency_issues(results)
    if concurrency_issues:
        pr_body_lines.append(format_concurrency_issues(concurrency_issues))
        pr_body_lines.append("")

    pr_body_lines.append(f"**Full results file:** `{repo_file_path}`")

    pr_body = "\n".join(pr_body_lines)
    pr_title = f"Evaluation results: {model_id} (session {session_id})"

    pr_response = github_request(
        "POST",
        f"{GITHUB_API}/repos/{owner}/{repo}/pulls",
        github_token,
        json_data={
            "title": pr_title,
            "head": branch_name,
            "base": base_branch,
            "body": pr_body,
        },
    )

    pr_url = pr_response["html_url"]
    pr_number = pr_response["number"]

    print(f"Created PR #{pr_number}: {pr_url}")

    return f"Created PR #{pr_number}: {pr_url}"
