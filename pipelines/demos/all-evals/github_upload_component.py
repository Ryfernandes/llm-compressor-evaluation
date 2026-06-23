from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["requests"]
)
def upload_results_to_github(
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
    compression_recipe = metadata.get("compression_recipe", "N/A")
    baseline_results = collated_data.get("baseline_results", [])
    compressed_results = collated_data.get("compressed_results", [])

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
    def format_results_table(results, result_type):
        """Format results as a markdown table."""
        if not results:
            return f"No {result_type} results available.\n"

        lines = [f"### {result_type.title()} Results\n"]
        lines.append("| Task | Metric | Mean | Std Dev | Reps |")
        lines.append("|------|--------|------|---------|------|")

        for result in sorted(results, key=lambda x: x["task_name"]):
            task_name = result["task_name"]
            aggregate_stats = result.get("aggregate_stats", {})
            metrics = aggregate_stats.get("metrics", {})
            num_reps = result.get("num_repetitions", 0)

            for metric_name, metric_data in metrics.items():
                value_stats = metric_data.get("value", {})
                mean = value_stats.get("mean")
                std = value_stats.get("std")

                mean_str = f"{mean:.4f}" if mean is not None else "N/A"
                std_str = f"{std:.4f}" if std is not None else "N/A"

                lines.append(f"| {task_name} | {metric_name} | {mean_str} | {std_str} | {num_reps} |")

        return "\n".join(lines)

    pr_body_lines = [
        f"## Evaluation Results: {model_id}",
        "",
        f"**Session ID:** `{session_id}`",
        f"**Model:** `{model_id}`",
        "",
        "**Compression Recipe:**",
        "```yaml",
        compression_recipe.strip(),
        "```",
        "",
        format_results_table(baseline_results, "baseline"),
        "",
        format_results_table(compressed_results, "compressed"),
        "",
        f"**Full results file:** `{repo_file_path}`",
    ]

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
