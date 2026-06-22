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

    # 4. Open a PR from the new branch to base
    pr_title = f"Evaluation results for session {session_id}"
    pr_body = f"Automated PR adding evaluation results: `{repo_file_path}`\n\nSession ID: `{session_id}`"

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
