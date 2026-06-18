from typing import List, Optional

import kfp
from kfp import dsl, kubernetes

PIPELINE_NAME = "github-upload-pipeline"

@dsl.component()
def create_json_artifact() -> str:
    """Create a small JSON artifact with timestamp and UUID in pvc storage."""
    from datetime import datetime, timezone
    import json
    import uuid
    from pathlib import Path

    # Generate timestamp and UUID
    timestamp = datetime.now(timezone.utc)
    uid = uuid.uuid4().hex[:8]
    formatted_date = timestamp.strftime("%Y%m%d-%H%M%S")

    # Create filename with required format
    filename = f"collated-{formatted_date}-{uid}.json"

    # Create the artifact content
    artifact_data = {
        "created_at_utc": timestamp.isoformat(),
        "uuid": uid,
        "filename": filename,
        "placeholder_data": {
            "status": "success",
            "pipeline": "github-upload-pipeline",
            "sample_metrics": {
                "count": 42,
                "score": 0.95
            }
        }
    }

    # Write to pvc storage location
    output_dir = Path("/tier2/tmp/example_jsons")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename

    # Write JSON file
    with output_file.open("w") as f:
        json.dump(artifact_data, f, indent=2, sort_keys=True)

    return str(output_file)

@dsl.component(
    packages_to_install=["requests"]
)
def submit_github_pr(
    artifact_path: str,
    owner: str = "Ryfernandes",
    repo: str = "llm-evaluation-pipeline-results",
    base_branch: str = "main",
    repo_results_dir: str = "results"
) -> str:
    """Submit the artifact file as a PR to GitHub repository."""
    import base64
    import os
    import time
    import uuid
    from pathlib import Path
    from urllib.parse import quote
    import requests

    # Read GitHub token from environment variable (injected via kubernetes.use_secret_as_env)
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set")

    # GitHub API helper function
    def github_request(
        method: str,
        url: str,
        token: str,
        json_data: dict = None,
    ) -> dict:
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

    local_file_path = Path(artifact_path)

    if not local_file_path.is_file():
        raise FileNotFoundError(f"Local file does not exist: {local_file_path}")

    filename = local_file_path.name

    # GitHub repository paths should not start with "/"
    repo_results_dir = repo_results_dir.strip("/")
    repo_file_path = f"{repo_results_dir}/{filename}"

    # 1. Get the current commit SHA of base branch
    ref_path = quote(f"heads/{base_branch}", safe="/")
    base_ref = github_request(
        "GET",
        f"{GITHUB_API}/repos/{owner}/{repo}/git/ref/{ref_path}",
        github_token,
    )
    base_sha = base_ref["object"]["sha"]

    # 2. Create a unique branch from base
    safe_stem = local_file_path.stem.replace(" ", "-")
    unique_suffix = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    branch_name = f"add-result-{safe_stem}-{unique_suffix}"

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
    file_bytes = local_file_path.read_bytes()
    encoded_content = base64.b64encode(file_bytes).decode("ascii")

    commit_message = f"Add result file {filename}"

    contents_path = quote(repo_file_path, safe="/")
    commit_response = github_request(
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
    pr_title = f"Add pipeline result {filename}"
    pr_body = f"Automated PR adding pipeline result: `{repo_file_path}`"

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

    return f"Created PR #{pr_number}: {pr_url}"

@dsl.pipeline(
    name="github-upload-pipeline",
    description="Demo pipeline that creates a JSON artifact and submits it to GitHub via PR"
)
def pipeline(
    github_owner: str = "Ryfernandes",
    github_repo: str = "llm-evaluation-pipeline-results",
    base_branch: str = "main",
):
    # Component 1: Create JSON artifact in pvc storage
    create_task = create_json_artifact()

    # Mount tier2 PVC for artifact creation
    kubernetes.mount_pvc(
        create_task,
        pvc_name="evaluation-pipeline-artifacts-tier-2",
        mount_path="/tier2" 
    )

    # Component 2: Submit to GitHub
    github_task = submit_github_pr(
        artifact_path=create_task.output,
        owner=github_owner,
        repo=github_repo,
        base_branch=base_branch
    )

    # Mount tier2 PVC for reading the artifact
    kubernetes.mount_pvc(
        github_task,
        pvc_name="evaluation-pipeline-artifacts-tier-2",
        mount_path="/tier2"
    )

    # Inject GitHub token from secret
    kubernetes.use_secret_as_env(
        github_task,
        secret_name="evaluation-pipeline-results-gh",
        secret_key_to_env={"GITHUB_TOKEN": "GITHUB_TOKEN"}
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )
