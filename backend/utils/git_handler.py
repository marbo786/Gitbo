import os
import re
from urllib.parse import urlparse, quote
import git
from github import Github

GITHUB_HOST = "github.com"


def parse_github_url(url: str):
    """Extract owner and repo name from a GitHub HTTPS or SSH URL."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # SSH: git@github.com:owner/repo
    if url.startswith("git@"):
        parts = url.split(":")[-1].split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def get_auth_url(repo_url: str, token: str) -> str:
    """
    Build an HTTPS clone URL with token auth.
    The token is URL-encoded so characters like '#', '@', ':' cannot break the URL.
    """
    parsed = urlparse(repo_url.strip())
    netloc = parsed.netloc or GITHUB_HOST
    path = parsed.path
    if not path.endswith(".git"):
        path += ".git"
    # URL-encode to prevent injection via special chars in the token
    encoded_token = quote(token, safe="")
    return f"https://{encoded_token}@{netloc}{path}"


def _scrub_token(url: str) -> str:
    """Return a URL with the token replaced by *** — safe for logging."""
    return re.sub(r"https://[^@]+@", "https://***@", url)


def clone_repo(repo_url: str, token: str, dest_dir: str) -> git.Repo:
    """Shallow-clone the repository into dest_dir."""
    auth_url = get_auth_url(repo_url, token)
    return git.Repo.clone_from(auth_url, dest_dir, depth=1)


def create_branch_and_commit(repo_dir: str, branch_name: str, commit_message: str):
    """
    Create a branch at the current HEAD, then stage and commit all
    working-tree changes.

    FIX #1: The previous implementation called repo.head.reset(working_tree=True)
    AFTER switching the branch reference.  That reset discarded the edits that
    apply_search_replace() had just written to disk, so the commit was empty.

    Correct flow:
      1. Create the branch pointer at current HEAD (no working-tree touch).
      2. Switch HEAD to that branch (symbolic-ref only — no reset).
      3. Stage all changes with git add -A.
      4. Commit.
    """
    repo = git.Repo(repo_dir)

    # If the branch already exists, remove it so we start fresh from current HEAD
    if branch_name in [h.name for h in repo.heads]:
        repo.delete_head(branch_name, force=True)

    new_branch = repo.create_head(branch_name)

    # Switch HEAD's symbolic ref to the new branch.
    # This does NOT touch the index or working tree — edits remain intact.
    repo.head.reference = new_branch

    # Stage every change (modifications, new files, deletions)
    repo.git.add(A=True)

    commit = repo.index.commit(commit_message)
    return commit


def get_git_diff(repo_dir: str) -> str:
    """Return the diff for the most recent commit."""
    repo = git.Repo(repo_dir)
    try:
        try:
            return repo.git.diff("HEAD~1", "HEAD")
        except Exception:
            pass
        for ref in ("origin/HEAD", "origin/main", "origin/master"):
            try:
                return repo.git.diff(ref, "HEAD")
            except Exception:
                continue
        return repo.git.diff()
    except Exception as e:
        return f"Error retrieving git diff: {e}"


def push_branch(repo_dir: str, repo_url: str, token: str, branch_name: str):
    """
    Push the current branch to the remote.

    FIX #4: The old implementation stored the token-bearing URL as the
    persistent remote origin URL, making it visible via `git remote -v`,
    exception messages, and debug logs.

    We now set the auth URL only for the duration of the push, then
    immediately restore the non-auth URL so the token is never persisted
    in .git/config.
    """
    repo = git.Repo(repo_dir)
    auth_url = get_auth_url(repo_url, token)
    # Canonical non-auth URL to restore after the push
    clean_url = repo_url if repo_url.endswith(".git") else repo_url + ".git"

    origin = repo.remote(name="origin")
    origin.set_url(auth_url)
    try:
        origin.push(refspec=f"HEAD:refs/heads/{branch_name}", force=True)
    finally:
        # Restore non-auth URL regardless of push success/failure
        try:
            origin.set_url(clean_url)
        except Exception:
            pass


def create_pull_request(
    repo_url: str, token: str, branch_name: str, title: str, body: str
) -> str:
    """Open a pull request; return the PR URL. Handles existing open PRs."""
    owner, repo_name = parse_github_url(repo_url)
    if not owner or not repo_name:
        raise ValueError(f"Could not parse owner/repo from URL: {repo_url}")

    g = Github(token)
    gh_repo = g.get_repo(f"{owner}/{repo_name}")
    default_branch = gh_repo.default_branch

    try:
        pr = gh_repo.create_pull(
            title=title, body=body, head=branch_name, base=default_branch
        )
        return pr.html_url
    except Exception as e:
        if "already exists" in str(e) or "422" in str(e):
            try:
                pulls = gh_repo.get_pulls(state="open", head=f"{owner}:{branch_name}")
                if pulls.totalCount > 0:
                    print(f"[Git Handler] PR already exists: {pulls[0].html_url}")
                    return pulls[0].html_url
            except Exception:
                pass
        raise
