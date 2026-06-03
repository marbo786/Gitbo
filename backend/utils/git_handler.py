import os
import re
from urllib.parse import urlparse
import git
from github import Github

def parse_github_url(url: str):
    """Extract owner and repo name from GitHub URL."""
    url = url.strip()
    if url.endswith('.git'):
        url = url[:-4]
    
    # Handle SSH URL format: git@github.com:owner/repo
    if url.startswith('git@'):
        parts = url.split(':')[-1].split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]
            
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split('/') if p]
    if len(parts) >= 2:
        return parts[0], parts[1]
        
    return None, None

def get_auth_url(repo_url: str, token: str) -> str:
    """Format the GitHub repository URL with the Personal Access Token for authentication."""
    parsed = urlparse(repo_url)
    netloc = parsed.netloc or "github.com"
    path = parsed.path
    if not path.endswith('.git'):
        path += '.git'
    return f"https://{token}@{netloc}{path}"

def clone_repo(repo_url: str, token: str, dest_dir: str) -> git.Repo:
    """Perform a shallow clone of the target repository."""
    auth_url = get_auth_url(repo_url, token)
    return git.Repo.clone_from(auth_url, dest_dir, depth=1)

def create_branch_and_commit(repo_dir: str, branch_name: str, commit_message: str):
    """Checkout a new branch, stage all changes, and commit them."""
    repo = git.Repo(repo_dir)
    # Check out branch (create if not exists)
    try:
        new_branch = repo.create_head(branch_name)
    except Exception:
        # Branch already exists, let's fetch it
        new_branch = repo.heads[branch_name]
        
    repo.head.reference = new_branch
    repo.head.reset(index=True, working_tree=True)
    
    # Stage all changes
    repo.git.add(A=True)
    commit = repo.index.commit(commit_message)
    return commit

def get_git_diff(repo_dir: str) -> str:
    """Get the git diff for the last commit (HEAD~1 vs HEAD)."""
    repo = git.Repo(repo_dir)
    try:
        # Try diffing against parent commit
        try:
            return repo.git.diff("HEAD~1", "HEAD")
        except Exception:
            # Fallback to diff of HEAD against origin HEAD or default branch
            try:
                return repo.git.diff("origin/HEAD", "HEAD")
            except Exception:
                try:
                    return repo.git.diff("origin/main", "HEAD")
                except Exception:
                    try:
                        return repo.git.diff("origin/master", "HEAD")
                    except Exception:
                        return repo.git.diff()
    except Exception as e:
        return f"Error retrieving git diff: {str(e)}"

def push_branch(repo_dir: str, repo_url: str, token: str, branch_name: str):
    """Push the branch to the remote origin."""
    repo = git.Repo(repo_dir)
    auth_url = get_auth_url(repo_url, token)
    
    # Update origin remote URL to the authenticated URL
    origin = repo.remote(name='origin')
    origin.set_url(auth_url)
    
    # Push HEAD to refs/heads/branch_name
    origin.push(refspec=f"HEAD:refs/heads/{branch_name}", force=True)

def create_pull_request(repo_url: str, token: str, branch_name: str, title: str, body: str) -> str:
    """Open a pull request against the main repository default branch, rescuing existing PRs."""
    owner, repo_name = parse_github_url(repo_url)
    if not owner or not repo_name:
        raise ValueError(f"Could not parse owner and repository name from URL: {repo_url}")
        
    g = Github(token)
    gh_repo = g.get_repo(f"{owner}/{repo_name}")
    default_branch = gh_repo.default_branch
    
    try:
        pr = gh_repo.create_pull(
            title=title,
            body=body,
            head=branch_name,
            base=default_branch
        )
        return pr.html_url
    except Exception as e:
        # Check if the error is due to an existing open pull request
        if "already exists" in str(e) or "422" in str(e):
            try:
                # Query for open PRs matching owner:branch
                pulls = gh_repo.get_pulls(state='open', head=f"{owner}:{branch_name}")
                if pulls.totalCount > 0:
                    print(f"[Git Handler] Pull request already exists, returning: {pulls[0].html_url}")
                    return pulls[0].html_url
            except Exception:
                pass
        raise e
