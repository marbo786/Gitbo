import pytest
from backend.utils.git_handler import parse_github_url, get_auth_url

def test_parse_github_url():
    assert parse_github_url("https://github.com/owner/repo") == ("owner", "repo")
    assert parse_github_url("https://github.com/owner/repo.git") == ("owner", "repo")
    assert parse_github_url("git@github.com:owner/repo.git") == ("owner", "repo")
    assert parse_github_url("https://github.com/owner/repo/") == ("owner", "repo")

def test_get_auth_url():
    # Token should be URL-encoded to prevent injection via special chars
    url = "https://github.com/owner/repo"
    token = "ghp_1234567890#@"
    auth_url = get_auth_url(url, token)
    assert "ghp_1234567890%23%40" in auth_url
    assert auth_url.startswith("https://ghp_1234567890%23%40@github.com/owner/repo.git")
