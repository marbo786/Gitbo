import pytest
from backend.agents import route_after_search, route_after_architect

def test_conditional_routing():
    # If there are errors, should route to cleanup
    state_with_error = {"errors": ["Some error"], "retrieved_context": [{"fake": "data"}]}
    assert route_after_search(state_with_error) == "cleanup"
    
    # If there's no retrieved context, should route to cleanup
    state_no_context = {"errors": [], "retrieved_context": []}
    assert route_after_search(state_no_context) == "cleanup"
    
    # Success path
    state_success = {"errors": [], "retrieved_context": [{"fake": "data"}]}
    assert route_after_search(state_success) == "solution_architect"
    
    # Same for architect
    state_arch_error = {"errors": ["Error"], "implementation_plan": {}}
    assert route_after_architect(state_arch_error) == "cleanup"
    
    state_arch_success = {"errors": [], "implementation_plan": {"plan": "..."}}
    assert route_after_architect(state_arch_success) == "pr_creator"
