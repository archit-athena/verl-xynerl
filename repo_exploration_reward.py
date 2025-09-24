def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    """Custom reward function for repository exploration tasks."""
    if not solution_str or len(solution_str.strip()) < 50:
        return 0.0
    
    response_lower = solution_str.lower()
    score = 0.0
    
    # Check for tool usage evidence
    if any(word in response_lower for word in ["todo", "task", "plan", "list"]):
        score += 0.25
    if any(word in response_lower for word in ["ls", "bash", "command", "directory", "execute"]):
        score += 0.25
    if any(word in response_lower for word in ["read", "file", "examine", "code", "analyze"]):
        score += 0.25
    if len(solution_str.strip()) > 200:
        score += 0.25
    
    return min(score, 1.0)
