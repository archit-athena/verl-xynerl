# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None, **kwargs) -> float:
    """
    Compute reward score for repository exploration tasks.
    
    Args:
        data_source: The data source identifier
        solution_str: The agent's response/analysis
        ground_truth: Expected analysis type (e.g., "repository_overview_analysis")
        extra_info: Additional context information
        **kwargs: Additional parameters
    
    Returns:
        Float score between 0.0 and 1.0
    """
    if not solution_str or len(solution_str.strip()) < 30:
        return 0.0
    
    response_lower = solution_str.lower()
    score = 0.0
    
    # Base scoring components
    tool_usage_score = _evaluate_tool_usage(response_lower)
    analysis_quality_score = _evaluate_analysis_quality(response_lower, ground_truth)
    completeness_score = _evaluate_completeness(solution_str, response_lower)
    methodology_score = _evaluate_methodology(response_lower)
    
    # Weighted combination
    score = (
        tool_usage_score * 0.35 +      # Heavy weight on tool usage
        analysis_quality_score * 0.30 +  # Quality of analysis 
        completeness_score * 0.20 +     # Thoroughness
        methodology_score * 0.15        # Systematic approach
    )
    
    # Bonus points for exceptional responses
    bonus = _calculate_bonus_points(response_lower, solution_str)
    score = min(score + bonus, 1.0)
    
    logger.debug(f"Repo exploration score: {score:.3f} (tool:{tool_usage_score:.2f}, quality:{analysis_quality_score:.2f}, complete:{completeness_score:.2f}, method:{methodology_score:.2f}, bonus:{bonus:.2f})")
    
    return score


def _evaluate_tool_usage(response_lower: str) -> float:
    """Evaluate how well the agent used available tools."""
    score = 0.0
    
    # Todo manager usage
    todo_indicators = ["todo", "task", "plan", "step", "organize", "list"]
    if any(indicator in response_lower for indicator in todo_indicators):
        score += 0.4
        
        # Bonus for specific todo actions
        if any(action in response_lower for action in ["add", "complete", "mark"]):
            score += 0.1
    
    # Bash tool usage
    bash_indicators = ["ls", "bash", "command", "execute", "directory", "find", "grep", "cat"]
    if any(indicator in response_lower for indicator in bash_indicators):
        score += 0.4
        
        # Bonus for multiple bash commands
        bash_command_count = sum(1 for cmd in ["ls", "find", "grep", "cat", "pwd", "tree"] if cmd in response_lower)
        if bash_command_count >= 2:
            score += 0.1
    
    # File reader usage
    file_indicators = ["read", "file", "examine", "content", "code", "implementation"]
    if any(indicator in response_lower for indicator in file_indicators):
        score += 0.3
        
        # Bonus for reading specific file types
        if any(ext in response_lower for ext in [".py", ".yaml", ".md", ".txt"]):
            score += 0.05
    
    return min(score, 1.0)


def _evaluate_analysis_quality(response_lower: str, ground_truth: str) -> float:
    """Evaluate the quality of analysis based on the expected ground truth."""
    score = 0.0
    
    if ground_truth == "repository_overview_analysis":
        # Check for architectural understanding
        if any(word in response_lower for word in ["structure", "architecture", "organize", "framework"]):
            score += 0.3
        
        # Check for purpose understanding
        if any(word in response_lower for word in ["purpose", "goal", "function", "system"]):
            score += 0.3
        
        # Check for component identification
        if any(word in response_lower for word in ["component", "module", "tool", "directory"]):
            score += 0.2
        
        # Check for VERL-specific knowledge
        if any(word in response_lower for word in ["verl", "reinforcement", "learning", "training"]):
            score += 0.2
            
    elif ground_truth == "tools_architecture_analysis":
        # Check for base tool understanding
        if any(word in response_lower for word in ["basetool", "base_tool", "inherit", "abstract"]):
            score += 0.4
        
        # Check for implementation pattern understanding
        if any(word in response_lower for word in ["method", "async", "execute", "create"]):
            score += 0.3
        
        # Check for tool system understanding
        if any(word in response_lower for word in ["schema", "config", "register", "load"]):
            score += 0.3
            
    elif ground_truth == "configuration_system_analysis":
        # Check for config understanding
        if any(word in response_lower for word in ["yaml", "config", "parameter", "setting"]):
            score += 0.4
        
        # Check for examples understanding
        if any(word in response_lower for word in ["example", "template", "sample"]):
            score += 0.3
        
        # Check for training config understanding
        if any(word in response_lower for word in ["model", "batch", "learning", "optimizer"]):
            score += 0.3
    
    else:
        # Generic analysis quality for unknown ground truth
        if any(word in response_lower for word in ["analyze", "understand", "examine", "investigate"]):
            score += 0.5
        
        if any(word in response_lower for word in ["pattern", "structure", "design", "implement"]):
            score += 0.3
        
        if any(word in response_lower for word in ["conclusion", "summary", "insight", "finding"]):
            score += 0.2
    
    return min(score, 1.0)


def _evaluate_completeness(solution_str: str, response_lower: str) -> float:
    """Evaluate how complete and thorough the analysis is."""
    score = 0.0
    
    # Length-based completeness
    length = len(solution_str.strip())
    if length >= 500:
        score += 0.4
    elif length >= 300:
        score += 0.3
    elif length >= 150:
        score += 0.2
    elif length >= 50:
        score += 0.1
    
    # Depth indicators
    depth_indicators = [
        "detailed", "comprehensive", "thorough", "extensive", "deep",
        "multiple", "various", "different", "several", "many"
    ]
    depth_count = sum(1 for indicator in depth_indicators if indicator in response_lower)
    if depth_count >= 3:
        score += 0.3
    elif depth_count >= 2:
        score += 0.2
    elif depth_count >= 1:
        score += 0.1
    
    # Coverage indicators
    if any(word in response_lower for word in ["overview", "summary", "conclusion", "insights"]):
        score += 0.2
    
    # Evidence of exploration breadth
    exploration_words = ["explore", "investigate", "discover", "find", "identify", "locate"]
    if sum(1 for word in exploration_words if word in response_lower) >= 2:
        score += 0.1
    
    return min(score, 1.0)


def _evaluate_methodology(response_lower: str) -> float:
    """Evaluate how systematic and methodical the approach was."""
    score = 0.0
    
    # Systematic approach indicators
    method_words = ["first", "next", "then", "finally", "step", "phase", "approach", "method"]
    method_count = sum(1 for word in method_words if word in response_lower)
    if method_count >= 3:
        score += 0.4
    elif method_count >= 2:
        score += 0.3
    elif method_count >= 1:
        score += 0.2
    
    # Planning indicators
    if any(word in response_lower for word in ["plan", "strategy", "workflow", "process"]):
        score += 0.3
    
    # Verification indicators
    if any(word in response_lower for word in ["verify", "check", "confirm", "validate"]):
        score += 0.2
    
    # Organization indicators
    if any(word in response_lower for word in ["organize", "structure", "systematic", "methodical"]):
        score += 0.2
    
    return min(score, 1.0)


def _calculate_bonus_points(response_lower: str, full_response: str) -> float:
    """Calculate bonus points for exceptional responses."""
    bonus = 0.0
    
    # Bonus for using all three tool types
    has_todo = any(word in response_lower for word in ["todo", "task", "plan"])
    has_bash = any(word in response_lower for word in ["ls", "bash", "command"])
    has_file = any(word in response_lower for word in ["read", "file", "examine"])
    
    if has_todo and has_bash and has_file:
        bonus += 0.1
    
    # Bonus for code examples or specific file references
    if any(pattern in full_response for pattern in [".py", "def ", "class ", "import "]):
        bonus += 0.05
    
    # Bonus for structured output (lists, sections, etc.)
    if full_response.count('\n') >= 10 and any(char in full_response for char in ['*', '-', '1.', '2.']):
        bonus += 0.05
    
    # Bonus for technical depth
    technical_terms = ["async", "await", "inherit", "class", "method", "function", "api", "schema"]
    tech_count = sum(1 for term in technical_terms if term in response_lower)
    if tech_count >= 3:
        bonus += 0.05
    
    # Bonus for actionable insights
    if any(word in response_lower for word in ["recommend", "suggest", "should", "could", "improve"]):
        bonus += 0.03
    
    return bonus


def compute_score_batch(data_list, **kwargs):
    """Batch version of compute_score for processing multiple samples at once."""
    scores = []
    for data in data_list:
        score = compute_score(
            data_source=data.get("data_source", ""),
            solution_str=data.get("solution_str", ""),
            ground_truth=data.get("ground_truth", ""),
            extra_info=data.get("extra_info"),
            **kwargs
        )
        scores.append(score)
    return scores