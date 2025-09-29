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

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    """Simple hardcoded reward function for repository exploration tasks."""
    
    if not solution_str or len(solution_str.strip()) < 50:
        return 0.0
    
    response_lower = solution_str.lower()
    score = 0.0
    
    # Tool usage indicators (40% of score)
    if any(word in response_lower for word in ["todo", "task", "plan", "list"]):
        score += 0.15
    if any(word in response_lower for word in ["ls", "bash", "command", "directory", "execute", "pwd", "find"]):
        score += 0.15
    if any(word in response_lower for word in ["read", "file", "examine", "code", "analyze"]):
        score += 0.10
    
    # Analysis quality (30% of score)
    if any(word in response_lower for word in ["structure", "architecture", "organize", "system"]):
        score += 0.10
    if any(word in response_lower for word in ["understand", "analyze", "explore", "investigate"]):
        score += 0.10
    if any(word in response_lower for word in ["implement", "pattern", "design", "method"]):
        score += 0.10
    
    # Completeness (30% of score)
    length = len(solution_str.strip())
    if length >= 500:
        score += 0.30
    elif length >= 300:
        score += 0.20
    elif length >= 150:
        score += 0.10
    
    return min(score, 1.0)