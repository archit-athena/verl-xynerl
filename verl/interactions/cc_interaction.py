# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RepoExplorationInteraction(BaseInteraction):
    """An interaction for guiding and evaluating repository exploration tasks.

    This interaction helps agents systematically explore codebases by:
    - Encouraging structured investigation approaches
    - Providing feedback on exploration completeness
    - Rewarding comprehensive analysis and proper tool usage
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.max_turns = config.get("max_turns", 5)
        self.min_analysis_length = config.get("min_analysis_length", 200)

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "turn_count": 0,
            "tool_usage_score": 0.0,
            "analysis_quality_score": 0.0,
            "completeness_score": 0.0,
            "has_used_todo": False,
            "has_used_bash": False,
            "has_used_file_reader": False,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        
        if instance_id not in self._instance_dict:
            return True, "Error: Instance not found", 0.0, {}
        
        instance = self._instance_dict[instance_id]
        instance["turn_count"] += 1
        
        # Extract the latest assistant response
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content", "")
                break
        
        instance["response"] = content
        
        # Analyze tool usage patterns in the response
        self._analyze_tool_usage(instance, content)
        
        # Calculate current turn score using the reward function logic
        turn_score = await self.calculate_score(instance_id)
        
        # Determine if interaction should continue
        should_terminate, feedback = self._evaluate_completion(instance, content)
        
        return should_terminate, feedback, turn_score, {
            "turn_count": instance["turn_count"],
            "tool_usage_score": instance["tool_usage_score"],
            "analysis_quality": instance["analysis_quality_score"]
        }

    def _analyze_tool_usage(self, instance: dict, content: str) -> None:
        """Analyze what tools were used in the response."""
        content_lower = content.lower()
        
        # Check for tool usage indicators
        if any(keyword in content_lower for keyword in ["todo", "task", "plan", "step"]):
            instance["has_used_todo"] = True
        
        if any(keyword in content_lower for keyword in ["ls", "bash", "command", "execute"]):
            instance["has_used_bash"] = True
            
        if any(keyword in content_lower for keyword in ["read", "file", "examine", "content"]):
            instance["has_used_file_reader"] = True
        
        # Calculate tool usage score
        tools_used = sum([
            instance["has_used_todo"],
            instance["has_used_bash"], 
            instance["has_used_file_reader"]
        ])
        instance["tool_usage_score"] = tools_used / 3.0

    def _evaluate_completion(self, instance: dict, content: str) -> tuple[bool, str]:
        """Evaluate if the exploration is complete and generate feedback."""
        
        content_lower = content.lower()
        ground_truth = instance.get("ground_truth", "")
        
        # Use the same scoring logic as the reward function
        analysis_score = self._calculate_analysis_quality(content_lower, ground_truth)
        instance["analysis_quality_score"] = analysis_score
        
        # Check completeness
        is_comprehensive = len(content) >= self.min_analysis_length
        has_good_tool_usage = instance["tool_usage_score"] >= 0.67  # Used at least 2/3 tools
        has_good_analysis = instance["analysis_quality_score"] >= 0.67
        
        instance["completeness_score"] = sum([is_comprehensive, has_good_tool_usage, has_good_analysis]) / 3.0
        
        # Determine termination and feedback
        if instance["completeness_score"] >= 0.8:
            return True, "Excellent analysis! You've systematically explored the codebase using multiple tools and provided comprehensive insights."
        
        elif instance["turn_count"] >= self.max_turns:
            return True, f"Analysis complete after {self.max_turns} turns. Consider using more tools for deeper exploration next time."
        
        else:
            # Provide constructive feedback to continue exploration
            feedback_parts = ["Good progress! Here's how to improve your analysis:"]
            
            if not instance["has_used_todo"]:
                feedback_parts.append("• Use the todo_manager to create a structured exploration plan")
            
            if not instance["has_used_bash"]:
                feedback_parts.append("• Use bash to explore directory structures and file listings")
                
            if not instance["has_used_file_reader"]:
                feedback_parts.append("• Use read_file to examine key implementation files")
            
            if len(content) < self.min_analysis_length:
                feedback_parts.append("• Provide more detailed analysis and insights")
                
            if instance["analysis_quality_score"] < 0.5:
                feedback_parts.append(f"• Focus on the key aspects for {ground_truth.replace('_', ' ')}")
            
            return False, "\n".join(feedback_parts)

    def _calculate_analysis_quality(self, response_lower: str, ground_truth: str) -> float:
        """Calculate analysis quality score - mirrors the reward function logic."""
        score = 0.0
        
        if ground_truth == "repository_overview_analysis":
            if any(word in response_lower for word in ["structure", "architecture", "organize", "framework"]):
                score += 0.3
            if any(word in response_lower for word in ["purpose", "goal", "function", "system"]):
                score += 0.3
            if any(word in response_lower for word in ["component", "module", "tool", "directory"]):
                score += 0.2
            if any(word in response_lower for word in ["verl", "reinforcement", "learning", "training"]):
                score += 0.2
                
        elif ground_truth == "tools_architecture_analysis":
            if any(word in response_lower for word in ["basetool", "base_tool", "inherit", "abstract"]):
                score += 0.4
            if any(word in response_lower for word in ["method", "async", "execute", "create"]):
                score += 0.3
            if any(word in response_lower for word in ["schema", "config", "register", "load"]):
                score += 0.3
                
        elif ground_truth == "configuration_system_analysis":
            if any(word in response_lower for word in ["yaml", "config", "parameter", "setting"]):
                score += 0.4
            if any(word in response_lower for word in ["example", "template", "sample"]):
                score += 0.3
            if any(word in response_lower for word in ["model", "batch", "learning", "optimizer"]):
                score += 0.3
        
        else:
            # Generic analysis quality
            if any(word in response_lower for word in ["analyze", "understand", "examine", "investigate"]):
                score += 0.5
            if any(word in response_lower for word in ["pattern", "structure", "design", "implement"]):
                score += 0.3
            if any(word in response_lower for word in ["conclusion", "summary", "insight", "finding"]):
                score += 0.2
        
        return min(score, 1.0)

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate comprehensive score for the repository exploration."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance = self._instance_dict[instance_id]
        
        # Weighted combination of different scoring aspects
        tool_weight = 0.4  # Emphasize tool usage
        analysis_weight = 0.4  # Quality of analysis
        completeness_weight = 0.2  # Overall completeness
        
        total_score = (
            instance["tool_usage_score"] * tool_weight +
            instance["analysis_quality_score"] * analysis_weight +
            instance["completeness_score"] * completeness_weight
        )
        
        return min(total_score, 1.0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up interaction instance."""
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            logger.debug(f"Finalizing repo exploration interaction {instance_id}: "
                        f"turns={instance['turn_count']}, "
                        f"final_score={instance.get('completeness_score', 0.0):.2f}")
            del self._instance_dict[instance_id]