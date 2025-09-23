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
"""
Generate synthetic codebase exploration tasks for VERL training
"""

import argparse
import os
import random

import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


# Hardcoded repository exploration questions
REPO_QUESTIONS = [
    {
        "question": "Start by executing 'ls -la' to explore the repository structure. Then examine the main directories and provide a comprehensive overview of what this codebase contains and its architectural purpose.",
        "ground_truth": "repository_overview_analysis"
    },
    {
        "question": "Explore the 'verl/tools/' directory by listing its contents, then read the 'base_tool.py' file to understand the tool system architecture. Explain how tools are implemented in this framework.",
        "ground_truth": "tools_architecture_analysis"
    },
    {
        "question": "Find and examine configuration files in the 'examples/' directory. Use bash commands to locate YAML files, read a few examples, and explain the configuration system used for training.",
        "ground_truth": "configuration_system_analysis"
    }
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_save_dir", default="~/data/repo_exploration", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")

    args = parser.parse_args()

    data_source = "synthetic/repo_exploration"

    # Generate synthetic dataset by repeating questions
    train_data = []
    test_data = []
    
    for i in range(args.num_samples):
        question_data = random.choice(REPO_QUESTIONS)
        
        data_entry = {
            "data_source": data_source,
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are an experienced software engineer working within the VERL codebase. Your role is to systematically analyze, understand, and solve problems using a methodical approach with the available tools.\n\n"
                        
                        "## Your Working Style:\n"
                        "- **Be methodical**: Break complex problems into clear, manageable steps\n"
                        "- **Understand before acting**: Always examine the codebase structure and existing patterns before making changes\n"
                        "- **Document your process**: Use the todo system to track your investigation and implementation steps\n"
                        "- **Verify your work**: Test assumptions and validate solutions using available tools\n"
                        "- **Follow existing conventions**: Respect the codebase's architectural patterns and coding style\n\n"
                        
                        "## Tool Usage Protocol:\n"
                        "1. **Planning Phase**: Use `todo_manager` to create a structured investigation plan\n"
                        "2. **Discovery Phase**: Use `verl_bash` to explore directory structures and `verl_read_file` to examine relevant code\n"
                        "3. **Analysis Phase**: Study existing implementations, understand patterns, identify dependencies\n"
                        "4. **Implementation Phase**: Apply findings systematically, updating your todo list as you progress\n"
                        "5. **Verification Phase**: Test your solution and mark tasks complete\n\n"
                        
                        "## Repository Analysis Tasks:\n"
                        "When exploring the codebase:\n"
                        "- Always start with creating a todo list for your investigation\n"
                        "- Use bash commands to explore the file system structure\n"
                        "- Read key files to understand implementation patterns\n"
                        "- Provide clear summaries and architectural insights\n"
                        "- Mark your investigation tasks complete as you finish them\n\n"
                        
                        "Be thorough, methodical, and professional. Focus on understanding the codebase architecture and providing clear insights."
                    ),
                },
                {
                    "role": "user",
                    "content": question_data["question"],
                },
            ],
            "ability": "code_analysis",
            "reward_model": {"style": "rule", "ground_truth": question_data["ground_truth"]},
            "extra_info": {
                "split": "train" if i < args.num_samples * 0.8 else "test",
                "index": i,
                "question": question_data["question"],
                "task_type": "repository_exploration",
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    # Todo manager tool for task organization
                    "todo_manager": {
                        "create_kwargs": {"ground_truth": question_data["ground_truth"]},
                    },
                    # VERL bash tool for exploration
                    "bash": {
                        "create_kwargs": {"ground_truth": question_data["ground_truth"]},
                    },
                    # VERL file reader tool for code examination
                    "read_file": {
                        "create_kwargs": {"ground_truth": question_data["ground_truth"]},
                    },
                },
                "interaction_kwargs": {
                    "query": question_data["question"],
                    "ground_truth": question_data["ground_truth"],
                },
            },
        }
        
        if i < args.num_samples * 0.8:
            train_data.append(data_entry)
        else:
            test_data.append(data_entry)

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    local_save_dir = args.local_dir if args.local_dir is not None else args.local_save_dir

    # Ensure the directory exists
    os.makedirs(os.path.expanduser(local_save_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(os.path.expanduser(local_save_dir), "train.parquet"))
    test_dataset.to_parquet(os.path.join(os.path.expanduser(local_save_dir), "test.parquet"))

    print(f"Repository exploration dataset generation complete!")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Saved to: {os.path.expanduser(local_save_dir)}")
    print(f"Questions used: {len(REPO_QUESTIONS)} unique questions")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)