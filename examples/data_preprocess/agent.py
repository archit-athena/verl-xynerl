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
Generate synthetic coding agent tasks for software engineering training
"""

import argparse
import os
import random

import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


# Comprehensive SWE task questions
SWE_TASKS = [
    # Repository exploration tasks
    {
        "question": "Start by executing 'ls -la' to explore the repository structure. Then examine the main directories and provide a comprehensive overview of what this codebase contains and its architectural purpose.",
        "ground_truth": "repository_overview_analysis",
        "requires_editing": False
    },
    {
        "question": "Analyze the project architecture. Use bash to find main entry points, read key files, and explain how the components interact.",
        "ground_truth": "architecture_analysis",
        "requires_editing": False
    },
    {
        "question": "Find and examine configuration files. Summarize how the configuration system is designed and what parameters are configurable.",
        "ground_truth": "configuration_system_analysis",
        "requires_editing": False
    },
    
    # Code analysis tasks
    {
        "question": "Find all Python files in the src/ directory, read the main modules, and create a summary of the codebase structure and key components.",
        "ground_truth": "codebase_structure_analysis",
        "requires_editing": False
    },
    {
        "question": "Investigate the testing infrastructure. Find test files, analyze the testing approach, and summarize the test coverage strategy.",
        "ground_truth": "testing_infrastructure_analysis",
        "requires_editing": False
    },
    {
        "question": "Examine the API endpoints or interfaces. List all available endpoints/functions and document their purposes.",
        "ground_truth": "api_documentation_analysis",
        "requires_editing": False
    },
    
    # Debugging and investigation tasks
    {
        "question": "Search for TODO, FIXME, or BUG comments in the codebase. Create a prioritized list of issues that need attention.",
        "ground_truth": "technical_debt_analysis",
        "requires_editing": False
    },
    {
        "question": "Find deprecated functions or outdated patterns in the code. Document what needs to be modernized.",
        "ground_truth": "code_modernization_analysis",
        "requires_editing": False
    },
    {
        "question": "Analyze import statements across the project. Identify dependencies and potential circular import issues.",
        "ground_truth": "dependency_analysis",
        "requires_editing": False
    },
    
    # Simple editing tasks
    {
        "question": "Find the main configuration file, read it, and update the version number from 1.0.0 to 1.0.1. Use edit_file tool to make this change.",
        "ground_truth": "version_update",
        "requires_editing": True
    },
    {
        "question": "Locate the README.md file and add a new section about contributing. Read the file first, then use edit_file to insert the new section.",
        "ground_truth": "documentation_update",
        "requires_editing": True
    },
    {
        "question": "Find Python files with hardcoded localhost URLs. Read one such file and replace 'http://localhost' with 'http://0.0.0.0' using edit_file.",
        "ground_truth": "configuration_fix",
        "requires_editing": True
    },
    
    # Refactoring tasks
    {
        "question": "Find functions longer than 50 lines. Read one, analyze it, and suggest how to break it into smaller functions. Then implement the refactoring using edit_file.",
        "ground_truth": "code_refactoring",
        "requires_editing": True
    },
    {
        "question": "Locate duplicate code blocks. Read the files, identify the duplication, and refactor by creating a shared utility function using edit_file.",
        "ground_truth": "code_deduplication",
        "requires_editing": True
    },
    
    # Bug fixing tasks
    {
        "question": "Search for print() statements used for debugging. Read the files and replace them with proper logging using edit_file.",
        "ground_truth": "logging_improvement",
        "requires_editing": True
    },
    {
        "question": "Find exception handling blocks that catch generic 'Exception'. Read the code and update them to catch specific exceptions using edit_file.",
        "ground_truth": "exception_handling_fix",
        "requires_editing": True
    },
    
    # Feature implementation tasks
    {
        "question": "Add a new configuration parameter. First read the config file to understand the format, then add 'enable_caching: true' using edit_file.",
        "ground_truth": "feature_addition",
        "requires_editing": True
    },
    {
        "question": "Find the main module, read it to understand the structure, then add a new helper function for input validation using edit_file.",
        "ground_truth": "utility_function_addition",
        "requires_editing": True
    },
    
    # Documentation tasks
    {
        "question": "Find functions without docstrings. Read several files and add proper docstrings using edit_file.",
        "ground_truth": "documentation_improvement",
        "requires_editing": True
    },
    {
        "question": "Update outdated comments in the code. Read files, identify outdated comments, and update them using edit_file.",
        "ground_truth": "comment_update",
        "requires_editing": True
    }
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_save_dir", default="~/data/swe_agent", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to generate")

    args = parser.parse_args()

    data_source = "synthetic/repo_exploration"

    # Generate synthetic dataset by repeating questions
    train_data = []
    test_data = []
    
    for i in range(args.num_samples):
        task_data = random.choice(SWE_TASKS)
        
        # System prompt based on whether editing is required
        if task_data["requires_editing"]:
            system_content = (
                "You are an experienced software engineer and coding agent. Your role is to systematically analyze codebases and make precise, safe edits.\n\n"
                
                "## Your Working Style:\n"
                "- **Plan first**: Create a todo list outlining investigation and editing steps\n"
                "- **Read before edit**: ALWAYS read a file completely using read_file BEFORE using edit_file\n"
                "- **Be precise**: When editing, provide exact strings to match - check for whitespace, indentation, and special characters\n"
                "- **Verify changes**: After editing, read the file again to confirm changes\n"
                "- **One change at a time**: Make focused, atomic changes rather than large sweeps\n\n"
                
                "## Tool Usage Protocol:\n"
                "1. **Planning**: Use `todo_manager` to plan your investigation and changes\n"
                "2. **Discovery**: Use `bash` to find relevant files\n"
                "3. **Analysis**: Use `read_file` to understand current code\n"
                "4. **Editing**: Use `edit_file` with exact old_string and new_string\n"
                "5. **Verification**: Read files again to confirm edits\n\n"
                
                "## CRITICAL - File Editing Rules:\n"
                "- NEVER use edit_file without reading the file first with read_file\n"
                "- Provide EXACT strings including whitespace, indentation, and line breaks\n"
                "- If unsure about exact string format, read the file again\n"
                "- Test on small changes before larger refactorings\n\n"
                
                "Be methodical, precise, and verify your work at each step."
            )
        else:
            system_content = (
                "You are an experienced software engineer specializing in code analysis and architecture review.\n\n"
                
                "## Your Working Style:\n"
                "- **Be methodical**: Break complex analysis into clear steps\n"
                "- **Use tools effectively**: Leverage bash for exploration and read_file for detailed analysis\n"
                "- **Document findings**: Use todo_manager to track investigation progress\n"
                "- **Provide insights**: Deliver clear, actionable analysis\n\n"
                
                "## Tool Usage Protocol:\n"
                "1. **Planning**: Use `todo_manager` to create investigation plan\n"
                "2. **Discovery**: Use `bash` to explore file structure and locate relevant code\n"
                "3. **Analysis**: Use `read_file` to examine implementation details\n"
                "4. **Synthesis**: Provide comprehensive summary of findings\n\n"
                
                "Focus on understanding architecture, identifying patterns, and delivering valuable insights."
            )
        
        # Build tools_kwargs based on task requirements
        tools_kwargs = {
            "todo_manager": {
                "create_kwargs": {"ground_truth": task_data["ground_truth"]},
            },
            "bash": {
                "create_kwargs": {"ground_truth": task_data["ground_truth"]},
            },
            "read_file": {
                "create_kwargs": {"ground_truth": task_data["ground_truth"]},
            },
            "edit_file": {
                "create_kwargs": {"ground_truth": task_data["ground_truth"]},
            }
        }
        
        # Add edit_file tool for editing tasks
        if task_data["requires_editing"]:
            tools_kwargs["edit_file"] = {
                "create_kwargs": {"ground_truth": task_data["ground_truth"]},
            }
        
        data_entry = {
            "data_source": data_source,
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": task_data["question"],
                },
            ],
            "ability": "software_engineering",
            "reward_model": {"style": "rule", "ground_truth": task_data["ground_truth"]},
            "extra_info": {
                "split": "train" if i < args.num_samples * 0.8 else "test",
                "index": i,
                "question": task_data["question"],
                "task_type": "software_engineering_task",
                "requires_editing": task_data["requires_editing"],
                "need_tools_kwargs": True,
                "tools_kwargs": tools_kwargs,
                "interaction_kwargs": {
                    "query": task_data["question"],
                    "ground_truth": task_data["ground_truth"],
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

    print(f"SWE agent dataset generation complete!")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Saved to: {os.path.expanduser(local_save_dir)}")
    print(f"Task types: {len(SWE_TASKS)} unique tasks")
    print(f"Tasks with editing: {sum(1 for t in SWE_TASKS if t['requires_editing'])}")
    print(f"Analysis-only tasks: {sum(1 for t in SWE_TASKS if not t['requires_editing'])}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)