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
Preprocess the GSM8k dataset to parquet format with multi-tool support
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer after `####`."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    
                    {
                        "role": "system",
            "content": 
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
                
                "## For Math Problems:\n"
                "Apply the same systematic approach:\n"
                "- Create a todo list for the problem-solving steps\n"
                "- Use bash tools for calculations when helpful\n"
                "- Show your work clearly\n"
                "- Always provide the final answer in the format: #### <answer>\n\n"
                
                "Be thorough, methodical, and professional. Focus on creating maintainable, well-integrated solutions."
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        # Todo manager tool for task organization
                        "todo_manager": {
                            "create_kwargs": {"ground_truth": solution},
                        },
                        # VERL bash tool for calculations and exploration
                        "bash": {
                            "create_kwargs": {"ground_truth": solution},
                        },
                        # VERL file reader tool for documentation access
                        "read_file": {
                            "create_kwargs": {"ground_truth": solution},
                        },
                        # Keep the original calc_gsm8k_reward for compatibility
                        "calc_gsm8k_reward": {
                            "create_kwargs": {"ground_truth": solution},
                        },
                    },
                    "interaction_kwargs": {
                        "query": question,
                        "ground_truth": solution,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    # Ensure the directory exists
    os.makedirs(os.path.expanduser(local_save_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(os.path.expanduser(local_save_dir), "train.parquet"))
    test_dataset.to_parquet(os.path.join(os.path.expanduser(local_save_dir), "test.parquet"))

    print(f"Dataset preprocessing complete!")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Saved to: {os.path.expanduser(local_save_dir)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)