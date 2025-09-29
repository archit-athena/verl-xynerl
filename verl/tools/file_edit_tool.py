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

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class FileEditTool(BaseTool):
    """A tool for editing files by replacing text strings."""

    def __init__(self, config: Dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # Configuration parameters
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
        self.create_backup = config.get("create_backup", True)
        
        # Get working directory
        self.verl_pwd = os.environ.get("VERL_PWD")
        if self.verl_pwd is None:
            logger.warning("VERL_PWD environment variable not set, using current directory")
            self.verl_pwd = os.getcwd()
        else:
            self.verl_pwd = os.path.abspath(os.path.expanduser(self.verl_pwd))
            
        if not os.path.isdir(self.verl_pwd):
            raise ValueError(f"VERL_PWD directory does not exist: {self.verl_pwd}")
        
        logger.info(f"Initialized FileEditTool with base directory: {self.verl_pwd}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Get the OpenAI tool schema for this file editing tool."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="edit_file",
                description=f"Edit a file by replacing old_string with new_string. Replaces the first occurrence only.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "filepath": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Relative path to the file (e.g., 'verl/tools/base_tool.py')",
                            enum=None,
                        ),
                        "old_string": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The exact string to replace",
                            enum=None,
                        ),
                        "new_string": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The new string to insert",
                            enum=None,
                        )
                    },
                    required=["filepath", "old_string", "new_string"],
                ),
                strict=False,
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """Create a tool instance for file editing."""
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        
        if instance_id not in self._instance_dict:
            self._instance_dict[instance_id] = {
                "edits_made": 0
            }
        
        response_text = f"File editor ready. Directory: {self.verl_pwd}"
        logger.debug(f"Created file editor instance: {instance_id}")
        return instance_id, ToolResponse(text=response_text)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, Dict[str, Any]]:
        """Replace old_string with new_string in the file."""
        filepath = parameters.get("filepath", "").strip()
        old_string = parameters.get("old_string", "")
        new_string = parameters.get("new_string", "")
        
        if not filepath:
            return ToolResponse(text="Error: filepath is required"), 0.0, {"error": "missing_filepath"}
        
        if not old_string:
            return ToolResponse(text="Error: old_string is required"), 0.0, {"error": "missing_old_string"}
        
        try:
            # Get safe path
            full_path = self._get_safe_path(filepath)
            if full_path is None:
                return ToolResponse(text=f"Error: Invalid path: {filepath}"), 0.0, {"error": "invalid_path"}
            
            if not os.path.isfile(full_path):
                return ToolResponse(text=f"Error: File not found: {filepath}"), 0.0, {"error": "file_not_found"}
            
            # Check file size
            if os.path.getsize(full_path) > self.max_file_size:
                return ToolResponse(text=f"Error: File too large"), 0.0, {"error": "file_too_large"}
            
            # Read file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if old_string exists
            if old_string not in content:
                return ToolResponse(text=f"Error: String not found in file"), 0.0, {"error": "string_not_found"}
            
            # Replace first occurrence
            new_content = content.replace(old_string, new_string, 1)
            
            # Create backup
            if self.create_backup:
                with open(str(full_path) + ".bak", 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Write new content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Update stats
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["edits_made"] += 1
            
            response_text = f"Successfully edited {filepath}\nReplaced 1 occurrence"
            if self.create_backup:
                response_text += f"\nBackup: {filepath}.bak"
            
            return ToolResponse(text=response_text), 1.0, {"filepath": filepath}
            
        except Exception as e:
            error_msg = f"Error editing file: {str(e)}"
            logger.error(error_msg)
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    def _get_safe_path(self, filepath: str) -> Optional[Path]:
        """Convert relative path to safe absolute path within VERL_PWD."""
        try:
            filepath = filepath.lstrip('/')
            full_path = Path(self.verl_pwd) / filepath
            resolved_path = full_path.resolve()
            
            # Security check
            verl_pwd_resolved = Path(self.verl_pwd).resolve()
            try:
                resolved_path.relative_to(verl_pwd_resolved)
            except ValueError:
                logger.warning(f"Path outside VERL_PWD: {filepath}")
                return None
            
            return resolved_path
        except Exception as e:
            logger.error(f"Error constructing path: {e}")
            return None

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward based on successful edits."""
        if instance_id not in self._instance_dict:
            return 0.0
        return float(self._instance_dict[instance_id]["edits_made"])

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]