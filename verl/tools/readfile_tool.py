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


class FileReaderTool(BaseTool):
    """A tool for reading files from the VERL project directory using relative paths."""

    def __init__(self, config: Dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # Configuration parameters
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10MB default
        self.max_lines = config.get("max_lines", 10000)  # Default max lines
        self.allowed_extensions = config.get("allowed_extensions", [])  # Empty = allow all
        
        # Get working directory from VERL_PWD environment variable
        self.verl_pwd = os.environ.get("VERL_PWD")
        if self.verl_pwd is None:
            logger.warning("VERL_PWD environment variable not set, using current directory")
            self.verl_pwd = os.getcwd()
        else:
            # Expand ~ if present and make absolute path
            self.verl_pwd = os.path.abspath(os.path.expanduser(self.verl_pwd))
            
        # Validate that the directory exists
        if not os.path.isdir(self.verl_pwd):
            logger.error(f"VERL_PWD directory does not exist: {self.verl_pwd}")
            raise ValueError(f"VERL_PWD directory does not exist: {self.verl_pwd}")
        
        logger.info(f"Initialized VerlFileReaderTool with base directory: {self.verl_pwd}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Get the OpenAI tool schema for this file reader tool."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="verl_read_file",
                description=f"Read files from the VERL project directory ({self.verl_pwd}) using relative paths. Can read specific number of lines from the beginning of the file.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "filepath": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Relative path to the file from VERL_PWD (e.g., 'verl/tools/base_tool.py', 'README.md')",
                            enum=None,
                        ),
                        "num_lines": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Number of lines to read from the beginning of the file (optional, defaults to entire file)",
                            enum=None,
                        )
                    },
                    required=["filepath"],
                ),
                strict=False,
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """Create a tool instance for file reading."""
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        
        if instance_id not in self._instance_dict:
            self._instance_dict[instance_id] = {
                "files_read": [],
                "total_lines_read": 0,
                "successful_reads": 0
            }
        
        response_text = f"VERL file reader ready. Base directory: {self.verl_pwd}"
        logger.debug(f"Created VERL file reader instance: {instance_id}")
        return instance_id, ToolResponse(text=response_text)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, Dict[str, Any]]:
        """Read file content from the specified relative path."""
        filepath = parameters.get("filepath", "").strip()
        num_lines = parameters.get("num_lines", None)
        
        if not filepath:
            return ToolResponse(text="Error: filepath parameter is required"), 0.0, {"error": "missing_filepath"}
        
        try:
            # Construct absolute path from relative path
            full_path = self._get_safe_path(filepath)
            if full_path is None:
                return ToolResponse(text=f"Error: Invalid or unsafe file path: {filepath}"), 0.0, {"error": "invalid_path"}
            
            # Check if file exists
            if not os.path.isfile(full_path):
                return ToolResponse(text=f"Error: File not found: {filepath}"), 0.0, {"error": "file_not_found"}
            
            # Check file extension if restrictions are configured
            if self.allowed_extensions:
                file_ext = os.path.splitext(full_path)[1].lower()
                if file_ext not in self.allowed_extensions:
                    return ToolResponse(text=f"Error: File type not allowed. Allowed extensions: {self.allowed_extensions}"), 0.0, {"error": "forbidden_extension"}
            
            # Check file size
            file_size = os.path.getsize(full_path)
            if file_size > self.max_file_size:
                return ToolResponse(text=f"Error: File too large ({file_size} bytes). Maximum allowed: {self.max_file_size} bytes"), 0.0, {"error": "file_too_large"}
            
            # Read file content
            content, lines_read, truncated = self._read_file_content(full_path, num_lines)
            
            # Update instance statistics
            if instance_id in self._instance_dict:
                instance = self._instance_dict[instance_id]
                instance["files_read"].append(filepath)
                instance["total_lines_read"] += lines_read
                instance["successful_reads"] += 1
            
            # Format response
            response_lines = [f"File: {filepath}", f"Lines read: {lines_read}"]
            if truncated:
                response_lines.append("(Content truncated)")
            response_lines.extend(["", "Content:", "```"])
            response_lines.append(content)
            response_lines.append("```")
            
            response_text = "\n".join(response_lines)
            
            metadata = {
                "filepath": filepath,
                "full_path": str(full_path),
                "lines_read": lines_read,
                "file_size": file_size,
                "truncated": truncated
            }
            
            return ToolResponse(text=response_text), 1.0, metadata
            
        except PermissionError:
            error_msg = f"Error: Permission denied reading file: {filepath}"
            return ToolResponse(text=error_msg), 0.0, {"error": "permission_denied"}
        except UnicodeDecodeError:
            error_msg = f"Error: Unable to decode file (not a text file?): {filepath}"
            return ToolResponse(text=error_msg), 0.0, {"error": "decode_error"}
        except Exception as e:
            error_msg = f"Error reading file {filepath}: {str(e)}"
            logger.error(error_msg)
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    def _get_safe_path(self, filepath: str) -> Optional[Path]:
        """Convert relative path to safe absolute path within VERL_PWD."""
        try:
            # Remove leading slashes to ensure it's treated as relative
            filepath = filepath.lstrip('/')
            
            # Create path relative to VERL_PWD
            full_path = Path(self.verl_pwd) / filepath
            
            # Resolve to handle any .. or . in the path
            resolved_path = full_path.resolve()
            
            # Security check: ensure the resolved path is still within VERL_PWD
            verl_pwd_resolved = Path(self.verl_pwd).resolve()
            try:
                resolved_path.relative_to(verl_pwd_resolved)
            except ValueError:
                # Path is outside of VERL_PWD
                logger.warning(f"Attempted to access file outside VERL_PWD: {filepath}")
                return None
            
            return resolved_path
            
        except Exception as e:
            logger.error(f"Error constructing safe path for {filepath}: {e}")
            return None

    def _read_file_content(self, full_path: Path, num_lines: Optional[int]) -> Tuple[str, int, bool]:
        """Read file content with optional line limit."""
        lines = []
        lines_read = 0
        truncated = False
        
        # Determine effective line limit
        effective_limit = min(num_lines or self.max_lines, self.max_lines)
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if lines_read >= effective_limit:
                        truncated = True
                        break
                    lines.append(line.rstrip('\n\r'))
                    lines_read += 1
                    
            content = '\n'.join(lines)
            
            # If num_lines was specified and we hit that limit, note if there were more lines
            if num_lines and lines_read == num_lines:
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        total_lines = sum(1 for _ in f)
                        if total_lines > num_lines:
                            truncated = True
                except:
                    pass  # If we can't count total lines, just continue
            
            return content, lines_read, truncated
            
        except Exception as e:
            raise e

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward based on successful file reads."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance = self._instance_dict[instance_id]
        return float(instance["successful_reads"])

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources associated with an instance."""
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            files_read = len(instance["files_read"])
            total_lines = instance["total_lines_read"]
            
            logger.debug(f"Releasing VERL file reader instance {instance_id}: "
                        f"{files_read} files read, {total_lines} total lines")
            del self._instance_dict[instance_id]