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

import asyncio
import logging
import os
import subprocess
import uuid
from typing import Any, Dict, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class BashTool(BaseTool):
    """A tool for executing bash commands using VERL_PWD environment variable as working directory."""

    def __init__(self, config: Dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # Configuration parameters
        self.default_timeout = config.get("timeout", 30)
        self.max_output_length = config.get("max_output_length", 10000)
        self.allowed_commands = config.get("allowed_commands", [])
        self.shell = config.get("shell", "/bin/bash")
        
        # Get working directory from VERL_PWD environment variable
        os.environ["VERL_PWD"] = "/workspace/hyperswitch"
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
        
        logger.info(f"Initialized VerlBashTool with working directory: {self.verl_pwd}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Get the OpenAI tool schema for this bash tool."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="verl_bash",
                description=f"Execute bash commands",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "command": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The bash command to execute.",
                            enum=None,
                        ),
                        "timeout": OpenAIFunctionPropertySchema(
                            type="integer", 
                            description="Timeout for command execution in seconds (optional, defaults to 30).",
                            enum=None,
                        )
                    },
                    required=["command"],
                ),
                strict=False,
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """Create a tool instance for bash command execution."""
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        
        if instance_id not in self._instance_dict:
            self._instance_dict[instance_id] = {
                "commands_executed": [],
                "outputs": [],
                "working_dir": self.verl_pwd,
                "total_reward": 0.0,
                "successful_commands": 0
            }
        
        response_text = f"bash tool ready. Working directory: {self.verl_pwd}"
        logger.debug(f"Created bash tool instance: {instance_id}")
        return instance_id, ToolResponse(text=response_text)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, Dict[str, Any]]:
        """Execute bash command in VERL_PWD directory."""
        command = parameters.get("command", "")
        timeout = parameters.get("timeout", self.default_timeout)
        
        if not command:
            return ToolResponse(text="Error: No command provided"), 0.0, {"error": "missing_command"}
        
        # Security check: validate against allowed commands if configured
        if self.allowed_commands:
            command_valid = False
            for allowed_cmd in self.allowed_commands:
                if command.strip().startswith(allowed_cmd):
                    command_valid = True
                    break
            
            if not command_valid:
                error_msg = f"Error: Command not allowed. Allowed commands start with: {self.allowed_commands}"
                return ToolResponse(text=error_msg), 0.0, {"error": "forbidden_command"}
        
        try:
            # Execute the bash command
            result = await self._execute_bash_command(command, timeout)
            
            # Store execution history
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["commands_executed"].append(command)
                self._instance_dict[instance_id]["outputs"].append(result)
                
                # Calculate reward based on execution success
                if not result.startswith("Error:"):
                    reward = 1.0
                    self._instance_dict[instance_id]["successful_commands"] += 1
                else:
                    reward = 0.0
                
                self._instance_dict[instance_id]["total_reward"] += reward
            else:
                reward = 1.0 if not result.startswith("Error:") else 0.0
            
            metadata = {
                "command": command,
                "timeout": timeout,
                "working_directory": self.verl_pwd,
                "success": reward > 0,
                "output_length": len(result)
            }
            
            return ToolResponse(text=result), reward, metadata
            
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            logger.error(error_msg)
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    async def _execute_bash_command(self, command: str, timeout: int) -> str:
        """Internal method to execute a bash command in VERL_PWD directory."""
        try:
            # Create subprocess with VERL_PWD as working directory
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.verl_pwd,
                executable=self.shell
            )
            
            try:
                # Wait for process completion with timeout
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                
                # Decode output
                stdout_str = stdout.decode('utf-8', errors='replace').strip()
                stderr_str = stderr.decode('utf-8', errors='replace').strip()
                
                # Format output based on return code
                if process.returncode == 0:
                    if stdout_str:
                        output = stdout_str
                    else:
                        output = f"Command executed successfully in {self.verl_pwd}"
                else:
                    if stderr_str:
                        output = f"Error (exit code {process.returncode}): {stderr_str}"
                    else:
                        output = f"Command failed with exit code {process.returncode}"
                    
                    # Include stdout if available for failed commands
                    if stdout_str:
                        output += f"\nStdout: {stdout_str}"
                
                # Add working directory info for context
                if len(output) < 500:  # Only add for shorter outputs to avoid clutter
                    output += f"\n[Executed in: {self.verl_pwd}]"
                
                # Truncate if output is too long
                if len(output) > self.max_output_length:
                    output = output[:self.max_output_length] + "\n... (output truncated)"
                
                return output
                
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                return f"Error: Command timed out after {timeout} seconds in {self.verl_pwd}"
                
        except Exception as e:
            return f"Error executing command in {self.verl_pwd}: {str(e)}"

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate cumulative reward for this instance."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance = self._instance_dict[instance_id]
        total_commands = len(instance["commands_executed"])
        successful_commands = instance["successful_commands"]
        
        if total_commands == 0:
            return 0.0
        
        # Return success rate as reward
        return successful_commands / total_commands

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up resources associated with an instance."""
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            total_commands = len(instance["commands_executed"])
            successful_commands = instance["successful_commands"]
            
            logger.debug(f"Releasing VERL bash tool instance {instance_id}: "
                        f"{successful_commands}/{total_commands} commands successful")
            del self._instance_dict[instance_id]