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

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class TodoItem:
    """Represents a single todo item."""
    
    def __init__(self, id: str, text: str, completed: bool = False, priority: str = "medium", created_at: str = None):
        self.id = id
        self.text = text
        self.completed = completed
        self.priority = priority  # low, medium, high
        self.created_at = created_at or datetime.now().isoformat()
        self.completed_at = None
    
    def mark_completed(self):
        self.completed = True
        self.completed_at = datetime.now().isoformat()
    
    def mark_incomplete(self):
        self.completed = False
        self.completed_at = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "completed": self.completed,
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }
    
    def __str__(self):
        status = "✅" if self.completed else "⬜"
        priority_emoji = {"low": "🔹", "medium": "🔸", "high": "🔴"}
        return f"{status} {priority_emoji.get(self.priority, '🔸')} {self.text}"


class TodoListTool(BaseTool):
    """A tool for managing todo lists with add, complete, list, and remove functionality."""

    def __init__(self, config: Dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.max_items = config.get("max_items", 100)
        self.persistent = config.get("persistent", False)  # Future: could save to file
        
        logger.info(f"Initialized TodoListTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Get the OpenAI tool schema for the todo list tool."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="todo_manager",
                description="Manage a todo list with the ability to add items, mark them as complete/incomplete, list all items, and remove items.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "action": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Action to perform on the todo list.",
                            enum=["add", "complete", "incomplete", "list", "remove", "clear", "search"]
                        ),
                        "item_text": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Text of the todo item (required for 'add' action).",
                            enum=None
                        ),
                        "item_id": OpenAIFunctionPropertySchema(
                            type="string",
                            description="ID of the todo item (required for 'complete', 'incomplete', 'remove' actions).",
                            enum=None
                        ),
                        "priority": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Priority level for the todo item (optional, defaults to 'medium').",
                            enum=["low", "medium", "high"]
                        ),
                        "search_query": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Search query to find todo items (required for 'search' action).",
                            enum=None
                        )
                    },
                    required=["action"],
                ),
                strict=False,
            )
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        """Create a todo list instance."""
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        
        if instance_id not in self._instance_dict:
            self._instance_dict[instance_id] = {
                "todos": {},  # Dict[str, TodoItem]
                "next_id": 1,
                "total_added": 0,
                "total_completed": 0
            }
        
        logger.debug(f"Created todo list instance: {instance_id}")
        return instance_id, ToolResponse(text="📝 Todo list created and ready to use!")

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, Dict[str, Any]]:
        """Execute todo list operations."""
        action = parameters.get("action", "")
        
        if instance_id not in self._instance_dict:
            return ToolResponse(text="Error: Todo list instance not found. Please create it first."), 0.0, {"error": "instance_not_found"}
        
        todo_list = self._instance_dict[instance_id]
        
        try:
            if action == "add":
                return await self._add_item(todo_list, parameters)
            elif action == "complete":
                return await self._complete_item(todo_list, parameters)
            elif action == "incomplete":
                return await self._incomplete_item(todo_list, parameters)
            elif action == "list":
                return await self._list_items(todo_list, parameters)
            elif action == "remove":
                return await self._remove_item(todo_list, parameters)
            elif action == "clear":
                return await self._clear_list(todo_list, parameters)
            elif action == "search":
                return await self._search_items(todo_list, parameters)
            else:
                return ToolResponse(text=f"Error: Unknown action '{action}'. Available actions: add, complete, incomplete, list, remove, clear, search"), 0.0, {"error": "unknown_action"}
                
        except Exception as e:
            error_msg = f"Error executing todo action '{action}': {str(e)}"
            logger.error(error_msg)
            return ToolResponse(text=error_msg), 0.0, {"error": str(e)}

    async def _add_item(self, todo_list: Dict, parameters: Dict[str, Any]) -> Tuple[ToolResponse, float, Dict]:
        """Add a new todo item."""
        item_text = parameters.get("item_text", "").strip()
        priority = parameters.get("priority", "medium")
        
        if not item_text:
            return ToolResponse(text="Error: item_text is required for adding a todo item."), 0.0, {"error": "missing_item_text"}
        
        if len(todo_list["todos"]) >= self.max_items:
            return ToolResponse(text=f"Error: Maximum number of todo items ({self.max_items}) reached."), 0.0, {"error": "max_items_reached"}
        
        # Create new todo item
        item_id = str(todo_list["next_id"])
        todo_item = TodoItem(id=item_id, text=item_text, priority=priority)
        
        # Add to list
        todo_list["todos"][item_id] = todo_item
        todo_list["next_id"] += 1
        todo_list["total_added"] += 1
        
        response_text = f"✅ Added todo item #{item_id}: '{item_text}' (Priority: {priority})"
        
        return ToolResponse(text=response_text), 1.0, {
            "action": "add",
            "item_id": item_id,
            "item_text": item_text,
            "priority": priority
        }

    async def _complete_item(self, todo_list: Dict, parameters: Dict[str, Any]) -> Tuple[ToolResponse, float, Dict]:
        """Mark a todo item as completed."""
        item_id = parameters.get("item_id", "").strip()
        
        if not item_id:
            return ToolResponse(text="Error: item_id is required for completing a todo item."), 0.0, {"error": "missing_item_id"}
        
        if item_id not in todo_list["todos"]:
            return ToolResponse(text=f"Error: Todo item #{item_id} not found."), 0.0, {"error": "item_not_found"}
        
        todo_item = todo_list["todos"][item_id]
        
        if todo_item.completed:
            return ToolResponse(text=f"ℹ️ Todo item #{item_id} is already completed."), 0.5, {"action": "complete", "already_completed": True}
        
        todo_item.mark_completed()
        todo_list["total_completed"] += 1
        
        response_text = f"🎉 Completed todo item #{item_id}: '{todo_item.text}'"
        
        return ToolResponse(text=response_text), 1.0, {
            "action": "complete",
            "item_id": item_id,
            "item_text": todo_item.text
        }

    async def _incomplete_item(self, todo_list: Dict, parameters: Dict[str, Any]) -> Tuple[ToolResponse, float, Dict]:
        """Mark a todo item as incomplete."""
        item_id = parameters.get("item_id", "").strip()
        
        if not item_id:
            return ToolResponse(text="Error: item_id is required for marking a todo item as incomplete."), 0.0, {"error": "missing_item_id"}
        
        if item_id not in todo_list["todos"]:
            return ToolResponse(text=f"Error: Todo item #{item_id} not found."), 0.0, {"error": "item_not_found"}
        
        todo_item = todo_list["todos"][item_id]
        
        if not todo_item.completed:
            return ToolResponse(text=f"ℹ️ Todo item #{item_id} is already incomplete."), 0.5, {"action": "incomplete", "already_incomplete": True}
        
        todo_item.mark_incomplete()
        todo_list["total_completed"] -= 1
        
        response_text = f"↩️ Marked todo item #{item_id} as incomplete: '{todo_item.text}'"
        
        return ToolResponse(text=response_text), 1.0, {
            "action": "incomplete",
            "item_id": item_id,
            "item_text": todo_item.text
        }

    async def _list_items(self, todo_list: Dict, parameters: Dict[str, Any]) -> Tuple[ToolResponse, float, Dict]:
        """List all todo items."""
        todos = todo_list["todos"]
        
        if not todos:
            return ToolResponse(text="📝 Your todo list is empty! Use 'add' action to create new items."), 1.0, {"action": "list", "total_items": 0}
        
        # Separate completed and incomplete items
        incomplete_items = []
        completed_items = []
        
        for todo_item in todos.values():
            if todo_item.completed:
                completed_items.append(todo_item)
            else:
                incomplete_items.append(todo_item)
        
        # Sort by priority (high first) then by creation time
        priority_order = {"high": 0, "medium": 1, "low": 2}
        incomplete_items.sort(key=lambda x: (priority_order.get(x.priority, 1), x.created_at))
        completed_items.sort(key=lambda x: x.completed_at or x.created_at, reverse=True)
        
        response_lines = ["📋 **Todo List**", ""]
        
        if incomplete_items:
            response_lines.append("**🔴 Pending Items:**")
            for item in incomplete_items:
                response_lines.append(f"  #{item.id} {str(item)}")
            response_lines.append("")
        
        if completed_items:
            response_lines.append("**✅ Completed Items:**")
            for item in completed_items[:10]:  # Show last 10 completed
                response_lines.append(f"  #{item.id} {str(item)}")
            if len(completed_items) > 10:
                response_lines.append(f"  ... and {len(completed_items) - 10} more completed items")
            response_lines.append("")
        
        # Summary
        total_items = len(todos)
        completion_rate = len(completed_items) / total_items * 100 if total_items > 0 else 0
        response_lines.append(f"📊 **Summary:** {len(incomplete_items)} pending, {len(completed_items)} completed ({completion_rate:.1f}% completion rate)")
        
        response_text = "\n".join(response_lines)
        
        return ToolResponse(text=response_text), 1.0, {
            "action": "list",
            "total_items": total_items,
            "pending_items": len(incomplete_items),
            "completed_items": len(completed_items),
            "completion_rate": completion_rate
        }

    async def _remove_item(self, todo_list: Dict, parameters: Dict[str, Any]) -> Tuple[ToolResponse, float, Dict]:
        """Remove a todo item."""
        item_id = parameters.get("item_id", "").strip()
        
        if not item_id:
            return ToolResponse(text="Error: item_id is required for removing a todo item."), 0.0, {"error": "missing_item_id"}
        
        if item_id not in todo_list["todos"]:
            return ToolResponse(text=f"Error: Todo item #{item_id} not found."), 0.0, {"error": "item_not_found"}
        
        todo_item = todo_list["todos"][item_id]
        item_text = todo_item.text
        was_completed = todo_item.completed
        
        # Remove the item
        del todo_list["todos"][item_id]
        if was_completed:
            todo_list["total_completed"] -= 1
        
        response_text = f"🗑️ Removed todo item #{item_id}: '{item_text}'"
        
        return ToolResponse(text=response_text), 1.0, {
            "action": "remove",
            "item_id": item_id,
            "item_text": item_text,
            "was_completed": was_completed
        }

    async def _clear_list(self, todo_list: Dict, parameters: Dict[str, Any]) -> Tuple[ToolResponse, float, Dict]:
        """Clear all todo items."""
        total_items = len(todo_list["todos"])
        completed_items = len([item for item in todo_list["todos"].values() if item.completed])
        
        todo_list["todos"].clear()
        todo_list["total_completed"] = 0
        
        response_text = f"🧹 Cleared all todo items ({total_items} items removed, {completed_items} were completed)"
        
        return ToolResponse(text=response_text), 1.0, {
            "action": "clear",
            "items_removed": total_items,
            "completed_items_removed": completed_items
        }

    async def _search_items(self, todo_list: Dict, parameters: Dict[str, Any]) -> Tuple[ToolResponse, float, Dict]:
        """Search for todo items."""
        search_query = parameters.get("search_query", "").strip().lower()
        
        if not search_query:
            return ToolResponse(text="Error: search_query is required for searching todo items."), 0.0, {"error": "missing_search_query"}
        
        todos = todo_list["todos"]
        matching_items = []
        
        for todo_item in todos.values():
            if search_query in todo_item.text.lower():
                matching_items.append(todo_item)
        
        if not matching_items:
            return ToolResponse(text=f"🔍 No todo items found matching '{search_query}'"), 1.0, {"action": "search", "matches": 0}
        
        response_lines = [f"🔍 **Search Results for '{search_query}':**", ""]
        
        for item in matching_items:
            response_lines.append(f"  #{item.id} {str(item)}")
        
        response_lines.append(f"\n📊 Found {len(matching_items)} matching item(s)")
        response_text = "\n".join(response_lines)
        
        return ToolResponse(text=response_text), 1.0, {
            "action": "search",
            "search_query": search_query,
            "matches": len(matching_items),
            "matching_ids": [item.id for item in matching_items]
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward based on todo list productivity."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        todo_list = self._instance_dict[instance_id]
        total_added = todo_list.get("total_added", 0)
        total_completed = todo_list.get("total_completed", 0)
        
        if total_added == 0:
            return 0.0
        
        # Reward based on completion rate
        completion_rate = total_completed / total_added
        return completion_rate

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the todo list instance."""
        if instance_id in self._instance_dict:
            todo_list = self._instance_dict[instance_id]
            total_items = len(todo_list["todos"])
            completed_items = len([item for item in todo_list["todos"].values() if item.completed])
            
            logger.debug(f"Releasing todo list instance {instance_id} with {total_items} total items ({completed_items} completed)")
            del self._instance_dict[instance_id]