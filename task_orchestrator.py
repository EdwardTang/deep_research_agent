"""
Task orchestration functionality for managing complex tasks and their dependencies.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import openai

logger = logging.getLogger(__name__)

@dataclass
class SubTask:
    """Represents a sub-task in the task decomposition."""
    id: str
    description: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    assigned_agent: Optional[str] = None
    parent_task: Optional[str] = None
    dependencies: List[str] = None  # List of sub-task IDs this task depends on
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = None
    updated_at: float = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = time.time()
        self.updated_at = self.created_at

    def update_status(self, status: str, result: Any = None, error: str = None):
        """Update the status and optionally set result or error."""
        self.status = status
        if result is not None:
            self.result = result
        if error is not None:
            self.error = error
        self.updated_at = time.time()

class TaskOrchestrator:
    """Handles task decomposition, assignment and state management."""
    
    def __init__(self, model: str = "o3-mini"):
        self.model = model
        self.client = openai.OpenAI()
        self.tasks: Dict[str, SubTask] = {}
        self.current_task_id: Optional[str] = None
        
    def decompose_task(self, task_description: str) -> List[SubTask]:
        """
        Decompose a complex task into smaller sub-tasks using LLM.
        
        Args:
            task_description: The main task to decompose
            
        Returns:
            List of created sub-tasks
        """
        # Prepare prompt for task decomposition
        example_json = '''{
  "subtasks": [
    {
      "description": "Search web for latest papers on topic X",
      "dependencies": [],
      "agent": "web_search"
    },
    {
      "description": "Extract key findings from papers",
      "dependencies": ["Search web for latest papers on topic X"],
      "agent": "text_inspector"
    }
  ]
}'''

        prompt = f"""Please help me break down this task into smaller, manageable sub-tasks:

{task_description}

For each sub-task, provide:
1. A clear description of what needs to be done
2. Any dependencies on other sub-tasks (if applicable)
3. The type of agent best suited for this sub-task (e.g., text inspector, VQA, web search)

Format your response as a JSON object with a 'subtasks' field containing an array of objects with these fields:
- description: sub-task description
- dependencies: list of dependency descriptions (empty list if none)
- agent: suggested agent type

Example:
{example_json}"""

        # Get LLM's decomposition suggestion
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a task decomposition expert."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            # Parse the response
            response_data = json.loads(response.choices[0].message.content)
            subtasks_data = response_data.get("subtasks", [])
            
            # Create SubTask objects
            created_tasks = []
            task_map = {}  # Map descriptions to IDs for dependency linking
            
            # First pass: Create tasks without dependencies
            for task_data in subtasks_data:
                task_id = str(uuid.uuid4())
                task = SubTask(
                    id=task_id,
                    description=task_data["description"],
                    status="pending",
                    assigned_agent=task_data["agent"],
                    dependencies=[]
                )
                self.tasks[task_id] = task
                created_tasks.append(task)
                task_map[task_data["description"]] = task_id
            
            # Second pass: Link dependencies
            for task_data, task in zip(subtasks_data, created_tasks):
                dep_ids = [task_map[dep] for dep in task_data["dependencies"]]
                task.dependencies = dep_ids
            
            return created_tasks
            
        except Exception as e:
            logger.error(f"Error decomposing task: {str(e)}")
            return []
    
    def get_next_task(self) -> Optional[SubTask]:
        """Get the next task that is ready to be executed."""
        for task in self.tasks.values():
            if task.status != "pending":
                continue
                
            # Check if all dependencies are completed
            deps_completed = all(
                self.tasks[dep_id].status == "completed"
                for dep_id in task.dependencies
            )
            
            if deps_completed:
                return task
        return None
    
    def update_task_status(self, task_id: str, status: str, result: Any = None, error: str = None):
        """Update the status of a task."""
        if task_id in self.tasks:
            self.tasks[task_id].update_status(status, result, error)
        else:
            logger.error(f"Task {task_id} not found")
    
    def get_task_context(self, task_id: str) -> Dict[str, Any]:
        """Get the context for a task, including results from dependencies."""
        if task_id not in self.tasks:
            return {}
            
        task = self.tasks[task_id]
        context = {
            "task_description": task.description,
            "dependencies": {}
        }
        
        # Add results from completed dependencies
        for dep_id in task.dependencies:
            dep_task = self.tasks[dep_id]
            if dep_task.status == "completed":
                context["dependencies"][dep_task.description] = dep_task.result
                
        return context
    
    def serialize_state(self) -> str:
        """Serialize the current task state to JSON for the scratchpad."""
        state = {
            "current_task": self.current_task_id,
            "tasks": {
                task_id: {
                    "description": task.description,
                    "status": task.status,
                    "agent": task.assigned_agent,
                    "dependencies": task.dependencies,
                    "result": task.result,
                    "error": task.error,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at
                }
                for task_id, task in self.tasks.items()
            }
        }
        return json.dumps(state, indent=2)
    
    def load_state(self, state_json: str):
        """Load task state from JSON (e.g., from scratchpad)."""
        try:
            state = json.loads(state_json)
            self.current_task_id = state["current_task"]
            
            # Recreate tasks
            self.tasks = {}
            for task_id, task_data in state["tasks"].items():
                self.tasks[task_id] = SubTask(
                    id=task_id,
                    description=task_data["description"],
                    status=task_data["status"],
                    assigned_agent=task_data["agent"],
                    dependencies=task_data["dependencies"],
                    result=task_data["result"],
                    error=task_data["error"],
                    created_at=task_data["created_at"],
                    updated_at=task_data["updated_at"]
                )
                
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}") 