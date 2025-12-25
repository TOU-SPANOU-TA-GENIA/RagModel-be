# app/workflows/engine.py
"""
Workflow Execution Engine.
Orchestrates workflow execution, manages state, handles node routing.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from app.workflows.storage import workflow_storage, WorkflowStorage
from app.workflows.models import NodeType, ExecutionStatus
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class NodeResult:
    """Result from a node execution."""
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    next_nodes: List[str] = field(default_factory=list)  # For conditional routing
    should_continue: bool = True  # False to stop execution


@dataclass
class ExecutionContext:
    """Context passed between nodes during execution."""
    execution_id: str
    workflow_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)  # node_id -> output
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        self.data[key] = value
    
    def get_node_output(self, node_id: str) -> Optional[Dict]:
        return self.node_outputs.get(node_id)


# Type alias for node handler
NodeHandler = Callable[[Dict, ExecutionContext], Awaitable[NodeResult]]


class WorkflowEngine:
    """
    Core workflow execution engine.
    
    Responsibilities:
    - Execute workflows node by node
    - Manage execution state and context
    - Handle errors and retries
    - Support pause/resume
    """
    
    def __init__(self, storage: Optional[WorkflowStorage] = None):
        self.storage = storage or workflow_storage
        self._node_handlers: Dict[str, NodeHandler] = {}
        self._running_executions: Dict[str, asyncio.Task] = {}
        self._register_default_handlers()
    
    def register_handler(self, node_type: str, handler: NodeHandler):
        """Register a handler for a node type."""
        self._node_handlers[node_type] = handler
        logger.debug(f"Registered handler for node type: {node_type}")
    
    def _register_default_handlers(self):
        """Register built-in node handlers."""
        from app.workflows.handlers import (
            handle_file_watcher_trigger,
            handle_schedule_trigger,
            handle_manual_trigger,
            handle_extract_content,
            handle_llm_analysis,
            handle_anomaly_detection,
            handle_cross_reference,
            handle_summarize,
            handle_generate_report,
            handle_send_email,
            handle_save_to_folder,
            handle_condition,
            handle_delay
        )
        
        # Triggers
        self._node_handlers['trigger_file_watcher'] = handle_file_watcher_trigger
        self._node_handlers['trigger_schedule'] = handle_schedule_trigger
        self._node_handlers['trigger_manual'] = handle_manual_trigger
        
        # Processors
        self._node_handlers['processor_extract_content'] = handle_extract_content
        self._node_handlers['processor_llm_analysis'] = handle_llm_analysis
        self._node_handlers['processor_anomaly_detection'] = handle_anomaly_detection
        self._node_handlers['processor_cross_reference'] = handle_cross_reference
        self._node_handlers['processor_summarize'] = handle_summarize
        
        # Actions
        self._node_handlers['action_generate_report'] = handle_generate_report
        self._node_handlers['action_send_email'] = handle_send_email
        self._node_handlers['action_save_to_folder'] = handle_save_to_folder
        
        # Flow
        self._node_handlers['flow_condition'] = handle_condition
        self._node_handlers['flow_delay'] = handle_delay
        
        logger.info(f"📦 Registered {len(self._node_handlers)} workflow handlers")
    
    async def start_execution(
        self,
        workflow_id: str,
        trigger_type: str = "manual",
        trigger_data: Optional[Dict] = None,
        initial_context: Optional[Dict] = None
    ) -> str:
        """Start a new workflow execution."""
        # Get workflow
        workflow = self.storage.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        if not workflow['is_enabled'] and trigger_type != "manual":
            raise ValueError(f"Workflow is disabled: {workflow_id}")
        
        # Create execution record
        execution_id = self.storage.create_execution(
            workflow_id=workflow_id,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            initial_context=initial_context
        )
        
        # Start async execution
        task = asyncio.create_task(
            self._run_execution(execution_id, workflow, trigger_data, initial_context)
        )
        self._running_executions[execution_id] = task
        
        # Update workflow last_run_at
        self.storage.update_workflow(workflow_id)
        
        logger.info(f"▶️  Started execution {execution_id} for workflow {workflow['name']}")
        return execution_id
    
    async def _run_execution(
        self,
        execution_id: str,
        workflow: Dict,
        trigger_data: Optional[Dict],
        initial_context: Optional[Dict]
    ):
        """Run a workflow execution."""
        # Create context
        context = ExecutionContext(
            execution_id=execution_id,
            workflow_id=workflow['id'],
            data=initial_context or {},
            trigger_data=trigger_data or {}
        )
        
        # Update status to running
        self.storage.update_execution(execution_id, status='running')
        
        try:
            # Find start nodes (nodes with no incoming edges)
            start_nodes = self._find_start_nodes(workflow)
            if not start_nodes:
                raise ValueError("No start nodes found in workflow")
            
            # Build adjacency list for traversal
            adjacency = self._build_adjacency_list(workflow)
            
            # Execute nodes
            completed_nodes = []
            failed_nodes = []
            queue = list(start_nodes)
            visited = set()
            
            while queue:
                node_id = queue.pop(0)
                
                if node_id in visited:
                    continue
                
                # Get node definition
                node = self._get_node_by_id(workflow['nodes'], node_id)
                if not node:
                    logger.warning(f"Node not found: {node_id}")
                    continue
                
                # Check if all dependencies are satisfied
                incoming = self._get_incoming_nodes(workflow, node_id)
                if not all(dep in completed_nodes for dep in incoming):
                    # Re-queue this node
                    queue.append(node_id)
                    continue
                
                visited.add(node_id)
                
                # Update current node
                self.storage.update_execution(
                    execution_id,
                    current_node_id=node_id,
                    completed_nodes=completed_nodes
                )
                
                # Execute node
                result = await self._execute_node(node, context)
                
                if result.success:
                    completed_nodes.append(node_id)
                    context.node_outputs[node_id] = result.output
                    
                    # Determine next nodes
                    if result.next_nodes:
                        # Conditional routing - use specified nodes
                        queue.extend(result.next_nodes)
                    elif node['type'] == 'condition':
                        # Condition node with no next_nodes means no branch matched
                        # Don't fall back to all edges - just continue to merge point
                        logger.info(f"Condition node {node_id} - no branch taken")
                    else:
                        # Default: all outgoing edges
                        queue.extend(adjacency.get(node_id, []))
                    
                    if not result.should_continue:
                        logger.info(f"Node {node_id} requested execution stop")
                        break
                else:
                    failed_nodes.append(node_id)
                    logger.error(f"Node {node_id} failed: {result.error}")
                    
                    # For now, stop on failure
                    # Could add retry logic or continue on failure option
                    self.storage.update_execution(
                        execution_id,
                        status='failed',
                        error_message=result.error,
                        error_node_id=node_id,
                        completed_nodes=completed_nodes,
                        failed_nodes=failed_nodes,
                        context=context.data
                    )
                    return
            
            # Success
            self.storage.update_execution(
                execution_id,
                status='completed',
                completed_nodes=completed_nodes,
                context=context.data
            )
            logger.info(f"✅ Execution {execution_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Execution {execution_id} failed with exception")
            self.storage.update_execution(
                execution_id,
                status='failed',
                error_message=str(e)
            )
        finally:
            # Clean up
            self._running_executions.pop(execution_id, None)
    
    async def _execute_node(
        self,
        node: Dict,
        context: ExecutionContext
    ) -> NodeResult:
        """Execute a single node."""
        node_id = node['id']
        node_type = node['type']
        config = node.get('config', {})
        
        # Log start
        start_time = time.time()
        log_id = self.storage.add_node_log(
            execution_id=context.execution_id,
            node_id=node_id,
            status='running',
            input_data={'config': config, 'context_keys': list(context.data.keys())}
        )
        
        try:
            # Get handler
            handler = self._node_handlers.get(node_type)
            
            if handler:
                result = await handler(config, context)
            else:
                # No handler - use default passthrough
                logger.warning(f"No handler for node type: {node_type}, using passthrough")
                result = NodeResult(success=True, output={'passthrough': True})
            
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Update log
            self.storage.update_node_log(
                log_id,
                status='completed' if result.success else 'failed',
                output_data=result.output,
                error_message=result.error,
                duration_ms=duration_ms
            )
            
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            
            self.storage.update_node_log(
                log_id,
                status='failed',
                error_message=error_msg,
                duration_ms=duration_ms
            )
            
            return NodeResult(success=False, error=error_msg)
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution."""
        task = self._running_executions.get(execution_id)
        if task and not task.done():
            task.cancel()
            self.storage.update_execution(execution_id, status='paused')
            logger.info(f"⏸️  Execution {execution_id} paused")
            return True
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        execution = self.storage.get_execution(execution_id)
        if not execution or execution['status'] != 'paused':
            return False
        
        workflow = self.storage.get_workflow(execution['workflow_id'])
        if not workflow:
            return False
        
        # Resume from current state
        task = asyncio.create_task(
            self._resume_execution(execution_id, execution, workflow)
        )
        self._running_executions[execution_id] = task
        
        logger.info(f"▶️  Execution {execution_id} resumed")
        return True
    
    async def _resume_execution(
        self,
        execution_id: str,
        execution: Dict,
        workflow: Dict
    ):
        """Resume a paused execution from its current state."""
        # Recreate context
        context = ExecutionContext(
            execution_id=execution_id,
            workflow_id=workflow['id'],
            data=execution['context'],
            trigger_data=execution['trigger_data']
        )
        
        # Get completed nodes
        completed_nodes = execution['completed_nodes']
        
        # Update status
        self.storage.update_execution(execution_id, status='running')
        
        try:
            # Build adjacency
            adjacency = self._build_adjacency_list(workflow)
            
            # Find next nodes to execute (outgoing from completed nodes not yet visited)
            all_next = set()
            for node_id in completed_nodes:
                all_next.update(adjacency.get(node_id, []))
            
            queue = [n for n in all_next if n not in completed_nodes]
            
            # Continue execution
            failed_nodes = execution['failed_nodes']
            visited = set(completed_nodes)
            
            while queue:
                node_id = queue.pop(0)
                
                if node_id in visited:
                    continue
                
                node = self._get_node_by_id(workflow['nodes'], node_id)
                if not node:
                    continue
                
                # Check dependencies
                incoming = self._get_incoming_nodes(workflow, node_id)
                if not all(dep in completed_nodes for dep in incoming):
                    queue.append(node_id)
                    continue
                
                visited.add(node_id)
                
                self.storage.update_execution(
                    execution_id,
                    current_node_id=node_id,
                    completed_nodes=completed_nodes
                )
                
                result = await self._execute_node(node, context)
                
                if result.success:
                    completed_nodes.append(node_id)
                    context.node_outputs[node_id] = result.output
                    
                    if result.next_nodes:
                        queue.extend(result.next_nodes)
                    else:
                        queue.extend(adjacency.get(node_id, []))
                    
                    if not result.should_continue:
                        break
                else:
                    failed_nodes.append(node_id)
                    self.storage.update_execution(
                        execution_id,
                        status='failed',
                        error_message=result.error,
                        error_node_id=node_id,
                        completed_nodes=completed_nodes,
                        failed_nodes=failed_nodes,
                        context=context.data
                    )
                    return
            
            # Success
            self.storage.update_execution(
                execution_id,
                status='completed',
                completed_nodes=completed_nodes,
                context=context.data
            )
            
        except Exception as e:
            logger.exception(f"Resumed execution {execution_id} failed")
            self.storage.update_execution(
                execution_id,
                status='failed',
                error_message=str(e)
            )
        finally:
            self._running_executions.pop(execution_id, None)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running or paused execution."""
        task = self._running_executions.get(execution_id)
        if task and not task.done():
            task.cancel()
        
        self.storage.update_execution(execution_id, status='cancelled')
        self._running_executions.pop(execution_id, None)
        
        logger.info(f"🛑 Execution {execution_id} cancelled")
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get current status of an execution."""
        return self.storage.get_execution(execution_id)
    
    # =========================================================================
    # Graph Utilities
    # =========================================================================
    
    def _find_start_nodes(self, workflow: Dict) -> List[str]:
        """Find nodes with no incoming edges (start nodes)."""
        nodes = workflow['nodes']
        edges = workflow['edges']
        
        all_node_ids = {n['id'] for n in nodes}
        nodes_with_incoming = {e['target'] for e in edges}
        
        start_nodes = all_node_ids - nodes_with_incoming
        
        # Also consider trigger nodes as start nodes
        trigger_types = {
            NodeType.TRIGGER_FILE_WATCHER.value,
            NodeType.TRIGGER_SCHEDULE.value,
            NodeType.TRIGGER_DATA_CONDITION.value,
            NodeType.TRIGGER_MANUAL.value
        }
        
        for node in nodes:
            if node['type'] in trigger_types:
                start_nodes.add(node['id'])
        
        return list(start_nodes)
    
    def _build_adjacency_list(self, workflow: Dict) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adjacency = {}
        for edge in workflow['edges']:
            source = edge['source']
            target = edge['target']
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(target)
        return adjacency
    
    def _get_node_by_id(self, nodes: List[Dict], node_id: str) -> Optional[Dict]:
        """Get a node by its ID."""
        for node in nodes:
            if node['id'] == node_id:
                return node
        return None
    
    def _get_incoming_nodes(self, workflow: Dict, node_id: str) -> List[str]:
        """Get all nodes that have edges pointing to this node."""
        return [e['source'] for e in workflow['edges'] if e['target'] == node_id]


# Global engine instance
workflow_engine = WorkflowEngine()