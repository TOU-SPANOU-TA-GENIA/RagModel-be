# app/workflows/storage.py
"""
Storage layer for workflow system.
Handles CRUD operations for workflows, executions, baselines, and settings.
"""

import sqlite3
import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class WorkflowStorage:
    """
    Storage layer for workflow data.
    Uses SQLite with JSON columns for flexible schema.
    """
    
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = Path(db_path)
    
    @contextmanager
    def get_db(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # =========================================================================
    # Workflow CRUD
    # =========================================================================
    
    def create_workflow(
        self,
        name: str,
        user_id: Optional[int] = None,
        description: Optional[str] = None,
        nodes: List[Dict] = None,
        edges: List[Dict] = None,
        is_shared: bool = False
    ) -> str:
        """Create a new workflow."""
        workflow_id = str(uuid.uuid4())
        nodes = nodes or []
        edges = edges or []
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflows (id, user_id, name, description, nodes, edges, is_shared)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                user_id,
                name,
                description,
                json.dumps(nodes),
                json.dumps(edges),
                is_shared
            ))
            conn.commit()
        
        logger.info(f"✅ Workflow created: {workflow_id} - {name}")
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Get a workflow by ID."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_workflow(row)
            return None
    
    def list_workflows(
        self,
        user_id: Optional[int] = None,
        include_shared: bool = True,
        enabled_only: bool = False
    ) -> List[Dict]:
        """List workflows with optional filters."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if user_id is not None:
                if include_shared:
                    conditions.append("(user_id = ? OR is_shared = 1)")
                else:
                    conditions.append("user_id = ?")
                params.append(user_id)
            
            if enabled_only:
                conditions.append("is_enabled = 1")
            
            query = "SELECT * FROM workflows"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY updated_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_workflow(row) for row in rows]
    
    def update_workflow(
        self,
        workflow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        nodes: Optional[List[Dict]] = None,
        edges: Optional[List[Dict]] = None,
        is_enabled: Optional[bool] = None,
        is_shared: Optional[bool] = None,
        triggers: Optional[List[Dict]] = None
    ) -> bool:
        """Update a workflow."""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if nodes is not None:
            updates.append("nodes = ?")
            params.append(json.dumps(nodes))
        
        if edges is not None:
            updates.append("edges = ?")
            params.append(json.dumps(edges))
        
        if is_enabled is not None:
            updates.append("is_enabled = ?")
            params.append(is_enabled)
        
        if is_shared is not None:
            updates.append("is_shared = ?")
            params.append(is_shared)
        
        if triggers is not None:
            updates.append("triggers = ?")
            params.append(json.dumps(triggers))
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(workflow_id)
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE workflows SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow and all related data."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"🗑️  Workflow deleted: {workflow_id}")
        return deleted
    
    def set_workflow_enabled(self, workflow_id: str, enabled: bool) -> bool:
        """Enable or disable a workflow."""
        return self.update_workflow(workflow_id, is_enabled=enabled)
    
    # =========================================================================
    # Execution CRUD
    # =========================================================================
    
    def create_execution(
        self,
        workflow_id: str,
        trigger_type: str = "manual",
        trigger_data: Optional[Dict] = None,
        initial_context: Optional[Dict] = None
    ) -> str:
        """Create a new execution record."""
        execution_id = str(uuid.uuid4())
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflow_executions 
                (id, workflow_id, status, trigger_type, trigger_data, context)
                VALUES (?, ?, 'pending', ?, ?, ?)
            """, (
                execution_id,
                workflow_id,
                trigger_type,
                json.dumps(trigger_data or {}),
                json.dumps(initial_context or {})
            ))
            conn.commit()
        
        logger.info(f"▶️  Execution created: {execution_id}")
        return execution_id
    
    def get_execution(self, execution_id: str) -> Optional[Dict]:
        """Get an execution by ID."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM workflow_executions WHERE id = ?",
                (execution_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_execution(row)
            return None
    
    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List executions with optional filters."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if workflow_id:
                conditions.append("workflow_id = ?")
                params.append(workflow_id)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            query = "SELECT * FROM workflow_executions"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += f" ORDER BY started_at DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_execution(row) for row in rows]
    
    def update_execution(
        self,
        execution_id: str,
        status: Optional[str] = None,
        current_node_id: Optional[str] = None,
        completed_nodes: Optional[List[str]] = None,
        failed_nodes: Optional[List[str]] = None,
        context: Optional[Dict] = None,
        error_message: Optional[str] = None,
        error_node_id: Optional[str] = None
    ) -> bool:
        """Update an execution."""
        updates = []
        params = []
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status in ('completed', 'failed', 'cancelled'):
                updates.append("completed_at = CURRENT_TIMESTAMP")
        
        if current_node_id is not None:
            updates.append("current_node_id = ?")
            params.append(current_node_id)
        
        if completed_nodes is not None:
            updates.append("completed_nodes = ?")
            params.append(json.dumps(completed_nodes))
        
        if failed_nodes is not None:
            updates.append("failed_nodes = ?")
            params.append(json.dumps(failed_nodes))
        
        if context is not None:
            updates.append("context = ?")
            params.append(json.dumps(context))
        
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        
        if error_node_id is not None:
            updates.append("error_node_id = ?")
            params.append(error_node_id)
        
        if not updates:
            return False
        
        params.append(execution_id)
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE workflow_executions SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def get_resumable_executions(self) -> List[Dict]:
        """Get executions that can be resumed (paused or running before crash)."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM workflow_executions 
                WHERE status IN ('running', 'paused')
                ORDER BY started_at DESC
            """)
            rows = cursor.fetchall()
            return [self._row_to_execution(row) for row in rows]
    
    # =========================================================================
    # Node Logs
    # =========================================================================
    
    def add_node_log(
        self,
        execution_id: str,
        node_id: str,
        status: str = "pending",
        input_data: Optional[Dict] = None,
        output_data: Optional[Dict] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> int:
        """Add a node execution log."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflow_node_logs 
                (execution_id, node_id, status, input_data, output_data, 
                 error_message, duration_ms, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                execution_id,
                node_id,
                status,
                json.dumps(input_data) if input_data else None,
                json.dumps(output_data) if output_data else None,
                error_message,
                duration_ms
            ))
            conn.commit()
            return cursor.lastrowid
    
    def update_node_log(
        self,
        log_id: int,
        status: Optional[str] = None,
        output_data: Optional[Dict] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> bool:
        """Update a node log."""
        updates = []
        params = []
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status in ('completed', 'failed', 'skipped'):
                updates.append("completed_at = CURRENT_TIMESTAMP")
        
        if output_data is not None:
            updates.append("output_data = ?")
            params.append(json.dumps(output_data))
        
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        
        if duration_ms is not None:
            updates.append("duration_ms = ?")
            params.append(duration_ms)
        
        if not updates:
            return False
        
        params.append(log_id)
        
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE workflow_node_logs SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def get_node_logs(self, execution_id: str) -> List[Dict]:
        """Get all node logs for an execution."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM workflow_node_logs 
                WHERE execution_id = ?
                ORDER BY started_at ASC
            """, (execution_id,))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Baselines
    # =========================================================================
    
    def upsert_baseline(
        self,
        baseline_type: str,
        baseline_key: str,
        statistics: Dict,
        workflow_id: Optional[str] = None,
        anomaly_threshold: float = 2.0
    ) -> int:
        """Insert or update a baseline."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            # Try to update existing
            cursor.execute("""
                UPDATE workflow_baselines 
                SET statistics = ?, 
                    sample_count = sample_count + 1,
                    last_sample_at = CURRENT_TIMESTAMP,
                    anomaly_threshold = ?
                WHERE workflow_id IS ? AND baseline_type = ? AND baseline_key = ?
            """, (
                json.dumps(statistics),
                anomaly_threshold,
                workflow_id,
                baseline_type,
                baseline_key
            ))
            
            if cursor.rowcount == 0:
                # Insert new
                cursor.execute("""
                    INSERT INTO workflow_baselines 
                    (workflow_id, baseline_type, baseline_key, statistics, 
                     sample_count, first_sample_at, anomaly_threshold)
                    VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?)
                """, (
                    workflow_id,
                    baseline_type,
                    baseline_key,
                    json.dumps(statistics),
                    anomaly_threshold
                ))
            
            conn.commit()
            return cursor.lastrowid or 0
    
    def get_baseline(
        self,
        baseline_type: str,
        baseline_key: str,
        workflow_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Get a baseline by type and key."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM workflow_baselines 
                WHERE workflow_id IS ? AND baseline_type = ? AND baseline_key = ?
            """, (workflow_id, baseline_type, baseline_key))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                result['statistics'] = json.loads(result['statistics'])
                return result
            return None
    
    # =========================================================================
    # User Settings
    # =========================================================================
    
    def get_user_settings(self, user_id: int) -> Optional[Dict]:
        """Get user settings."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def upsert_user_settings(self, user_id: int, settings: Dict) -> bool:
        """Insert or update user settings."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            # Build dynamic update
            fields = []
            values = []
            for key, value in settings.items():
                if key != 'user_id':
                    fields.append(key)
                    values.append(value)
            
            if not fields:
                return False
            
            # Try update first
            set_clause = ", ".join(f"{f} = ?" for f in fields)
            cursor.execute(
                f"UPDATE user_settings SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                values + [user_id]
            )
            
            if cursor.rowcount == 0:
                # Insert new
                fields.append("user_id")
                values.append(user_id)
                placeholders = ", ".join("?" * len(fields))
                cursor.execute(
                    f"INSERT INTO user_settings ({', '.join(fields)}) VALUES ({placeholders})",
                    values
                )
            
            conn.commit()
            return True
    
    # =========================================================================
    # File Watch Registry
    # =========================================================================
    
    def register_file_watch(
        self,
        workflow_id: str,
        watch_path: str,
        watch_type: str = "folder",
        recursive: bool = True,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None
    ) -> int:
        """Register a file/folder watch."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO file_watch_registry 
                (workflow_id, watch_path, watch_type, recursive, include_patterns, exclude_patterns)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                watch_path,
                watch_type,
                recursive,
                json.dumps(include_patterns or ["*"]),
                json.dumps(exclude_patterns or [])
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_active_watches(self) -> List[Dict]:
        """Get all active file watches."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT fwr.*, w.is_enabled as workflow_enabled
                FROM file_watch_registry fwr
                JOIN workflows w ON w.id = fwr.workflow_id
                WHERE fwr.is_active = 1 AND w.is_enabled = 1
            """)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                r = dict(row)
                r['include_patterns'] = json.loads(r['include_patterns'])
                r['exclude_patterns'] = json.loads(r['exclude_patterns'])
                results.append(r)
            return results
    
    # =========================================================================
    # Scheduled Tasks
    # =========================================================================
    
    def create_scheduled_task(
        self,
        workflow_id: str,
        schedule_type: str,
        schedule_config: Dict,
        next_run_at: Optional[datetime] = None
    ) -> int:
        """Create a scheduled task."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scheduled_tasks 
                (workflow_id, schedule_type, schedule_config, next_run_at)
                VALUES (?, ?, ?, ?)
            """, (
                workflow_id,
                schedule_type,
                json.dumps(schedule_config),
                next_run_at.isoformat() if next_run_at else None
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_due_tasks(self) -> List[Dict]:
        """Get tasks that are due to run."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT st.*, w.is_enabled as workflow_enabled
                FROM scheduled_tasks st
                JOIN workflows w ON w.id = st.workflow_id
                WHERE st.is_active = 1 
                AND w.is_enabled = 1
                AND (st.next_run_at IS NULL OR st.next_run_at <= CURRENT_TIMESTAMP)
            """)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                r = dict(row)
                r['schedule_config'] = json.loads(r['schedule_config'])
                results.append(r)
            return results
    
    def update_task_next_run(self, task_id: int, next_run_at: datetime) -> bool:
        """Update the next run time for a task."""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE scheduled_tasks 
                SET last_run_at = CURRENT_TIMESTAMP, next_run_at = ?
                WHERE id = ?
            """, (next_run_at.isoformat(), task_id))
            conn.commit()
            return cursor.rowcount > 0
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _row_to_workflow(self, row: sqlite3.Row) -> Dict:
        """Convert a database row to workflow dict."""
        result = dict(row)
        result['nodes'] = json.loads(result['nodes'] or '[]')
        result['edges'] = json.loads(result['edges'] or '[]')
        result['triggers'] = json.loads(result['triggers'] or '[]')
        return result
    
    def _row_to_execution(self, row: sqlite3.Row) -> Dict:
        """Convert a database row to execution dict."""
        result = dict(row)
        result['completed_nodes'] = json.loads(result['completed_nodes'] or '[]')
        result['failed_nodes'] = json.loads(result['failed_nodes'] or '[]')
        result['context'] = json.loads(result['context'] or '{}')
        result['trigger_data'] = json.loads(result['trigger_data'] or '{}')
        return result


# Global instance
workflow_storage = WorkflowStorage()