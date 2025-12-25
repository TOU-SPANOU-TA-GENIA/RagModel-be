# app/db/workflow_schema.py
"""
Database schema for workflow system.
Supports workflow definitions, executions, baselines, and user settings.
"""

import sqlite3
from pathlib import Path
from typing import Optional
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class WorkflowSchemaManager:
    """Manages workflow-related database tables."""
    
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def initialize(self):
        """Create all workflow-related tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # =================================================================
            # Workflows Table - The blueprint/definition
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    name TEXT NOT NULL,
                    description TEXT,
                    
                    -- Graph definition (nodes and edges as JSON)
                    nodes JSON NOT NULL DEFAULT '[]',
                    edges JSON NOT NULL DEFAULT '[]',
                    
                    -- Workflow settings
                    is_enabled BOOLEAN DEFAULT 0,
                    is_shared BOOLEAN DEFAULT 0,
                    
                    -- Trigger configuration (JSON)
                    triggers JSON NOT NULL DEFAULT '[]',
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_run_at TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)
            
            # =================================================================
            # Workflow Executions - Run history and state for resume
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_executions (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    
                    -- Execution status
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'running', 'paused', 'completed', 'failed', 'cancelled')),
                    
                    -- Progress tracking
                    current_node_id TEXT,
                    completed_nodes JSON DEFAULT '[]',
                    failed_nodes JSON DEFAULT '[]',
                    
                    -- Context data passed between nodes (JSON)
                    context JSON DEFAULT '{}',
                    
                    -- Error information
                    error_message TEXT,
                    error_node_id TEXT,
                    
                    -- Timing
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    
                    -- Trigger info (what started this execution)
                    trigger_type TEXT,
                    trigger_data JSON,
                    
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                )
            """)
            
            # =================================================================
            # Workflow Node Logs - Detailed per-node execution logs
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_node_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    
                    -- Node execution details
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
                    
                    -- Input/Output data (JSON)
                    input_data JSON,
                    output_data JSON,
                    
                    -- Error details
                    error_message TEXT,
                    
                    -- Timing
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration_ms INTEGER,
                    
                    FOREIGN KEY (execution_id) REFERENCES workflow_executions(id) ON DELETE CASCADE
                )
            """)
            
            # =================================================================
            # Workflow Baselines - Learned patterns for anomaly detection
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT,
                    
                    -- What this baseline is for
                    baseline_type TEXT NOT NULL,  -- 'folder_stats', 'metric_avg', 'pattern'
                    baseline_key TEXT NOT NULL,   -- e.g., folder path, metric name
                    
                    -- Statistical data (JSON)
                    -- For numeric: {mean, std, min, max, count, last_n_values}
                    -- For patterns: {common_values, frequencies, anomalies}
                    statistics JSON NOT NULL,
                    
                    -- Learning metadata
                    sample_count INTEGER DEFAULT 0,
                    first_sample_at TIMESTAMP,
                    last_sample_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Configuration
                    is_active BOOLEAN DEFAULT 1,
                    anomaly_threshold REAL DEFAULT 2.0,  -- Standard deviations
                    
                    UNIQUE(workflow_id, baseline_type, baseline_key)
                )
            """)
            
            # =================================================================
            # User Settings - Email config, notification preferences
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    
                    -- Email configuration
                    email_notifications_enabled BOOLEAN DEFAULT 0,
                    notification_email TEXT,
                    
                    -- SMTP settings (for sending)
                    smtp_host TEXT,
                    smtp_port INTEGER DEFAULT 587,
                    smtp_username TEXT,
                    smtp_password TEXT,  -- Should be encrypted in production
                    smtp_use_tls BOOLEAN DEFAULT 1,
                    
                    -- Notification preferences
                    notify_on_workflow_complete BOOLEAN DEFAULT 1,
                    notify_on_workflow_failure BOOLEAN DEFAULT 1,
                    notify_on_anomaly_detected BOOLEAN DEFAULT 1,
                    
                    -- Quiet hours (don't send notifications)
                    quiet_hours_start TIME,
                    quiet_hours_end TIME,
                    
                    -- Metadata
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # =================================================================
            # File Watch Registry - Track watched folders/files
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_watch_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    
                    -- Watch configuration
                    watch_path TEXT NOT NULL,
                    watch_type TEXT DEFAULT 'folder'
                        CHECK(watch_type IN ('folder', 'file', 'pattern')),
                    recursive BOOLEAN DEFAULT 1,
                    
                    -- File patterns to watch
                    include_patterns JSON DEFAULT '["*"]',
                    exclude_patterns JSON DEFAULT '[]',
                    
                    -- State tracking
                    last_check_at TIMESTAMP,
                    last_change_at TIMESTAMP,
                    file_hash TEXT,  -- For single file watches
                    
                    -- Status
                    is_active BOOLEAN DEFAULT 1,
                    
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                )
            """)
            
            # =================================================================
            # Scheduled Tasks - Cron-like scheduling
            # =================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    
                    -- Schedule configuration
                    schedule_type TEXT NOT NULL
                        CHECK(schedule_type IN ('cron', 'interval', 'daily', 'weekly', 'monthly')),
                    schedule_config JSON NOT NULL,
                    -- For cron: {"cron_expression": "0 8 * * *"}
                    -- For interval: {"seconds": 3600}
                    -- For daily: {"time": "08:00", "timezone": "Europe/Athens"}
                    -- For weekly: {"day": "monday", "time": "08:00"}
                    -- For monthly: {"day": 1, "time": "08:00"}
                    
                    -- State
                    is_active BOOLEAN DEFAULT 1,
                    last_run_at TIMESTAMP,
                    next_run_at TIMESTAMP,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                )
            """)
            
            # =================================================================
            # Indexes for performance
            # =================================================================
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflows_user_id 
                ON workflows(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflows_enabled 
                ON workflows(is_enabled)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_workflow_id 
                ON workflow_executions(workflow_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_status 
                ON workflow_executions(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_node_logs_execution 
                ON workflow_node_logs(execution_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_baselines_workflow 
                ON workflow_baselines(workflow_id, baseline_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_watch_workflow 
                ON file_watch_registry(workflow_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run 
                ON scheduled_tasks(next_run_at) WHERE is_active = 1
            """)
            
            conn.commit()
            logger.info("✅ Workflow schema initialized successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Workflow schema initialization failed: {e}")
            raise
        finally:
            conn.close()
    
    def drop_all(self):
        """Drop all workflow tables (USE WITH CAUTION)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            tables = [
                'scheduled_tasks',
                'file_watch_registry',
                'user_settings',
                'workflow_baselines',
                'workflow_node_logs',
                'workflow_executions',
                'workflows'
            ]
            
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            
            conn.commit()
            logger.warning("🗑️  All workflow tables dropped")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Failed to drop workflow tables: {e}")
            raise
        finally:
            conn.close()


def init_workflow_schema(db_path: str = "data/app.db"):
    """Convenience function to initialize workflow schema."""
    manager = WorkflowSchemaManager(db_path)
    manager.initialize()


if __name__ == "__main__":
    init_workflow_schema()
    print("Workflow schema initialized successfully!")