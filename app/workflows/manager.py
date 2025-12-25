# app/workflows/manager.py
"""
Workflow Manager.
Central coordinator that ties together engine, services, and handlers.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.workflows.engine import workflow_engine, WorkflowEngine
from app.workflows.storage import workflow_storage, WorkflowStorage
from app.workflows.services.file_watcher import file_watcher_service, FileWatcherService, WatchConfig
from app.workflows.services.scheduler import scheduler_service, SchedulerService, ScheduleConfig
from app.workflows.models import NodeType, TriggerType
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class WorkflowManager:
    """
    Central manager for workflow system.
    
    Coordinates:
    - Workflow CRUD operations
    - Trigger services (file watcher, scheduler)
    - Execution engine
    - Handler registration
    """
    
    def __init__(
        self,
        storage: Optional[WorkflowStorage] = None,
        engine: Optional[WorkflowEngine] = None,
        file_watcher: Optional[FileWatcherService] = None,
        scheduler: Optional[SchedulerService] = None
    ):
        self.storage = storage or workflow_storage
        self.engine = engine or workflow_engine
        self.file_watcher = file_watcher or file_watcher_service
        self.scheduler = scheduler or scheduler_service
        self._initialized = False
    
    async def initialize(self):
        """Initialize the workflow system."""
        if self._initialized:
            return
        
        # Initialize database schema
        from app.db.workflow_schema import init_workflow_schema
        init_workflow_schema()
        
        # Register node handlers
        self._register_handlers()
        
        # Set trigger callbacks
        self.file_watcher.set_trigger_callback(self._on_file_trigger)
        self.scheduler.set_trigger_callback(self._on_schedule_trigger)
        
        # Start services
        await self.file_watcher.start()
        await self.scheduler.start()
        
        # Resume any interrupted executions
        await self._resume_interrupted_executions()
        
        self._initialized = True
        logger.info("✅ Workflow manager initialized")
    
    async def shutdown(self):
        """Shutdown the workflow system."""
        await self.file_watcher.stop()
        await self.scheduler.stop()
        self._initialized = False
        logger.info("🛑 Workflow manager shutdown")
    
    def _register_handlers(self):
        """Register all node handlers with the engine."""
        from app.workflows.handlers.triggers import (
            handle_file_watcher_trigger,
            handle_schedule_trigger,
            handle_manual_trigger
        )
        from app.workflows.handlers.processors import (
            handle_extract_content,
            handle_llm_analysis,
            handle_anomaly_detection,
            handle_cross_reference,
            handle_summarize
        )
        from app.workflows.handlers.actions import (
            handle_generate_report,
            handle_send_email,
            handle_save_to_folder
        )
        from app.workflows.handlers.flow import (
            handle_condition,
            handle_delay
        )
        
        # Triggers
        self.engine.register_handler(NodeType.TRIGGER_FILE_WATCHER.value, handle_file_watcher_trigger)
        self.engine.register_handler(NodeType.TRIGGER_SCHEDULE.value, handle_schedule_trigger)
        self.engine.register_handler(NodeType.TRIGGER_MANUAL.value, handle_manual_trigger)
        
        # Processors
        self.engine.register_handler(NodeType.PROCESSOR_EXTRACT_CONTENT.value, handle_extract_content)
        self.engine.register_handler(NodeType.PROCESSOR_LLM_ANALYSIS.value, handle_llm_analysis)
        self.engine.register_handler(NodeType.PROCESSOR_ANOMALY_DETECTION.value, handle_anomaly_detection)
        self.engine.register_handler(NodeType.PROCESSOR_CROSS_REFERENCE.value, handle_cross_reference)
        self.engine.register_handler(NodeType.PROCESSOR_SUMMARIZE.value, handle_summarize)
        
        # Actions
        self.engine.register_handler(NodeType.ACTION_GENERATE_REPORT.value, handle_generate_report)
        self.engine.register_handler(NodeType.ACTION_SEND_EMAIL.value, handle_send_email)
        self.engine.register_handler(NodeType.ACTION_SAVE_TO_FOLDER.value, handle_save_to_folder)
        
        # Flow
        self.engine.register_handler(NodeType.FLOW_CONDITION.value, handle_condition)
        self.engine.register_handler(NodeType.FLOW_DELAY.value, handle_delay)
        
        logger.info("📦 Node handlers registered")
    
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
        workflow_id = self.storage.create_workflow(
            name=name,
            user_id=user_id,
            description=description,
            nodes=nodes or [],
            edges=edges or [],
            is_shared=is_shared
        )
        
        # Setup triggers if nodes contain trigger nodes
        if nodes:
            self._setup_triggers(workflow_id, nodes)
        
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Get a workflow by ID."""
        return self.storage.get_workflow(workflow_id)
    
    def list_workflows(
        self,
        user_id: Optional[int] = None,
        include_shared: bool = True,
        enabled_only: bool = False
    ) -> List[Dict]:
        """List workflows."""
        return self.storage.list_workflows(
            user_id=user_id,
            include_shared=include_shared,
            enabled_only=enabled_only
        )
    
    def update_workflow(
        self,
        workflow_id: str,
        **kwargs
    ) -> bool:
        """Update a workflow."""
        success = self.storage.update_workflow(workflow_id, **kwargs)
        
        # Re-setup triggers if nodes changed
        if success and 'nodes' in kwargs:
            self._teardown_triggers(workflow_id)
            self._setup_triggers(workflow_id, kwargs['nodes'])
        
        return success
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        self._teardown_triggers(workflow_id)
        return self.storage.delete_workflow(workflow_id)
    
    def enable_workflow(self, workflow_id: str) -> bool:
        """Enable a workflow."""
        success = self.storage.set_workflow_enabled(workflow_id, True)
        
        if success:
            workflow = self.storage.get_workflow(workflow_id)
            if workflow:
                self._setup_triggers(workflow_id, workflow['nodes'])
        
        return success
    
    def disable_workflow(self, workflow_id: str) -> bool:
        """Disable a workflow."""
        self._teardown_triggers(workflow_id)
        return self.storage.set_workflow_enabled(workflow_id, False)
    
    # =========================================================================
    # Execution
    # =========================================================================
    
    async def run_workflow(
        self,
        workflow_id: str,
        trigger_data: Optional[Dict] = None,
        initial_context: Optional[Dict] = None
    ) -> str:
        """Manually run a workflow."""
        return await self.engine.start_execution(
            workflow_id=workflow_id,
            trigger_type='manual',
            trigger_data=trigger_data,
            initial_context=initial_context
        )
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution."""
        return await self.engine.pause_execution(execution_id)
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        return await self.engine.resume_execution(execution_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an execution."""
        return await self.engine.cancel_execution(execution_id)
    
    def get_execution(self, execution_id: str) -> Optional[Dict]:
        """Get execution status."""
        execution = self.storage.get_execution(execution_id)
        if execution:
            execution['node_logs'] = self.storage.get_node_logs(execution_id)
        return execution
    
    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List executions."""
        return self.storage.list_executions(
            workflow_id=workflow_id,
            status=status,
            limit=limit
        )
    
    async def _resume_interrupted_executions(self):
        """Resume executions that were interrupted (e.g., server restart)."""
        resumable = self.storage.get_resumable_executions()
        
        for execution in resumable:
            try:
                logger.info(f"Resuming execution: {execution['id']}")
                await self.engine.resume_execution(execution['id'])
            except Exception as e:
                logger.error(f"Failed to resume execution {execution['id']}: {e}")
    
    # =========================================================================
    # Trigger Setup
    # =========================================================================
    
    def _setup_triggers(self, workflow_id: str, nodes: List[Dict]):
        """Setup triggers from workflow nodes."""
        for node in nodes:
            node_type = node.get('type', '')
            config = node.get('config', {})
            
            if node_type == NodeType.TRIGGER_FILE_WATCHER.value:
                self._setup_file_watcher(workflow_id, config)
            
            elif node_type == NodeType.TRIGGER_SCHEDULE.value:
                self._setup_schedule(workflow_id, config)
    
    def _teardown_triggers(self, workflow_id: str):
        """Remove all triggers for a workflow."""
        self.file_watcher.remove_watch(workflow_id)
        # Schedule removal would need task_id tracking
    
    def _setup_file_watcher(self, workflow_id: str, config: Dict):
        """Setup a file watcher trigger."""
        watch_config = WatchConfig(
            workflow_id=workflow_id,
            watch_path=config.get('watch_path', ''),
            recursive=config.get('recursive', True),
            include_patterns=config.get('include_patterns', ['*']),
            exclude_patterns=config.get('exclude_patterns', []),
            debounce_seconds=config.get('debounce_seconds', 5.0)
        )
        
        # Register in database
        self.storage.register_file_watch(
            workflow_id=workflow_id,
            watch_path=watch_config.watch_path,
            recursive=watch_config.recursive,
            include_patterns=watch_config.include_patterns,
            exclude_patterns=watch_config.exclude_patterns
        )
        
        # Add to active watcher
        self.file_watcher.add_watch(watch_config)
    
    def _setup_schedule(self, workflow_id: str, config: Dict):
        """Setup a schedule trigger."""
        schedule_type = config.get('schedule_type', 'daily')
        
        # Create task in database
        task_id = self.storage.create_scheduled_task(
            workflow_id=workflow_id,
            schedule_type=schedule_type,
            schedule_config=config
        )
        
        # Add to scheduler
        schedule_config = ScheduleConfig(
            task_id=task_id,
            workflow_id=workflow_id,
            schedule_type=schedule_type,
            config=config,
            timezone=config.get('timezone', 'Europe/Athens')
        )
        
        self.scheduler.add_schedule(schedule_config)
    
    # =========================================================================
    # Trigger Callbacks
    # =========================================================================
    
    async def _on_file_trigger(self, workflow_id: str, trigger_data: Dict):
        """Handle file watcher trigger."""
        workflow = self.storage.get_workflow(workflow_id)
        
        if not workflow or not workflow['is_enabled']:
            return
        
        logger.info(f"📁 File trigger for workflow: {workflow['name']}")
        
        await self.engine.start_execution(
            workflow_id=workflow_id,
            trigger_type='file_watcher',
            trigger_data=trigger_data
        )
    
    async def _on_schedule_trigger(self, workflow_id: str, trigger_data: Dict):
        """Handle schedule trigger."""
        workflow = self.storage.get_workflow(workflow_id)
        
        if not workflow or not workflow['is_enabled']:
            return
        
        logger.info(f"⏰ Schedule trigger for workflow: {workflow['name']}")
        
        await self.engine.start_execution(
            workflow_id=workflow_id,
            trigger_type='schedule',
            trigger_data=trigger_data
        )
    
    # =========================================================================
    # User Settings
    # =========================================================================
    
    def get_user_settings(self, user_id: int) -> Optional[Dict]:
        """Get user settings."""
        return self.storage.get_user_settings(user_id)
    
    def update_user_settings(self, user_id: int, settings: Dict) -> bool:
        """Update user settings."""
        return self.storage.upsert_user_settings(user_id, settings)
    
    # =========================================================================
    # Templates
    # =========================================================================
    
    def get_workflow_templates(self) -> List[Dict]:
        """Get predefined workflow templates."""
        return [
            {
                'id': 'intelligence_fusion',
                'name': 'Intelligence Fusion Pipeline',
                'name_el': 'Σύνθεση Πληροφοριών',
                'description': 'Process new documents, cross-reference existing intel, generate briefing',
                'description_el': 'Επεξεργασία νέων εγγράφων, διασταύρωση με υπάρχουσες πληροφορίες, δημιουργία ενημέρωσης',
                'nodes': self._get_intelligence_template_nodes(),
                'edges': self._get_intelligence_template_edges()
            },
            {
                'id': 'supply_chain_anomaly',
                'name': 'Supply Chain Anomaly Detection',
                'name_el': 'Ανίχνευση Ανωμαλιών Εφοδιαστικής',
                'description': 'Monitor logistics data, detect anomalies, generate investigation report',
                'description_el': 'Παρακολούθηση δεδομένων εφοδιαστικής, ανίχνευση ανωμαλιών, δημιουργία αναφοράς',
                'nodes': self._get_supply_chain_template_nodes(),
                'edges': self._get_supply_chain_template_edges()
            }
        ]
    
    def _get_intelligence_template_nodes(self) -> List[Dict]:
        """Get nodes for intelligence fusion template."""
        return [
            {
                'id': 'trigger_1',
                'type': NodeType.TRIGGER_FILE_WATCHER.value,
                'label': 'Νέα Έγγραφα',
                'position': {'x': 100, 'y': 100},
                'config': {
                    'watch_path': 'Z:/intelligence/incoming',
                    'include_patterns': ['*.pdf', '*.docx', '*.xlsx'],
                    'recursive': True
                }
            },
            {
                'id': 'extract_1',
                'type': NodeType.PROCESSOR_EXTRACT_CONTENT.value,
                'label': 'Εξαγωγή Περιεχομένου',
                'position': {'x': 300, 'y': 100},
                'config': {
                    'file_types': ['pdf', 'docx', 'xlsx'],
                    'ocr_enabled': True,
                    'extract_tables': True
                }
            },
            {
                'id': 'analyze_1',
                'type': NodeType.PROCESSOR_LLM_ANALYSIS.value,
                'label': 'Ανάλυση',
                'position': {'x': 500, 'y': 100},
                'config': {
                    'prompt_template': '''Analyze the following intelligence documents and identify:
1. Key entities (people, organizations, locations)
2. Important dates and events
3. Potential threats or concerns
4. Actionable intelligence

Content: {content}''',
                    'language': 'greek',
                    'max_tokens': 2000
                }
            },
            {
                'id': 'crossref_1',
                'type': NodeType.PROCESSOR_CROSS_REFERENCE.value,
                'label': 'Διασταύρωση',
                'position': {'x': 700, 'y': 100},
                'config': {
                    'reference_source': 'folder',
                    'reference_path': 'Z:/intelligence/archive'
                }
            },
            {
                'id': 'report_1',
                'type': NodeType.ACTION_GENERATE_REPORT.value,
                'label': 'Αναφορά',
                'position': {'x': 900, 'y': 100},
                'config': {
                    'output_format': 'docx',
                    'output_path': 'Z:/intelligence/reports',
                    'filename_template': 'intel_briefing_{timestamp}',
                    'include_sections': ['summary', 'details', 'recommendations']
                }
            },
            {
                'id': 'condition_1',
                'type': NodeType.FLOW_CONDITION.value,
                'label': 'Κρίσιμο;',
                'position': {'x': 700, 'y': 250},
                'config': {
                    'condition_type': 'expression',
                    'expression': 'has_anomalies or anomaly_count > 0',
                    'true_branch': 'email_1',
                    'false_branch': None
                }
            },
            {
                'id': 'email_1',
                'type': NodeType.ACTION_SEND_EMAIL.value,
                'label': 'Ειδοποίηση',
                'position': {'x': 900, 'y': 250},
                'config': {
                    'subject_template': '⚠️ Κρίσιμη Πληροφορία - {timestamp}',
                    'body_template': 'Εντοπίστηκε κρίσιμη πληροφορία.\n\n{summary}\n\nΑναφορά: {report_path}',
                    'attach_report': True
                }
            }
        ]
    
    def _get_intelligence_template_edges(self) -> List[Dict]:
        """Get edges for intelligence fusion template."""
        return [
            {'id': 'e1', 'source': 'trigger_1', 'target': 'extract_1'},
            {'id': 'e2', 'source': 'extract_1', 'target': 'analyze_1'},
            {'id': 'e3', 'source': 'analyze_1', 'target': 'crossref_1'},
            {'id': 'e4', 'source': 'crossref_1', 'target': 'report_1'},
            {'id': 'e5', 'source': 'crossref_1', 'target': 'condition_1'},
            {'id': 'e6', 'source': 'condition_1', 'target': 'email_1'}
        ]
    
    def _get_supply_chain_template_nodes(self) -> List[Dict]:
        """Get nodes for supply chain anomaly template."""
        return [
            {
                'id': 'trigger_1',
                'type': NodeType.TRIGGER_SCHEDULE.value,
                'label': 'Καθημερινός Έλεγχος',
                'position': {'x': 100, 'y': 100},
                'config': {
                    'schedule_type': 'daily',
                    'time': '08:00',
                    'timezone': 'Europe/Athens'
                }
            },
            {
                'id': 'trigger_2',
                'type': NodeType.TRIGGER_FILE_WATCHER.value,
                'label': 'Νέα Αρχεία',
                'position': {'x': 100, 'y': 250},
                'config': {
                    'watch_path': 'Z:/logistics/daily',
                    'include_patterns': ['*.xlsx', '*.csv'],
                    'recursive': False
                }
            },
            {
                'id': 'extract_1',
                'type': NodeType.PROCESSOR_EXTRACT_CONTENT.value,
                'label': 'Εξαγωγή Δεδομένων',
                'position': {'x': 300, 'y': 175},
                'config': {
                    'file_types': ['xlsx', 'csv'],
                    'extract_tables': True
                }
            },
            {
                'id': 'anomaly_1',
                'type': NodeType.PROCESSOR_ANOMALY_DETECTION.value,
                'label': 'Ανίχνευση Ανωμαλιών',
                'position': {'x': 500, 'y': 175},
                'config': {
                    'detection_type': 'statistical',
                    'threshold_std': 2.0,
                    'use_baseline': True,
                    'fields_to_monitor': ['fuel', 'ammo', 'supplies', 'consumption']
                }
            },
            {
                'id': 'condition_1',
                'type': NodeType.FLOW_CONDITION.value,
                'label': 'Ανωμαλίες;',
                'position': {'x': 700, 'y': 175},
                'config': {
                    'condition_type': 'threshold',
                    'field': 'anomaly_count',
                    'operator': 'gt',
                    'threshold': 0,
                    'true_branch': 'analyze_1',
                    'false_branch': 'save_1'
                }
            },
            {
                'id': 'analyze_1',
                'type': NodeType.PROCESSOR_LLM_ANALYSIS.value,
                'label': 'Ανάλυση Αιτίων',
                'position': {'x': 900, 'y': 100},
                'config': {
                    'prompt_template': '''Analyze these logistics anomalies and provide:
1. Possible causes for each anomaly
2. Historical comparison
3. Recommended investigation steps
4. Priority level

Anomalies: {content}''',
                    'language': 'greek'
                }
            },
            {
                'id': 'report_1',
                'type': NodeType.ACTION_GENERATE_REPORT.value,
                'label': 'Αναφορά Έρευνας',
                'position': {'x': 1100, 'y': 100},
                'config': {
                    'output_format': 'docx',
                    'output_path': 'Z:/logistics/reports',
                    'filename_template': 'anomaly_report_{timestamp}',
                    'include_sections': ['summary', 'anomalies', 'recommendations']
                }
            },
            {
                'id': 'email_1',
                'type': NodeType.ACTION_SEND_EMAIL.value,
                'label': 'Ειδοποίηση',
                'position': {'x': 1100, 'y': 250},
                'config': {
                    'subject_template': '🔴 Ανωμαλίες Εφοδιαστικής - {anomaly_count} εντοπίστηκαν',
                    'body_template': 'Εντοπίστηκαν {anomaly_count} ανωμαλίες στα δεδομένα εφοδιαστικής.\n\nΔείτε την αναφορά.',
                    'attach_report': True
                }
            },
            {
                'id': 'save_1',
                'type': NodeType.ACTION_SAVE_TO_FOLDER.value,
                'label': 'Αρχειοθέτηση',
                'position': {'x': 900, 'y': 300},
                'config': {
                    'destination_path': 'Z:/logistics/archive',
                    'create_subfolders': True
                }
            }
        ]
    
    def _get_supply_chain_template_edges(self) -> List[Dict]:
        """Get edges for supply chain anomaly template."""
        return [
            {'id': 'e1', 'source': 'trigger_1', 'target': 'extract_1'},
            {'id': 'e2', 'source': 'trigger_2', 'target': 'extract_1'},
            {'id': 'e3', 'source': 'extract_1', 'target': 'anomaly_1'},
            {'id': 'e4', 'source': 'anomaly_1', 'target': 'condition_1'},
            {'id': 'e5', 'source': 'condition_1', 'target': 'analyze_1'},
            {'id': 'e6', 'source': 'condition_1', 'target': 'save_1'},
            {'id': 'e7', 'source': 'analyze_1', 'target': 'report_1'},
            {'id': 'e8', 'source': 'report_1', 'target': 'email_1'},
            {'id': 'e9', 'source': 'report_1', 'target': 'save_1'}
        ]


# Global instance
workflow_manager = WorkflowManager()