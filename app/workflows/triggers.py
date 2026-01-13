import threading
import time
import schedule # Requires 'pip install schedule' or use simpler loop
from typing import List, Dict
from app.workflows.schemas import WorkflowDefinition
from app.workflows.engine import WorkflowEngine
from app.core.events import event_bus
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TriggerService:
    """
    Manages background listeners for workflows.
    """
    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
        self.workflows: List[WorkflowDefinition] = []
        self._running = False
        self._scheduler_thread = None

    def load_workflows(self, workflows: List[WorkflowDefinition]):
        """Register all workflows and set up their triggers."""
        self.workflows = workflows
        
        # Clear existing event subscriptions (Conceptually)
        # In this simple impl, we just re-scan
        
        for wf in self.workflows:
            if not wf.enabled:
                continue
                
            if wf.trigger.type == "event":
                self._setup_event_trigger(wf)
            elif wf.trigger.type == "schedule":
                self._setup_schedule_trigger(wf)

    def _setup_event_trigger(self, wf: WorkflowDefinition):
        event_name = wf.trigger.config.get("event")
        if not event_name:
            return

        def handler(data):
            logger.info(f"Event '{event_name}' triggered workflow '{wf.name}'")
            # Run in a separate thread to not block the event bus
            threading.Thread(target=self.engine.run, args=(wf, data)).start()

        event_bus.subscribe(event_name, handler)
        logger.info(f"Workflow '{wf.name}' subscribed to event '{event_name}'")

    def _setup_schedule_trigger(self, wf: WorkflowDefinition):
        cron = wf.trigger.config.get("cron") # Simplified: supporting 'every_X_minutes' for now
        interval_min = wf.trigger.config.get("interval_minutes")
        
        if interval_min:
            schedule.every(int(interval_min)).minutes.do(self._run_scheduled, wf)
            logger.info(f"Workflow '{wf.name}' scheduled every {interval_min} mins")

    def _run_scheduled(self, wf: WorkflowDefinition):
        logger.info(f"Schedule triggered workflow '{wf.name}'")
        self.engine.run(wf, {"source": "scheduler"})

    def start(self):
        if self._running:
            return
            
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()

    def _scheduler_loop(self):
        while self._running:
            schedule.run_pending()
            time.sleep(1)