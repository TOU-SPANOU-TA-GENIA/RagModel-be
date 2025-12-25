# app/workflows/services/scheduler.py
"""
Scheduler Service.
Handles cron-like scheduling for workflow triggers.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from app.workflows.storage import workflow_storage
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import APScheduler
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed, using basic scheduler")


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled task."""
    task_id: int
    workflow_id: str
    schedule_type: str  # 'cron', 'interval', 'daily', 'weekly', 'monthly'
    config: Dict
    timezone: str = "Europe/Athens"


class SchedulerService:
    """
    Service that handles scheduled workflow triggers.
    
    Uses APScheduler if available, otherwise uses a simple async loop.
    """
    
    def __init__(self):
        self._scheduler: Optional["AsyncIOScheduler"] = None
        self._running = False
        self._tasks: Dict[int, ScheduleConfig] = {}
        self._on_trigger: Optional[Callable] = None
        self._basic_task: Optional[asyncio.Task] = None
    
    def set_trigger_callback(self, callback: Callable):
        """Set callback for when a workflow should be triggered."""
        self._on_trigger = callback
    
    async def start(self):
        """Start the scheduler service."""
        if self._running:
            return
        
        self._running = True
        
        # Load scheduled tasks
        await self._load_tasks()
        
        if APSCHEDULER_AVAILABLE:
            self._start_apscheduler()
        else:
            self._basic_task = asyncio.create_task(self._basic_scheduler_loop())
        
        logger.info(f"⏰ Scheduler service started ({len(self._tasks)} tasks)")
    
    async def stop(self):
        """Stop the scheduler service."""
        self._running = False
        
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        
        if self._basic_task:
            self._basic_task.cancel()
            try:
                await self._basic_task
            except asyncio.CancelledError:
                pass
            self._basic_task = None
        
        logger.info("⏰ Scheduler service stopped")
    
    async def _load_tasks(self):
        """Load scheduled tasks from database."""
        tasks = workflow_storage.get_due_tasks()
        
        for task in tasks:
            config = ScheduleConfig(
                task_id=task['id'],
                workflow_id=task['workflow_id'],
                schedule_type=task['schedule_type'],
                config=task['schedule_config'],
                timezone=task['schedule_config'].get('timezone', 'Europe/Athens')
            )
            self._tasks[task['id']] = config
    
    def add_schedule(self, config: ScheduleConfig) -> bool:
        """Add a new scheduled task."""
        self._tasks[config.task_id] = config
        
        if APSCHEDULER_AVAILABLE and self._scheduler:
            self._add_apscheduler_job(config)
        
        logger.info(f"Added schedule for workflow {config.workflow_id}: {config.schedule_type}")
        return True
    
    def remove_schedule(self, task_id: int):
        """Remove a scheduled task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
        
        if APSCHEDULER_AVAILABLE and self._scheduler:
            try:
                self._scheduler.remove_job(str(task_id))
            except Exception:
                pass
        
        logger.info(f"Removed schedule task {task_id}")
    
    # =========================================================================
    # APScheduler Mode
    # =========================================================================
    
    def _start_apscheduler(self):
        """Start APScheduler."""
        self._scheduler = AsyncIOScheduler()
        
        for task_id, config in self._tasks.items():
            self._add_apscheduler_job(config)
        
        self._scheduler.start()
    
    def _add_apscheduler_job(self, config: ScheduleConfig):
        """Add a job to APScheduler."""
        trigger = self._create_trigger(config)
        
        if trigger:
            self._scheduler.add_job(
                self._trigger_workflow,
                trigger,
                id=str(config.task_id),
                args=[config.workflow_id, config.task_id],
                replace_existing=True
            )
    
    def _create_trigger(self, config: ScheduleConfig):
        """Create an APScheduler trigger from config."""
        schedule_type = config.schedule_type
        schedule_config = config.config
        
        try:
            if schedule_type == 'cron':
                cron_expr = schedule_config.get('cron_expression', '0 8 * * *')
                return CronTrigger.from_crontab(cron_expr, timezone=config.timezone)
            
            elif schedule_type == 'interval':
                seconds = schedule_config.get('seconds', 3600)
                return IntervalTrigger(seconds=seconds)
            
            elif schedule_type == 'daily':
                time_str = schedule_config.get('time', '08:00')
                hour, minute = map(int, time_str.split(':'))
                return CronTrigger(
                    hour=hour,
                    minute=minute,
                    timezone=config.timezone
                )
            
            elif schedule_type == 'weekly':
                time_str = schedule_config.get('time', '08:00')
                day = schedule_config.get('day', 'monday')
                hour, minute = map(int, time_str.split(':'))
                
                day_map = {
                    'monday': 0, 'tuesday': 1, 'wednesday': 2,
                    'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
                }
                day_num = day_map.get(day.lower(), 0)
                
                return CronTrigger(
                    day_of_week=day_num,
                    hour=hour,
                    minute=minute,
                    timezone=config.timezone
                )
            
            elif schedule_type == 'monthly':
                time_str = schedule_config.get('time', '08:00')
                day = schedule_config.get('day', 1)
                hour, minute = map(int, time_str.split(':'))
                
                return CronTrigger(
                    day=day,
                    hour=hour,
                    minute=minute,
                    timezone=config.timezone
                )
        
        except Exception as e:
            logger.error(f"Failed to create trigger: {e}")
        
        return None
    
    async def _trigger_workflow(self, workflow_id: str, task_id: int):
        """Trigger a workflow from schedule."""
        if self._on_trigger:
            await self._on_trigger(
                workflow_id=workflow_id,
                trigger_data={
                    'task_id': task_id,
                    'scheduled_at': datetime.utcnow().isoformat()
                }
            )
        
        # Update next run time
        next_run = self._calculate_next_run(task_id)
        if next_run:
            workflow_storage.update_task_next_run(task_id, next_run)
    
    def _calculate_next_run(self, task_id: int) -> Optional[datetime]:
        """Calculate the next run time for a task."""
        config = self._tasks.get(task_id)
        if not config:
            return None
        
        schedule_type = config.schedule_type
        schedule_config = config.config
        now = datetime.utcnow()
        
        if schedule_type == 'interval':
            seconds = schedule_config.get('seconds', 3600)
            return now + timedelta(seconds=seconds)
        
        elif schedule_type == 'daily':
            time_str = schedule_config.get('time', '08:00')
            hour, minute = map(int, time_str.split(':'))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        
        elif schedule_type == 'weekly':
            time_str = schedule_config.get('time', '08:00')
            day = schedule_config.get('day', 'monday')
            hour, minute = map(int, time_str.split(':'))
            
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2,
                'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
            }
            target_day = day_map.get(day.lower(), 0)
            
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return next_run
        
        elif schedule_type == 'monthly':
            time_str = schedule_config.get('time', '08:00')
            day = schedule_config.get('day', 1)
            hour, minute = map(int, time_str.split(':'))
            
            next_run = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                # Move to next month
                if now.month == 12:
                    next_run = next_run.replace(year=now.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=now.month + 1)
            return next_run
        
        return None
    
    # =========================================================================
    # Basic Scheduler Mode (without APScheduler)
    # =========================================================================
    
    async def _basic_scheduler_loop(self):
        """Basic scheduler loop when APScheduler is not available."""
        check_interval = 60  # Check every minute
        
        while self._running:
            try:
                now = datetime.utcnow()
                
                for task_id, config in list(self._tasks.items()):
                    if self._should_run(config, now):
                        await self._trigger_workflow(config.workflow_id, task_id)
                        
                        # Update next run
                        next_run = self._calculate_next_run(task_id)
                        if next_run:
                            workflow_storage.update_task_next_run(task_id, next_run)
            
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            
            await asyncio.sleep(check_interval)
    
    def _should_run(self, config: ScheduleConfig, now: datetime) -> bool:
        """Check if a task should run now (basic mode)."""
        schedule_type = config.schedule_type
        schedule_config = config.config
        
        if schedule_type == 'interval':
            # For interval, check if enough time has passed
            # This would need last_run tracking
            return False  # Simplified - would need proper tracking
        
        elif schedule_type == 'daily':
            time_str = schedule_config.get('time', '08:00')
            hour, minute = map(int, time_str.split(':'))
            return now.hour == hour and now.minute == minute
        
        elif schedule_type == 'weekly':
            time_str = schedule_config.get('time', '08:00')
            day = schedule_config.get('day', 'monday')
            hour, minute = map(int, time_str.split(':'))
            
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2,
                'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
            }
            target_day = day_map.get(day.lower(), 0)
            
            return (
                now.weekday() == target_day and
                now.hour == hour and
                now.minute == minute
            )
        
        return False


# Global instance
scheduler_service = SchedulerService()