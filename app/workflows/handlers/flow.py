# app/workflows/handlers/flow.py
"""
Flow control node handlers.
These nodes control workflow execution flow: conditions, loops, delays.
"""

import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta

from app.workflows.engine import NodeResult, ExecutionContext
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


async def handle_condition(
    config: Dict[str, Any],
    context: ExecutionContext
) -> NodeResult:
    """
    Conditional branching node.
    
    Config:
        condition_type: str - 'expression', 'contains', 'threshold', 'flag'
        expression: str - Condition expression
        field: str - Context field to check
        value: any - Value to compare
        true_branch: str - Node ID for true path
        false_branch: str - Node ID for false path
    
    Returns NodeResult with next_nodes set based on condition.
    """
    condition_type = config.get('condition_type', 'expression')
    expression = config.get('expression', 'True')
    true_branch = config.get('true_branch')
    false_branch = config.get('false_branch')
    
    # Warn if branches not configured
    if not true_branch and not false_branch:
        logger.warning("⚠️ Condition node has no true_branch or false_branch configured!")
        logger.warning("   Configure these in the node settings to enable proper branching.")
    
    result = False
    
    try:
        if condition_type == 'flag':
            # Simple boolean flag check
            field = config.get('field', 'critical_found')
            value = context.get(field, False)
            result = bool(value)
            logger.info(f"🔀 Flag check: {field} = {value} -> {result}")
        
        elif condition_type == 'expression':
            # Evaluate expression with context
            # Convert JavaScript-style expressions to Python
            py_expression = expression
            # Handle common JS to Python conversions
            py_expression = py_expression.replace('.length', ')')
            py_expression = py_expression.replace('critical_keywords)', 'len(critical_keywords)')
            py_expression = py_expression.replace('anomalies)', 'len(anomalies)')
            
            eval_context = {
                'context': context,
                'data': context.data,
                'anomaly_count': context.get('anomaly_count', 0),
                'file_count': len(context.get('source_files', [])),
                'has_anomalies': len(context.get('anomalies', [])) > 0,
                'critical_found': context.get('critical_found', False),
                'critical_keywords': context.get('critical_keywords', []),
                'anomalies': context.get('anomalies', []),
                'True': True,
                'False': False,
                'len': len,
            }
            
            # Add all context data for easy access
            eval_context.update(context.data)
            
            # Add node outputs
            for node_id, output in context.node_outputs.items():
                eval_context[f'node_{node_id}'] = output
            
            result = eval(py_expression, {"__builtins__": {'len': len}}, eval_context)
        
        elif condition_type == 'contains':
            field = config.get('field', '')
            value = config.get('value', '')
            data = context.get(field, '')
            if isinstance(data, dict):
                data = str(data)
            result = value in str(data)
        
        elif condition_type == 'threshold':
            field = config.get('field', 'anomaly_count')
            threshold = config.get('threshold', 0)
            operator = config.get('operator', 'gt')
            
            actual_value = context.get(field, 0)
            if isinstance(actual_value, list):
                actual_value = len(actual_value)
            
            if operator == 'gt':
                result = actual_value > threshold
            elif operator == 'gte':
                result = actual_value >= threshold
            elif operator == 'lt':
                result = actual_value < threshold
            elif operator == 'lte':
                result = actual_value <= threshold
            elif operator == 'eq':
                result = actual_value == threshold
            elif operator == 'ne':
                result = actual_value != threshold
    
    except Exception as e:
        logger.error(f"Condition evaluation failed: {e}")
        result = False
    
    # Determine next nodes
    next_nodes = []
    if result and true_branch:
        next_nodes.append(true_branch)
    elif not result and false_branch:
        next_nodes.append(false_branch)
    
    logger.info(f"🔀 Condition '{condition_type}' evaluated to {result}")
    
    return NodeResult(
        success=True,
        output={
            'condition': expression if condition_type == 'expression' else config.get('field', ''),
            'result': result,
            'next_branch': 'true' if result else 'false'
        },
        next_nodes=next_nodes
    )


async def handle_delay(
    config: Dict[str, Any],
    context: ExecutionContext
) -> NodeResult:
    """
    Delay node - pauses execution for specified time.
    
    Config:
        delay_seconds: int - Seconds to wait
        delay_until: str - Time to wait until (HH:MM)
    """
    delay_seconds = config.get('delay_seconds', 0)
    delay_until = config.get('delay_until')
    
    if delay_until:
        # Calculate seconds until target time
        try:
            target_time = datetime.strptime(delay_until, '%H:%M').time()
            now = datetime.utcnow()
            target = datetime.combine(now.date(), target_time)
            
            if target <= now:
                # Target is tomorrow
                target += timedelta(days=1)
            
            delay_seconds = (target - now).total_seconds()
        except ValueError:
            logger.warning(f"Invalid delay_until format: {delay_until}")
    
    if delay_seconds > 0:
        logger.info(f"⏳ Delaying execution for {delay_seconds} seconds")
        await asyncio.sleep(delay_seconds)
    
    return NodeResult(
        success=True,
        output={
            'delayed_seconds': delay_seconds,
            'resumed_at': datetime.utcnow().isoformat()
        }
    )


async def handle_loop(
    config: Dict[str, Any],
    context: ExecutionContext
) -> NodeResult:
    """
    Loop node - iterates over items.
    
    Config:
        loop_type: str - 'foreach', 'while', 'count'
        source_field: str - Context field to iterate
        max_iterations: int - Safety limit
    
    Note: This is a simplified implementation. Full loop support
    would require changes to the engine to support re-execution.
    """
    loop_type = config.get('loop_type', 'foreach')
    source_field = config.get('source_field', '')
    max_iterations = config.get('max_iterations', 100)
    
    # Get current iteration from context
    loop_index = context.get('_loop_index', 0)
    
    if loop_type == 'foreach':
        items = context.get(source_field, [])
        
        if loop_index < len(items) and loop_index < max_iterations:
            # Set current item and increment index
            context.set('_loop_current', items[loop_index])
            context.set('_loop_index', loop_index + 1)
            context.set('_loop_has_more', loop_index + 1 < len(items))
            
            return NodeResult(
                success=True,
                output={
                    'iteration': loop_index,
                    'total': len(items),
                    'current_item': items[loop_index]
                }
            )
        else:
            # Loop complete
            context.set('_loop_complete', True)
            return NodeResult(
                success=True,
                output={
                    'loop_complete': True,
                    'total_iterations': loop_index
                },
                should_continue=True  # Continue to next node after loop
            )
    
    elif loop_type == 'count':
        count = config.get('count', 1)
        
        if loop_index < count and loop_index < max_iterations:
            context.set('_loop_index', loop_index + 1)
            return NodeResult(
                success=True,
                output={
                    'iteration': loop_index,
                    'total': count
                }
            )
        else:
            context.set('_loop_complete', True)
            return NodeResult(
                success=True,
                output={'loop_complete': True}
            )
    
    return NodeResult(success=True, output={'loop_type': loop_type})


async def handle_merge(
    config: Dict[str, Any],
    context: ExecutionContext
) -> NodeResult:
    """
    Merge node - waits for multiple branches to complete.
    
    This is handled implicitly by the engine's dependency checking.
    The merge node passes through when all incoming edges are satisfied.
    """
    merge_strategy = config.get('strategy', 'all')  # 'all', 'any'
    
    # Collect outputs from all incoming nodes
    incoming_outputs = {}
    incoming_nodes = config.get('_incoming_nodes', [])
    
    for node_id in incoming_nodes:
        output = context.get_node_output(node_id)
        if output:
            incoming_outputs[node_id] = output
    
    context.set('merged_outputs', incoming_outputs)
    
    logger.info(f"🔗 Merge completed from {len(incoming_outputs)} branches")
    
    return NodeResult(
        success=True,
        output={
            'merged_count': len(incoming_outputs),
            'merged_from': list(incoming_outputs.keys())
        }
    )