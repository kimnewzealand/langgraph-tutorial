"""Performance monitoring utilities for LangGraph agent."""

import time
import logging
from typing import Dict, Any, Optional
from functools import wraps
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track performance metrics for the agent."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation."""
        self.metrics[f"{operation}_duration"].append(duration)
        self.counters[f"{operation}_count"] += 1
        
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.counters[f"{cache_type}_cache_hits"] += 1
        
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.counters[f"{cache_type}_cache_misses"] += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        # Calculate averages for timing metrics
        for key, values in self.metrics.items():
            if values and key.endswith('_duration'):
                stats[f"{key}_avg"] = sum(values) / len(values)
                stats[f"{key}_min"] = min(values)
                stats[f"{key}_max"] = max(values)
                stats[f"{key}_count"] = len(values)
        
        # Add counters
        stats.update(dict(self.counters))
        
        # Calculate cache hit rates
        for cache_type in ['response', 'vectorstore']:
            hits = self.counters.get(f"{cache_type}_cache_hits", 0)
            misses = self.counters.get(f"{cache_type}_cache_misses", 0)
            total = hits + misses
            if total > 0:
                stats[f"{cache_type}_cache_hit_rate"] = hits / total
                
        return stats
        
    def log_stats(self):
        """Log current performance statistics."""
        stats = self.get_stats()
        logger.info("Performance Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                performance_monitor.record_timing(operation_name, duration)
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"Slow operation {operation_name}: {duration:.3f}s")
        return wrapper
    return decorator

def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics."""
    return performance_monitor.get_stats()

def log_performance_stats():
    """Log current performance statistics."""
    performance_monitor.log_stats()
