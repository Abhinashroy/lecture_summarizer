"""
Base Agent class for the multi-agent lecture summarizer system.
Provides common functionality and interface for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
import time

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.performance_metrics = {}
        
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and return results."""
        pass
    
    def log_performance(self, start_time: float, operation: str) -> float:
        """Log performance metrics for an operation."""
        duration = time.time() - start_time
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(duration)
        self.logger.info(f"{operation} completed in {duration:.2f}s")
        return duration
    
    def get_average_performance(self, operation: str) -> float:
        """Get average performance for a specific operation."""
        if operation in self.performance_metrics:
            return sum(self.performance_metrics[operation]) / len(self.performance_metrics[operation])
        return 0.0
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        total_operations = sum(len(times) for times in self.performance_metrics.values())
        avg_performance = sum(
            sum(times) / len(times) for times in self.performance_metrics.values()
        ) / len(self.performance_metrics) if self.performance_metrics else 0
        
        return {
            "name": self.name,
            "status": "active" if total_operations > 0 else "ready",
            "total_operations": total_operations,
            "performance_metrics": self.performance_metrics,
            "average_times": {
                op: self.get_average_performance(op) 
                for op in self.performance_metrics.keys()
            },
            "overall_avg_time": avg_performance,
            "quality_score": getattr(self, 'last_quality_score', 0.8)  # Default good score
        }