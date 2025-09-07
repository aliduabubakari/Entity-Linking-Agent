import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

class AgentEventType(Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL = "tool_call"
    KB_QUERY = "kb_query"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"
    STATE_TRANSITION = "state_transition"

@dataclass
class AgentEvent:
    event_id: str
    request_id: str
    agent_name: str
    event_type: AgentEventType
    timestamp: datetime
    duration_ms: Optional[float] = None
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    metadata: Dict[str, Any] = None
    parent_event_id: Optional[str] = None

class AgentMonitor:
    """Comprehensive monitoring for agent activities"""
    
    def __init__(self):
        self.events: List[AgentEvent] = []
        self.active_events: Dict[str, float] = {}
        self.logger = logging.getLogger("agent_monitor")
        
        # Configure structured logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:  # Prevent duplicate handlers
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def generate_event_id(self) -> str:
        return f"evt_{int(time.time() * 1000000)}"
    
    @contextmanager
    def track_agent_execution(self, request_id: str, agent_name: str, input_data: Dict = None):
        """Context manager to track agent execution"""
        event_id = self.generate_event_id()
        start_time = time.time()
        
        # Log start event
        start_event = AgentEvent(
            event_id=event_id,
            request_id=request_id,
            agent_name=agent_name,
            event_type=AgentEventType.AGENT_START,
            timestamp=datetime.now(),
            input_data=input_data or {},
            metadata={"status": "started"}
        )
        
        self.events.append(start_event)
        self.active_events[event_id] = start_time
        
        self.logger.info(f"🚀 Agent {agent_name} started - Event: {event_id}")
        
        try:
            yield event_id
            
            # Success end event
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            end_event = AgentEvent(
                event_id=self.generate_event_id(),
                request_id=request_id,
                agent_name=agent_name,
                event_type=AgentEventType.AGENT_END,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                metadata={"status": "completed", "parent_event": event_id}
            )
            
            self.events.append(end_event)
            self.active_events.pop(event_id, None)
            
            self.logger.info(f"✅ Agent {agent_name} completed - Duration: {duration_ms:.2f}ms")
            
        except Exception as e:
            # Error end event
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            error_event = AgentEvent(
                event_id=self.generate_event_id(),
                request_id=request_id,
                agent_name=agent_name,
                event_type=AgentEventType.ERROR_OCCURRED,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                metadata={"status": "error", "error": str(e), "parent_event": event_id}
            )
            
            self.events.append(error_event)
            self.active_events.pop(event_id, None)
            
            self.logger.error(f"❌ Agent {agent_name} failed - Error: {str(e)}")
            raise
    
    def log_tool_call(self, request_id: str, agent_name: str, tool_name: str, 
                      input_data: Dict = None, output_data: Dict = None, duration_ms: float = None):
        """Log tool/KB calls"""
        event = AgentEvent(
            event_id=self.generate_event_id(),
            request_id=request_id,
            agent_name=agent_name,
            event_type=AgentEventType.TOOL_CALL,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            input_data=input_data or {},
            output_data=output_data or {},
            metadata={"tool_name": tool_name}
        )
        
        self.events.append(event)
        self.logger.info(f"🔧 {agent_name} used tool {tool_name} - Duration: {duration_ms:.2f}ms")
    
    def log_decision(self, request_id: str, agent_name: str, decision_type: str, 
                     decision_data: Dict, reasoning: str = None):
        """Log agent decisions"""
        event = AgentEvent(
            event_id=self.generate_event_id(),
            request_id=request_id,
            agent_name=agent_name,
            event_type=AgentEventType.DECISION_MADE,
            timestamp=datetime.now(),
            output_data=decision_data or {},
            metadata={"decision_type": decision_type, "reasoning": reasoning or ""}
        )
        
        self.events.append(event)
        self.logger.info(f"🎯 {agent_name} made decision: {decision_type}")
    
    def get_execution_timeline(self, request_id: str) -> List[Dict]:
        """Get chronological timeline of events for a request"""
        request_events = [e for e in self.events if e.request_id == request_id]
        request_events.sort(key=lambda x: x.timestamp)
        
        return [asdict(event) for event in request_events]
    
    def get_agent_performance_stats(self, agent_name: str = None) -> Dict:
        """Get performance statistics"""
        relevant_events = self.events
        if agent_name:
            relevant_events = [e for e in self.events if e.agent_name == agent_name]
        
        agent_stats = {}
        for event in relevant_events:
            if event.event_type in [AgentEventType.AGENT_END, AgentEventType.ERROR_OCCURRED]:
                agent = event.agent_name
                if agent not in agent_stats:
                    agent_stats[agent] = {
                        "total_executions": 0,
                        "successful_executions": 0,
                        "failed_executions": 0,
                        "total_duration_ms": 0,
                        "avg_duration_ms": 0
                    }
                
                agent_stats[agent]["total_executions"] += 1
                
                if event.event_type == AgentEventType.AGENT_END:
                    agent_stats[agent]["successful_executions"] += 1
                else:
                    agent_stats[agent]["failed_executions"] += 1
                
                if event.duration_ms:
                    agent_stats[agent]["total_duration_ms"] += event.duration_ms
        
        # Calculate averages
        for agent, stats in agent_stats.items():
            if stats["total_executions"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["total_executions"]
                stats["success_rate"] = stats["successful_executions"] / stats["total_executions"]
            else:
                stats["avg_duration_ms"] = 0
                stats["success_rate"] = 0
        
        return agent_stats

# Global monitor instance
agent_monitor = AgentMonitor()