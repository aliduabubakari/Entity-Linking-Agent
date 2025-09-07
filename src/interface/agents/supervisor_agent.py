from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from interface.schemas import AgentState
from infrastructure.monitoring.agent_monitor import agent_monitor, AgentEventType
from infrastructure.llm.llm_service import AzureOpenAILanguageModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """Master orchestrator that coordinates all other agents"""
    
    def __init__(self, llm_service: AzureOpenAILanguageModel):
        self.llm = llm_service
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the supervision workflow"""
        workflow = StateGraph(AgentState)
        
        # Supervision nodes
        workflow.add_node("initialize_request", self.initialize_request)
        workflow.add_node("validate_input", self.validate_input)
        workflow.add_node("plan_execution", self.plan_execution)
        workflow.add_node("monitor_progress", self.monitor_progress)
        workflow.add_node("quality_check", self.quality_check)
        workflow.add_node("finalize_results", self.finalize_results)
        
        # Conditional routing
        workflow.add_conditional_edges(
            "validate_input",
            self.should_continue_after_validation,
            {
                "continue": "plan_execution",
                "reject": END
            }
        )
        
        workflow.add_conditional_edges(
            "quality_check", 
            self.should_reprocess,
            {
                "approve": "finalize_results",
                "reprocess": "plan_execution", 
                "fail": END
            }
        )
        
        workflow.set_entry_point("initialize_request")
        workflow.add_edge("initialize_request", "validate_input")
        workflow.add_edge("plan_execution", "monitor_progress")
        workflow.add_edge("monitor_progress", "quality_check")
        workflow.add_edge("finalize_results", END)
        
        return workflow.compile()
    
    async def initialize_request(self, state: AgentState) -> Dict[str, Any]:
        """Initialize and set up the request"""
        request_id = state.metadata.get("request_id", f"req_{int(datetime.now().timestamp())}")
        
        with agent_monitor.track_agent_execution(request_id, "SupervisorAgent", 
                                                {"action": "initialize_request"}):
            
            # Add supervision metadata
            supervision_metadata = {
                "request_id": request_id,
                "start_time": datetime.now().isoformat(),
                "planned_agents": [],
                "execution_plan": {},
                "quality_gates": {
                    "min_confidence": 0.6,
                    "min_success_rate": 0.7,
                    "max_processing_time": 30.0
                }
            }
            
            agent_monitor.log_decision(
                request_id, "SupervisorAgent", "initialization",
                {"column_name": state.column_name, "values_count": len(state.column_values)},
                "Request initialized with supervision metadata"
            )
            
            return {
                "metadata": {**state.metadata, **supervision_metadata}
            }
    
    async def validate_input(self, state: AgentState) -> Dict[str, Any]:
        """Validate input data quality and completeness"""
        request_id = state.metadata["request_id"]
        
        with agent_monitor.track_agent_execution(request_id, "SupervisorAgent",
                                                {"action": "validate_input"}):
            validation_results = {
                "is_valid": True,
                "issues": [],
                "warnings": []
            }
            
            # Check for empty values
            if not state.column_values:
                validation_results["is_valid"] = False
                validation_results["issues"].append("No column values provided")
            
            # Check for too many empty/null values
            empty_values = sum(1 for v in state.column_values if not v or v.strip() == "")
            if empty_values > len(state.column_values) * 0.5:
                validation_results["warnings"].append(f"High empty value rate: {empty_values}/{len(state.column_values)}")
            
            # Check column name
            if not state.column_name or state.column_name.strip() == "":
                validation_results["warnings"].append("Column name is empty")
            
            agent_monitor.log_decision(
                request_id, "SupervisorAgent", "validation",
                validation_results,
                f"Input validation: {'passed' if validation_results['is_valid'] else 'failed'}"
            )
            
            return {
                "metadata": {**state.metadata, "validation": validation_results}
            }
    
    def should_continue_after_validation(self, state: AgentState) -> str:
        """Decide whether to continue after validation"""
        validation = state.metadata.get("validation", {})
        return "continue" if validation.get("is_valid", False) else "reject"
    
    async def plan_execution(self, state: AgentState) -> Dict[str, Any]:
        """Plan the execution strategy"""
        request_id = state.metadata["request_id"]
        
        with agent_monitor.track_agent_execution(request_id, "SupervisorAgent",
                                                {"action": "plan_execution"}):
            # Create execution plan
            execution_plan = {
                "phase_1": "column_analysis",
                "phase_2": "planning", 
                "phase_3": "candidate_retrieval",
                "phase_4": "disambiguation",
                "estimated_duration": len(state.column_values) * 0.5,  # 0.5s per value estimate
                "parallel_processing": len(state.column_values) > 10
            }
            
            agent_monitor.log_decision(
                request_id, "SupervisorAgent", "execution_planning",
                execution_plan,
                f"Planned execution for {len(state.column_values)} values"
            )
            
            return {
                "metadata": {**state.metadata, "execution_plan": execution_plan}
            }
    
    async def monitor_progress(self, state: AgentState) -> Dict[str, Any]:
        """Monitor execution progress"""
        request_id = state.metadata["request_id"]
        
        # Calculate progress metrics
        total_values = len(state.column_values)
        processed_values = len(state.candidates) if state.candidates else 0
        progress_percentage = (processed_values / total_values * 100) if total_values > 0 else 0
        
        progress_report = {
            "progress_percentage": progress_percentage,
            "processed_count": processed_values,
            "total_count": total_values,
            "current_phase": "monitoring",
            "issues_detected": len(state.processing_errors) if state.processing_errors else 0
        }
        
        agent_monitor.log_decision(
            request_id, "SupervisorAgent", "progress_monitoring",
            progress_report,
            f"Progress: {progress_percentage:.1f}% complete"
        )
        
        return {
            "metadata": {**state.metadata, "progress": progress_report}
        }
    
    async def quality_check(self, state: AgentState) -> Dict[str, Any]:
        """Perform quality checks on results"""
        request_id = state.metadata["request_id"]
        
        with agent_monitor.track_agent_execution(request_id, "SupervisorAgent",
                                                {"action": "quality_check"}):
            
            quality_gates = state.metadata.get("quality_gates", {})
            
            # Calculate quality metrics
            if state.confidence_scores:
                avg_confidence = sum(state.confidence_scores.values()) / len(state.confidence_scores)
                successful_links = sum(1 for score in state.confidence_scores.values() 
                                     if score >= quality_gates.get("min_confidence", 0.6))
                success_rate = successful_links / len(state.column_values)
            else:
                avg_confidence = 0
                success_rate = 0
            
            quality_report = {
                "avg_confidence": avg_confidence,
                "success_rate": success_rate,
                "meets_confidence_threshold": avg_confidence >= quality_gates.get("min_confidence", 0.6),
                "meets_success_rate_threshold": success_rate >= quality_gates.get("min_success_rate", 0.7),
                "error_count": len(state.processing_errors) if state.processing_errors else 0,
                "overall_quality": "high" if (avg_confidence >= 0.8 and success_rate >= 0.8) else 
                                  "medium" if (avg_confidence >= 0.6 and success_rate >= 0.6) else "low"
            }
            
            agent_monitor.log_decision(
                request_id, "SupervisorAgent", "quality_assessment",
                quality_report,
                f"Quality: {quality_report['overall_quality']}, Success: {success_rate:.1%}"
            )
            
            return {
                "metadata": {**state.metadata, "quality_report": quality_report}
            }
    
    def should_reprocess(self, state: AgentState) -> str:
        """Decide whether to approve, reprocess, or fail"""
        quality_report = state.metadata.get("quality_report", {})
        
        if quality_report.get("overall_quality") == "high":
            return "approve"
        elif quality_report.get("overall_quality") == "medium":
            # Could implement retry logic here
            return "approve"  # For now, accept medium quality
        else:
            return "fail"
    
    async def finalize_results(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and package results"""
        request_id = state.metadata["request_id"]
        
        final_report = {
            "completion_time": datetime.now().isoformat(),
            "supervision_summary": {
                "total_agents_used": len(set(event.agent_name for event in agent_monitor.events 
                                           if event.request_id == request_id)),
                "total_events": len([e for e in agent_monitor.events if e.request_id == request_id]),
                "quality_score": state.metadata.get("quality_report", {}).get("overall_quality", "unknown")
            }
        }
        
        agent_monitor.log_decision(
            request_id, "SupervisorAgent", "finalization",
            final_report,
            "Request processing completed successfully"
        )
        
        return {
            "metadata": {**state.metadata, "final_report": final_report}
        }
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the supervision workflow"""
        return await self.graph.ainvoke(state)