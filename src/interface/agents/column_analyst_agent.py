from typing import Dict, Any, List
from langgraph.graph import StateGraph, END  # Add END import
from domain.entities import TableColumn, ColumnType
from interface.schemas import AgentState
from infrastructure.llm.llm_service import AzureOpenAILanguageModel
from infrastructure.monitoring import agent_monitor
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ColumnAnalystAgent:
    """Agent responsible for analyzing table columns and extracting context"""
    
    def __init__(self, llm_service: AzureOpenAILanguageModel):
        self.llm = llm_service
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph for column analysis"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("analyze_column_type", self.analyze_column_type)
        workflow.add_node("extract_context", self.extract_context)
        workflow.add_node("validate_analysis", self.validate_analysis)
        
        # Define edges
        workflow.set_entry_point("analyze_column_type")
        workflow.add_edge("analyze_column_type", "extract_context")
        workflow.add_edge("extract_context", "validate_analysis")
        workflow.add_edge("validate_analysis", END)  # Now END is imported
        
        return workflow.compile()
    
    async def analyze_column_type(self, state: AgentState) -> Dict[str, Any]:
        """Analyze column type using LLM"""
        request_id = state.metadata.get("request_id", "unknown")
        
        with agent_monitor.track_agent_execution(request_id, "ColumnAnalystAgent", 
                                               {"action": "analyze_column_type", 
                                                "column_name": state.column_name}):
            try:
                # Simple heuristic-based analysis for now
                column_type = self._infer_column_type_heuristic(state.column_values, state.column_name)
                
                agent_monitor.log_decision(
                    request_id, "ColumnAnalystAgent", "type_inference",
                    {"inferred_type": column_type, "confidence": 0.8},
                    f"Inferred type {column_type} based on column analysis"
                )
                
                return {
                    "column_type": column_type,
                    "metadata": {
                        **state.metadata,
                        "analysis_method": "heuristic_inference",
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                }
                
            except Exception as e:
                logger.error(f"Column analysis failed: {e}")
                return {
                    "column_type": ColumnType.UNKNOWN.value,
                    "processing_errors": [f"Column analysis error: {str(e)}"],
                    "metadata": {**state.metadata, "analysis_failed": True}
                }
    
    def _infer_column_type_heuristic(self, values: List[str], column_name: str) -> str:
        """Simple heuristic-based column type inference"""
        column_name_lower = column_name.lower()
        
        # Check column name for hints
        if any(keyword in column_name_lower for keyword in ['city', 'location', 'place', 'country', 'state']):
            return ColumnType.LOCATION.value
        elif any(keyword in column_name_lower for keyword in ['person', 'name', 'author', 'scientist', 'people']):
            return ColumnType.PERSON.value
        elif any(keyword in column_name_lower for keyword in ['company', 'organization', 'org', 'institution']):
            return ColumnType.ORGANIZATION.value
        elif any(keyword in column_name_lower for keyword in ['movie', 'film', 'book', 'song', 'work', 'title']):
            return ColumnType.WORK.value
        
        # Check values for patterns
        sample_values = [v.strip() for v in values[:5] if v.strip()]
        
        # Look for location patterns
        if any(len(v.split()) <= 2 and v.istitle() for v in sample_values):
            # Check if they look like place names
            place_indicators = ['paris', 'london', 'berlin', 'tokyo', 'rome', 'madrid', 'moscow']
            if any(v.lower() in place_indicators for v in sample_values):
                return ColumnType.LOCATION.value
        
        # Look for person patterns (often have multiple words, proper case)
        person_indicators = ['einstein', 'newton', 'tesla', 'darwin', 'shakespeare']
        if any(any(indicator in v.lower() for indicator in person_indicators) for v in sample_values):
            return ColumnType.PERSON.value
        
        # Default to LITERAL for mixed content
        return ColumnType.LITERAL.value
    
    async def extract_context(self, state: AgentState) -> Dict[str, Any]:
        """Extract table context for disambiguation"""
        request_id = state.metadata.get("request_id", "unknown")
        
        with agent_monitor.track_agent_execution(request_id, "ColumnAnalystAgent",
                                               {"action": "extract_context"}):
            try:
                if not state.table_context:
                    # Create enhanced table context
                    table_context = {
                        "headers": [state.column_name],
                        "sample_rows": [{state.column_name: value} for value in state.column_values[:3]],
                        "inferred_semantics": {
                            "column_type": state.column_type,
                            "domain": self._infer_domain(state.column_values, state.column_name)
                        },
                        "statistics": {
                            "total_values": len(state.column_values),
                            "unique_values": len(set(state.column_values)),
                            "empty_values": sum(1 for v in state.column_values if not v.strip())
                        }
                    }
                    
                    agent_monitor.log_decision(
                        request_id, "ColumnAnalystAgent", "context_extraction",
                        {"context_created": True, "domain": table_context["inferred_semantics"]["domain"]},
                        "Enhanced table context created"
                    )
                    
                    return {"table_context": table_context}
                
                return {"table_context": state.table_context}
                
            except Exception as e:
                logger.error(f"Context extraction failed: {e}")
                return {
                    "table_context": {},
                    "processing_errors": [f"Context extraction error: {str(e)}"]
                }
    
    def _infer_domain(self, values: List[str], column_name: str) -> str:
        """Infer domain from values and column name"""
        column_lower = column_name.lower()
        
        # Domain keywords
        if any(kw in column_lower for kw in ['city', 'country', 'location', 'place']):
            return 'geography'
        elif any(kw in column_lower for kw in ['scientist', 'person', 'author', 'researcher']):
            return 'science'
        elif any(kw in column_lower for kw in ['movie', 'film', 'book', 'entertainment']):
            return 'entertainment'
        elif any(kw in column_lower for kw in ['company', 'business', 'organization']):
            return 'business'
        
        return 'general'
    
    async def validate_analysis(self, state: AgentState) -> Dict[str, Any]:
        """Validate the column analysis results"""
        request_id = state.metadata.get("request_id", "unknown")
        
        # Simple validation
        is_valid = state.column_type and state.column_type != ColumnType.UNKNOWN.value
        
        agent_monitor.log_decision(
            request_id, "ColumnAnalystAgent", "validation",
            {"is_valid": is_valid, "column_type": state.column_type},
            f"Analysis validation: {'passed' if is_valid else 'failed'}"
        )
        
        return {
            "metadata": {
                **state.metadata,
                "analysis_validated": is_valid,
                "validation_timestamp": datetime.now().isoformat()
            }
        }
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the column analysis workflow"""
        return await self.graph.ainvoke(state)