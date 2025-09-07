from typing import Dict, Any, List
from datetime import datetime  # Add this line
from langgraph.graph import StateGraph, END
from interface.schemas import AgentState
from infrastructure.monitoring.agent_monitor import agent_monitor
from infrastructure.llm.llm_service import AzureOpenAILanguageModel
from config.settings import knowledge_bases_config
from domain.entities import KnowledgeBaseConfig, ColumnType
import logging

logger = logging.getLogger(__name__)

class PlanningAgent:
    """Strategic planning agent that decides which tools and KBs to use"""
    
    def __init__(self, llm_service: AzureOpenAILanguageModel):
        self.llm = llm_service
        self.available_kbs = self._load_available_kbs()
        self.graph = self._build_graph()
    
    def _load_available_kbs(self) -> List[KnowledgeBaseConfig]:
        """Load available knowledge bases"""
        return [KnowledgeBaseConfig(**kb) for kb in knowledge_bases_config["knowledge_bases"]]
    
    def _build_graph(self):
        """Build the planning workflow"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze_requirements", self.analyze_requirements)
        workflow.add_node("select_knowledge_bases", self.select_knowledge_bases)
        workflow.add_node("plan_processing_strategy", self.plan_processing_strategy)
        workflow.add_node("optimize_execution_order", self.optimize_execution_order)
        workflow.add_node("create_execution_plan", self.create_execution_plan)
        
        workflow.set_entry_point("analyze_requirements")
        workflow.add_edge("analyze_requirements", "select_knowledge_bases")
        workflow.add_edge("select_knowledge_bases", "plan_processing_strategy")
        workflow.add_edge("plan_processing_strategy", "optimize_execution_order")
        workflow.add_edge("optimize_execution_order", "create_execution_plan")
        workflow.add_edge("create_execution_plan", "__end__")
        
        return workflow.compile()
    
    async def analyze_requirements(self, state: AgentState) -> Dict[str, Any]:
        """Analyze requirements and constraints"""
        request_id = state.metadata.get("request_id", "unknown")
        
        with agent_monitor.track_agent_execution(request_id, "PlanningAgent",
                                                {"action": "analyze_requirements"}):
            
            requirements = {
                "column_type": state.column_type,
                "data_size": len(state.column_values),
                "complexity": self._assess_complexity(state),
                "domain": state.table_context.get("domain", "unknown"),
                "performance_requirements": self._assess_performance_needs(state),
                "quality_requirements": {
                    "min_confidence": 0.7,
                    "precision_over_recall": True
                }
            }
            
            agent_monitor.log_decision(
                request_id, "PlanningAgent", "requirements_analysis",
                requirements,
                f"Analyzed requirements for {requirements['data_size']} values"
            )
            
            return {
                "metadata": {**state.metadata, "requirements": requirements}
            }
    
    def _assess_complexity(self, state: AgentState) -> str:
        """Assess processing complexity"""
        data_size = len(state.column_values)
        has_context = bool(state.table_context)
        
        if data_size > 100:
            return "high"
        elif data_size > 20 or has_context:
            return "medium"
        else:
            return "low"
    
    def _assess_performance_needs(self, state: AgentState) -> Dict[str, Any]:
        """Assess performance requirements"""
        data_size = len(state.column_values)
        
        return {
            "max_processing_time": min(60, data_size * 0.5),  # Max 60s or 0.5s per item
            "prefer_speed": data_size > 50,
            "prefer_quality": data_size <= 10,
            "enable_parallel": data_size > 20
        }
    
    async def select_knowledge_bases(self, state: AgentState) -> Dict[str, Any]:
        """Select optimal knowledge bases"""
        request_id = state.metadata.get("request_id", "unknown")
        requirements = state.metadata.get("requirements", {})
        
        with agent_monitor.track_agent_execution(request_id, "PlanningAgent",
                                                {"action": "select_knowledge_bases"}):
            
            selected_kbs = []
            column_type = requirements.get("column_type")
            domain = requirements.get("domain", "unknown")
            
            # Score each KB
            kb_scores = {}
            for kb_config in self.available_kbs:
                if not kb_config.enabled:
                    continue
                
                score = self._score_knowledge_base(kb_config, column_type, domain, requirements)
                kb_scores[kb_config.name] = score
                
                if score > 0.3:  # Threshold for selection
                    selected_kbs.append({
                        "name": kb_config.name,
                        "type": kb_config.type,
                        "score": score,
                        "priority": kb_config.priority,
                        "expected_coverage": self._estimate_coverage(kb_config, column_type)
                    })
            
            # Sort by score and priority
            selected_kbs.sort(key=lambda x: (x["score"], -x["priority"]), reverse=True)
            
            selection_plan = {
                "selected_kbs": selected_kbs,
                "primary_kb": selected_kbs[0]["name"] if selected_kbs else None,
                "fallback_kbs": [kb["name"] for kb in selected_kbs[1:3]],  # Top 2 fallbacks
                "kb_scores": kb_scores
            }
            
            agent_monitor.log_decision(
                request_id, "PlanningAgent", "kb_selection",
                selection_plan,
                f"Selected {len(selected_kbs)} knowledge bases for {column_type}"
            )
            
            return {
                "metadata": {**state.metadata, "kb_selection": selection_plan}
            }
    
    def _score_knowledge_base(self, kb_config: KnowledgeBaseConfig, 
                             column_type: str, domain: str, requirements: Dict) -> float:
        """Score a knowledge base for the given requirements"""
        score = 0.0
        
        # Type compatibility (0-0.4)
        if column_type in [ct.value for ct in kb_config.supported_column_types]:
            score += 0.4
        
        # Domain-specific bonuses (0-0.2)
        if domain == "geography" and kb_config.type == "geonames":
            score += 0.2
        elif domain in ["science", "entertainment"] and kb_config.type == "lamapi":
            score += 0.2
        elif kb_config.type == "lamapi":  # General purpose bonus
            score += 0.1
        
        # Performance considerations (0-0.2)
        performance_needs = requirements.get("performance_requirements", {})
        if performance_needs.get("prefer_speed") and kb_config.type in ["lamapi", "geonames"]:
            score += 0.1
        
        # Priority adjustment (0-0.2)
        priority_bonus = max(0, (5 - kb_config.priority) * 0.04)  # Higher priority = lower number
        score += priority_bonus
        
        return min(1.0, score)
    
    def _estimate_coverage(self, kb_config: KnowledgeBaseConfig, column_type: str) -> float:
        """Estimate coverage for this KB and column type"""
        coverage_estimates = {
            ("lamapi", "PERSON"): 0.8,
            ("lamapi", "WORK"): 0.9,
            ("lamapi", "ORGANIZATION"): 0.7,
            ("geonames", "LOCATION"): 0.95,
            ("sparql", "PERSON"): 0.6,
            ("alligator", "PERSON"): 0.7
        }
        
        return coverage_estimates.get((kb_config.type, column_type), 0.5)
    
    async def plan_processing_strategy(self, state: AgentState) -> Dict[str, Any]:
        """Plan the processing strategy"""
        request_id = state.metadata.get("request_id", "unknown")
        requirements = state.metadata.get("requirements", {})
        
        strategy = {
            "batch_size": self._calculate_optimal_batch_size(requirements),
            "parallel_processing": requirements.get("performance_requirements", {}).get("enable_parallel", False),
            "retry_strategy": {
                "max_retries": 2,
                "retry_on_low_confidence": True,
                "fallback_enabled": True
            },
            "optimization_mode": "speed" if requirements.get("performance_requirements", {}).get("prefer_speed") else "quality"
        }
        
        agent_monitor.log_decision(
            request_id, "PlanningAgent", "strategy_planning",
            strategy,
            f"Planned {strategy['optimization_mode']} optimization strategy"
        )
        
        return {
            "metadata": {**state.metadata, "processing_strategy": strategy}
        }
    
    def _calculate_optimal_batch_size(self, requirements: Dict) -> int:
        """Calculate optimal batch size"""
        data_size = requirements.get("data_size", 10)
        complexity = requirements.get("complexity", "medium")
        
        if complexity == "high":
            return min(5, data_size)
        elif complexity == "medium":
            return min(10, data_size)
        else:
            return min(20, data_size)
    
    async def optimize_execution_order(self, state: AgentState) -> Dict[str, Any]:
        """Optimize execution order"""
        request_id = state.metadata.get("request_id", "unknown")
        kb_selection = state.metadata.get("kb_selection", {})
        
        # Create execution order based on scores and expected performance
        execution_order = {
            "phase_1": "column_analysis",  # Always first
            "phase_2": "candidate_retrieval",
            "phase_3": "disambiguation", 
            "phase_4": "validation",
            "kb_query_order": [kb["name"] for kb in kb_selection.get("selected_kbs", [])],
            "parallel_kb_queries": len(kb_selection.get("selected_kbs", [])) > 1
        }
        
        agent_monitor.log_decision(
            request_id, "PlanningAgent", "execution_optimization",
            execution_order,
            "Optimized execution order for maximum efficiency"
        )
        
        return {
            "metadata": {**state.metadata, "execution_order": execution_order}
        }
    
    async def create_execution_plan(self, state: AgentState) -> Dict[str, Any]:
        """Create comprehensive execution plan"""
        request_id = state.metadata.get("request_id", "unknown")
        
        execution_plan = {
            "plan_id": f"plan_{request_id}",
            "created_at": datetime.now().isoformat(),
            "requirements": state.metadata.get("requirements"),
            "kb_selection": state.metadata.get("kb_selection"),
            "processing_strategy": state.metadata.get("processing_strategy"),
            "execution_order": state.metadata.get("execution_order"),
            "estimated_duration": self._estimate_duration(state),
            "resource_allocation": {
                "cpu_intensive": False,
                "memory_intensive": len(state.column_values) > 100,
                "network_intensive": True
            }
        }
        
        agent_monitor.log_decision(
            request_id, "PlanningAgent", "plan_creation",
            {"plan_id": execution_plan["plan_id"], "estimated_duration": execution_plan["estimated_duration"]},
            "Comprehensive execution plan created"
        )
        
        return {
            "metadata": {**state.metadata, "execution_plan": execution_plan}
        }
    
    def _estimate_duration(self, state: AgentState) -> float:
        """Estimate total processing duration"""
        base_time = len(state.column_values) * 0.3  # 0.3s per value base
        complexity_multiplier = {
            "low": 1.0,
            "medium": 1.5, 
            "high": 2.0
        }
        
        requirements = state.metadata.get("requirements", {})
        complexity = requirements.get("complexity", "medium")
        
        return base_time * complexity_multiplier.get(complexity, 1.5)
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the planning workflow"""
        return await self.graph.ainvoke(state)