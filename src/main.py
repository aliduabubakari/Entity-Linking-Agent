import asyncio
import logging
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager

from config.settings import settings
from interface.schemas import AgentState, EntityLinkingRequest, EntityLinkingResponse, SystemHealth
from interface.agents import ColumnAnalystAgent, CandidateRetrieverAgent, DisambiguationAgent
from interface.agents.supervisor_agent import SupervisorAgent
from interface.agents.planning_agent import PlanningAgent
from infrastructure import create_llm_service
from infrastructure.monitoring import agent_monitor
from domain.entities import KnowledgeBaseConfig
from infrastructure.knowledge_bases import KnowledgeBaseFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
processing_requests = {}

def update_agent_state(current_state: AgentState, updates: dict) -> AgentState:
    """Helper function to properly update AgentState"""
    # Get current state as dict
    if isinstance(current_state, dict):
        state_dict = current_state
    else:
        state_dict = current_state.model_dump()
    
    # Apply updates
    for key, value in updates.items():
        if key == "metadata":
            # Merge metadata dictionaries
            state_dict["metadata"] = {**state_dict.get("metadata", {}), **value}
        elif key == "processing_errors":
            # Append to existing errors
            existing_errors = state_dict.get("processing_errors", [])
            if isinstance(value, list):
                state_dict["processing_errors"] = existing_errors + value
            else:
                state_dict["processing_errors"] = existing_errors + [value]
        else:
            state_dict[key] = value
    
    # Return new AgentState object
    return AgentState(**state_dict)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Entity Linking Agentic System")
    
    # Initialize LLM service
    app.state.llm_service = create_llm_service()
    logger.info("✅ LLM service initialized")
    
    # Initialize all agents (including supervisor and planning)
    app.state.supervisor_agent = SupervisorAgent(app.state.llm_service)
    app.state.planning_agent = PlanningAgent(app.state.llm_service)
    app.state.column_analyst = ColumnAnalystAgent(app.state.llm_service)
    app.state.candidate_retriever = CandidateRetrieverAgent()
    app.state.disambiguation_agent = DisambiguationAgent(app.state.llm_service)
    
    logger.info("✅ All agents initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Entity Linking Agentic System")

app = FastAPI(
    title="Entity Linking Agentic System",
    description="Multi-agent system for entity linking in tables using LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

async def process_entity_linking_with_supervision(request_id: str, request: EntityLinkingRequest):
    """Enhanced background task with supervision"""
    try:
        processing_requests[request_id] = {
            "status": "processing",
            "started_at": datetime.now(),
            "request": request,
            "current_phase": "initialization"
        }
        
        # Initialize agent state with supervision metadata
        state = AgentState(
            column_name=request.column_name,
            column_values=request.column_values,
            table_context=request.table_context or {},
            metadata={
                "request_id": request_id,
                "supervision_enabled": True,
                "start_time": datetime.now().isoformat()
            }
        )
        
        # Phase 1: Supervision & Planning
        processing_requests[request_id]["current_phase"] = "supervision"
        supervisor_result = await app.state.supervisor_agent.execute(state)
        state = update_agent_state(state, supervisor_result)
        
        processing_requests[request_id]["current_phase"] = "planning"
        planning_result = await app.state.planning_agent.execute(state)
        state = update_agent_state(state, planning_result)
        
        # Phase 2: Core Processing
        processing_requests[request_id]["current_phase"] = "column_analysis"
        analysis_result = await app.state.column_analyst.execute(state)
        state = update_agent_state(state, analysis_result)
        
        processing_requests[request_id]["current_phase"] = "candidate_retrieval"
        retrieval_result = await app.state.candidate_retriever.execute(state)
        state = update_agent_state(state, retrieval_result)
        
        processing_requests[request_id]["current_phase"] = "disambiguation"
        disambiguation_result = await app.state.disambiguation_agent.execute(state)
        state = update_agent_state(state, disambiguation_result)
        
        # Prepare results with enhanced metrics
        execution_timeline = agent_monitor.get_execution_timeline(request_id)
        agent_stats = agent_monitor.get_agent_performance_stats()
        
        results = {
            "column_name": state.column_name,
            "column_type": state.column_type,
            "linking_results": state.disambiguated_results,
            "confidence_scores": state.confidence_scores,
            "processing_metrics": {
                "total_mentions": len(state.column_values),
                "successful_links": sum(1 for score in state.confidence_scores.values() if score > 0.6) if state.confidence_scores else 0,
                "average_confidence": sum(state.confidence_scores.values()) / len(state.confidence_scores) if state.confidence_scores else 0,
                "total_events": len(execution_timeline),
                "agent_performance": agent_stats,
                "supervision_metadata": state.metadata.get("final_report", {})
            },
            "execution_timeline": execution_timeline
        }
        
        processing_requests[request_id].update({
            "status": "completed",
            "completed_at": datetime.now(),
            "results": results,
            "errors": state.processing_errors or [],
            "current_phase": "completed"
        })
        
        logger.info(f"✅ Successfully processed request {request_id}")
        
    except Exception as e:
        logger.error(f"Processing failed for request {request_id}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        processing_requests[request_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "errors": [str(e)],
            "current_phase": "failed"
        })

@app.post("/api/entity-linking", response_model=EntityLinkingResponse)
async def entity_linking(request: EntityLinkingRequest, background_tasks: BackgroundTasks):
    """Enhanced endpoint with supervision"""
    request_id = str(uuid.uuid4())
    
    logger.info(f"🚀 Starting entity linking request: {request_id}")
    logger.info(f"Column: {request.column_name}, Values: {request.column_values}")
    
    background_tasks.add_task(process_entity_linking_with_supervision, request_id, request)
    
    return EntityLinkingResponse(
        request_id=request_id,
        status="processing",
        created_at=datetime.now()
    )

@app.get("/api/entity-linking/{request_id}", response_model=EntityLinkingResponse)
async def get_entity_linking_result(request_id: str):
    """Enhanced endpoint with detailed monitoring"""
    if request_id not in processing_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request_data = processing_requests[request_id]
    
    # Add real-time monitoring data
    response_data = {
        "request_id": request_id,
        "status": request_data["status"],
        "current_phase": request_data.get("current_phase", "unknown"),
        "results": request_data.get("results"),
        "errors": request_data.get("errors"),
        "created_at": request_data["started_at"],
        "completed_at": request_data.get("completed_at")
    }
    
    # Add execution timeline if available
    if request_data["status"] == "completed" and request_data.get("results"):
        response_data["execution_timeline"] = request_data["results"].get("execution_timeline", [])
    
    return EntityLinkingResponse(**response_data)

@app.get("/api/monitoring/stats")
async def get_monitoring_stats():
    """Get agent performance statistics"""
    return {
        "agent_performance": agent_monitor.get_agent_performance_stats(),
        "total_events": len(agent_monitor.events),
        "active_requests": len([r for r in processing_requests.values() if r["status"] == "processing"]),
        "recent_events": agent_monitor.events[-10:] if len(agent_monitor.events) > 10 else agent_monitor.events
    }

@app.get("/api/monitoring/timeline/{request_id}")
async def get_request_timeline(request_id: str):
    """Get detailed execution timeline for a request"""
    timeline = agent_monitor.get_execution_timeline(request_id)
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not found")
    
    return {
        "request_id": request_id,
        "timeline": timeline,
        "summary": {
            "total_events": len(timeline),
            "agents_involved": len(set(event.get("agent_name") for event in timeline)),
            "total_duration": sum(event.get("duration_ms", 0) for event in timeline if event.get("duration_ms"))
        }
    }

@app.get("/api/health", response_model=SystemHealth)
async def health_check():
    """Enhanced health check with agent monitoring"""
    # Check knowledge base statuses (existing code)
    kb_statuses = []
    kb_configs = [KnowledgeBaseConfig(**kb) for kb in settings.knowledge_bases_config["knowledge_bases"]]
    
    for config in kb_configs:
        if config.enabled:
            try:
                gateway = KnowledgeBaseFactory.create_gateway(config)
                health_status = await gateway.health_check()
                kb_statuses.append({
                    "name": config.name,
                    "type": config.type,
                    "enabled": config.enabled,
                    "priority": config.priority,
                    "health_status": health_status,
                    "last_checked": datetime.now()
                })
            except Exception as e:
                logger.error(f"Health check failed for {config.name}: {e}")
                kb_statuses.append({
                    "name": config.name,
                    "type": config.type,
                    "enabled": config.enabled,
                    "priority": config.priority,
                    "health_status": False,
                    "last_checked": datetime.now()
                })
    
    # Add agent performance data
    agent_stats = agent_monitor.get_agent_performance_stats()
    
    return SystemHealth(
        status="healthy",
        knowledge_bases=kb_statuses,
        llm_available=True,
        cache_enabled=settings.USE_CACHE,
        total_processed=len(processing_requests),
        uptime=0,  # Would calculate actual uptime
        agent_performance=agent_stats
    )

@app.get("/")
async def root():
    """Root endpoint with enhanced info"""
    return {
        "message": "Entity Linking Agentic System",
        "version": "1.0.0",
        "features": [
            "Multi-agent supervision",
            "Strategic planning", 
            "Real-time monitoring",
            "Comprehensive auditing"
        ],
        "documentation": "/docs",
        "monitoring": "/api/monitoring/stats"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=settings.API_WORKERS
    )