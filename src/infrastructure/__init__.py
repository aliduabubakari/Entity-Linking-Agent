from typing import List
from domain.interfaces import KnowledgeBaseGateway
from domain.entities import KnowledgeBaseConfig
from .knowledge_bases import KnowledgeBaseFactory
from .llm.llm_service import AzureOpenAILanguageModel, LLMColumnAnalysisService, LLMDisambiguationService, LLMValidationService

def create_knowledge_base_gateways(kb_configs: List[KnowledgeBaseConfig]) -> List[KnowledgeBaseGateway]:
    """Create all enabled knowledge base gateways"""
    gateways = []
    for config in kb_configs:
        if config.enabled:
            try:
                gateway = KnowledgeBaseFactory.create_gateway(config)
                gateways.append(gateway)
            except Exception as e:
                print(f"Failed to create gateway for {config.name}: {e}")
    return sorted(gateways, key=lambda g: g.get_config().priority)

def create_llm_service(model: str = None, temperature: float = 0.1):
    """Create LLM service instance"""
    return AzureOpenAILanguageModel(model=model, temperature=temperature)

def create_column_analysis_service(llm: AzureOpenAILanguageModel):
    """Create column analysis service"""
    return LLMColumnAnalysisService(llm)

def create_disambiguation_service(llm: AzureOpenAILanguageModel):
    """Create disambiguation service"""
    return LLMDisambiguationService(llm)

def create_validation_service(llm: AzureOpenAILanguageModel):
    """Create validation service"""
    return LLMValidationService(llm)