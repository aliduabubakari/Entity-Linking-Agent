from .lamapi_gateway import LamAPIGateway
from .geonames_gateway import GeoNamesGateway
from .alligator_gateway import AlligatorGateway
from .sparql_gateway import SPARQLGateway
from domain.entities import KnowledgeBaseConfig
from domain.interfaces import KnowledgeBaseGateway

class KnowledgeBaseFactory:
    """Factory to create appropriate knowledge base gateways"""
    
    @staticmethod
    def create_gateway(config: KnowledgeBaseConfig) -> KnowledgeBaseGateway:
        if config.type == "lamapi":
            return LamAPIGateway(config)
        elif config.type == "geonames":
            return GeoNamesGateway(config)
        elif config.type == "alligator":
            return AlligatorGateway(config)
        elif config.type == "sparql":
            return SPARQLGateway(config)
        else:
            raise ValueError(f"Unsupported knowledge base type: {config.type}")