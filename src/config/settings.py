import json
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Entity Linking Agentic System"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # LLM settings
    LLM_PROVIDER: str = "azure"  # options: "openai", "anthropic", "azure", "local"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4000
    DISABLE_LLM: bool = False  # Add this field

    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
        default="https://socialstocks2.openai.azure.com/",
        description="Azure OpenAI endpoint URL"
    )
    AZURE_OPENAI_KEY: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key"
    )
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2024-08-01-preview",
        description="Azure OpenAI API version"
    )

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # Knowledge Base settings
    KNOWLEDGE_BASES_CONFIG_PATH: str = "config/knowledge_bases.json"

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.6
    REEVALUATION_THRESHOLD: float = 0.4

    # Processing settings
    MAX_CANDIDATES_PER_MENTION: int = 10
    BATCH_SIZE: int = 50
    REQUEST_TIMEOUT: int = 30

    # Cache settings
    USE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour

    # Validation settings
    ENABLE_LLM_VALIDATION: bool = True
    ENABLE_AUTO_CORRECTION: bool = True

    model_config = {
        "env_file": ".env",  # load secrets like AZURE_OPENAI_KEY from here
        "case_sensitive": True,
    }

    @field_validator("KNOWLEDGE_BASES_CONFIG_PATH")
    def validate_config_path(cls, v: str) -> str:
        # Try to find the config file relative to the current working directory
        config_path = Path(v)
        if not config_path.exists():
            # Try relative to the src directory
            src_path = Path(__file__).parent / "knowledge_bases.json"
            if src_path.exists():
                return str(src_path)
            else:
                raise ValueError(f"Knowledge bases config file not found: {v}")
        return v


# Utility function: Load knowledge bases configuration
def load_knowledge_bases_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load knowledge bases config: {e}")


# Create settings instance
settings = Settings()

# Load knowledge bases
knowledge_bases_config = load_knowledge_bases_config(settings.KNOWLEDGE_BASES_CONFIG_PATH)