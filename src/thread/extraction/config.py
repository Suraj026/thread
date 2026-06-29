"""Configuration for the Thread extraction pipeline.

All settings can be overridden via environment variables with the THREAD_ prefix.
Example: THREAD_OPENROUTER_API_KEY=sk-... THREAD_OPENROUTER_MODEL_NAME=...
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for the extraction pipeline.

    Uses Pydantic BaseSettings with environment variable prefix THREAD_.
    """

    model_config = {"env_prefix": "THREAD_", "populate_by_name": True}

    # OpenRouter configuration
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_model_name: str = Field(
        default="openrouter/qwen/qwen3-coder:free",
        description="Default model for entity extraction",
    )
    openrouter_sort: str = Field(
        default="price",
        description="Model selection strategy: 'price' for cost-effectiveness",
    )

    # Extraction parameters
    extraction_max_tokens: int = Field(default=4096, ge=1, le=32000)
    extraction_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    extraction_max_input_length: int = Field(default=20000, ge=100, le=100000)
    extraction_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    extraction_retry_count: int = Field(default=2, ge=0, le=10)

    # Fallback model for ambiguous or failed extractions
    extraction_fallback_model: str = Field(
        default="openrouter/google/gemma-4-31b-it:free",
        description="Fallback model when primary extraction fails",
    )