"""Base extractor class with Pydantic AI harness and confidence scoring.

This module defines:
- BaseExtractor: Abstract base class for all extractors
- AIHarness: Typed wrapper around Pydantic AI Agent for structured LLM extraction
"""

import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

from thread.extraction.config import Settings
from thread.extraction.models import BaseEntity

logger = logging.getLogger(__name__)

EntityT = TypeVar("EntityT", bound=BaseEntity)


class AIHarness(Generic[EntityT]):
    """Typed wrapper around Pydantic AI Agent for structured entity extraction."""

    def __init__(self, entity_type: type[EntityT], settings: Settings):
        """Initialize the harness with an entity type and settings.

        Args:
            entity_type: The Pydantic model class to extract (e.g., GOAL, DECISION).
            settings: Application settings with model configuration.
        """
        self.entity_type = entity_type
        self.settings = settings
        self._agent: Optional[object] = None  # Pydantic AI Agent — lazy init

    @property
    def entity_type_name(self) -> str:
        """Return the entity type class name (e.g., 'GOAL', 'DECISION')."""
        return self.entity_type.__name__

    def _build_system_prompt(self) -> str:
        """Build a system prompt describing the entity to extract.

        Returns:
            A string prompt describing required fields and output format.
        """
        field_descriptions = []
        for field_name, field_info in self.entity_type.model_fields.items():
            if field_name == "confidence":
                continue  # Skip extraction metadata
            field_type = field_info.annotation
            desc = field_info.description or ""
            required = "REQUIRED" if field_info.is_required() else "optional"
            field_descriptions.append(f"  - {field_name} ({field_type}): {desc} [{required}]")

        fields_str = "\n".join(field_descriptions)

        return (
            f"You are an expert SDLC entity extractor. Your task is to extract a "
            f"{self.entity_type_name} entity from the provided text.\n\n"
            f"Fields to populate:\n{fields_str}\n\n"
            f"Return ONLY a valid JSON object matching the {self.entity_type_name} schema. "
            f"Do not include any explanation or markdown formatting. "
            f"If the text does not contain enough information, set confidence to a low value."
        )

    def run(self, text: str) -> Optional[EntityT]:
        """Run extraction on a single text input.

        Args:
            text: Natural language text to extract from.

        Returns:
            Extracted entity or None if extraction failed.
        """
        try:
            from pydantic_ai import Agent
            from pydantic_ai.exceptions import AgentRunError

            # Truncate input if necessary
            truncated = text[: self.settings.extraction_max_input_length]

            # Lazy-init the agent with the model and output type
            if self._agent is None:
                self._agent = Agent(
                    model=self.settings.openrouter_model_name,
                    output_type=self.entity_type,
                    model_settings={
                        "max_tokens": self.settings.extraction_max_tokens,
                        "temperature": self.settings.extraction_temperature,
                    },
                    system_prompt=self._build_system_prompt(),
                )

            # Run extraction with retry logic
            last_error = None
            for attempt in range(self.settings.extraction_retry_count + 1):
                try:
                    result = self._agent.run_sync(truncated)
                    entity = result.data

                    # Set the extraction confidence — LLM returned valid structured data
                    entity.confidence = 1.0
                    return entity

                except AgentRunError as e:
                    last_error = e
                    logger.warning(
                        "Extraction attempt %d/%d failed for %s: %s",
                        attempt + 1,
                        self.settings.extraction_retry_count + 1,
                        self.entity_type_name,
                        e,
                    )
                    if attempt < self.settings.extraction_retry_count:
                        continue

            logger.error(
                "All extraction attempts failed for %s: %s",
                self.entity_type_name,
                last_error,
            )
            return None

        except Exception as e:
            logger.exception("Unexpected error in AIHarness.run(): %s", e)
            return None

    def run_batch(self, texts: list[str]) -> list[Optional[EntityT]]:
        """Run extraction on multiple text inputs.

        Args:
            texts: List of natural language texts.

        Returns:
            List of extracted entities (None for failures).
        """
        return [self.run(text) for text in texts]


class BaseExtractor(ABC):
    """Abstract base class for all entity extractors.

    Subclasses implement extract() to dispatch to appropriate AIHarness instances
    for their SDLC phase group.

    All extractors share this base class interface.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the extractor with optional settings.

        Args:
            settings: Application settings. Creates default if None.
        """
        self.settings = settings or Settings()
        self._harnesses: dict[str, AIHarness] = {}

    def _create_harness(self, entity_type: type[BaseEntity]) -> AIHarness:
        """Create or retrieve a cached AIHarness for the given entity type.

        Args:
            entity_type: The Pydantic model class.

        Returns:
            AIHarness configured for this entity type.
        """
        type_name = entity_type.__name__
        if type_name not in self._harnesses:
            self._harnesses[type_name] = AIHarness(entity_type, self.settings)
        return self._harnesses[type_name]

    @abstractmethod
    def extract(self, text: str) -> list[BaseEntity]:
        """Extract entities from natural language text.

        Args:
            text: Natural language text to extract entities from.

        Returns:
            List of extracted entities with confidence scores populated.
        """
        ...

    @staticmethod
    def _score_confidence(
        instance: Optional[BaseModel],
        llm_confidence: float,
        extracted_fields: dict,
        entity_certainty: float,
        required_keys: Optional[set[str]] = None,
    ) -> float:
        """Compute 3-signal confidence score.

        Signals:
        1. LLM self-report confidence (from the model's output)
        2. Field completeness ratio (non-None required fields / total required)
        3. Entity type certainty (classification confidence from pre-extraction)

        Args:
            instance: The extracted entity instance (or None).
            llm_confidence: LLM's self-reported confidence (0.0-1.0).
            extracted_fields: Dict of extracted field name -> value.
            entity_certainty: Entity type classification certainty (0.0-1.0).
            required_keys: Set of field names considered required. If None,
                          computed from the entity type's required fields.

        Returns:
            Clamped confidence score in [0.0, 1.0].
        """
        # Signal 2: Field completeness
        if required_keys is not None and extracted_fields is not None:
            field_completeness = BaseExtractor.confidence_field_completeness(
                extracted_fields, required_keys
            )
        else:
            field_completeness = 1.0

        # Average the 3 signals
        raw = (llm_confidence + field_completeness + entity_certainty) / 3.0

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, raw))

    @staticmethod
    def confidence_field_completeness(fields: dict, required_keys: set[str]) -> float:
        """Calculate what fraction of required fields have non-None, non-empty values.

        Args:
            fields: Dict mapping field name to value.
            required_keys: Set of field names considered required.

        Returns:
            Float in [0.0, 1.0] representing field completeness.
        """
        if not required_keys:
            return 1.0

        present = sum(
            1 for key in required_keys
            if fields.get(key) is not None and fields.get(key) != ""
        )
        return present / len(required_keys)

    @staticmethod
    def _truncate_input(text: str, max_length: Optional[int] = None) -> str:
        """Truncate text at word boundary to max_length.

        Args:
            text: Input text to truncate.
            max_length: Maximum character length. If None, uses 10000.

        Returns:
            Truncated text ending with '...' if truncation occurred.
        """
        if max_length is None:
            max_length = 10000

        if len(text) <= max_length:
            return text

        # Truncate at last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")

        if last_space > 0:
            return text[:last_space] + "..."
        else:
            return truncated + "..."

    @staticmethod
    def _build_extraction_prompt(
        entity_type_name: str, text: str
    ) -> list[dict[str, str]]:
        """Build structured messages for LLM extraction.

        Args:
            entity_type_name: Name of the entity type (e.g., 'GOAL').
            text: Input text to extract from.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return [
            {
                "role": "system",
                "content": (
                    f"Extract a {entity_type_name} entity from the following text. "
                    f"Return only valid JSON matching the {entity_type_name} schema."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]
