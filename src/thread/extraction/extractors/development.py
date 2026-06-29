"""DevelopmentExtractor — extracts development-phase entities.

Handles: IMPLEMENTATION, PR, COMMIT.
"""

from thread.extraction.base import BaseExtractor
from thread.extraction.models import COMMIT, IMPLEMENTATION, PR, BaseEntity
from thread.extraction.extractors.planning import (
    _TYPE_PRIORITY,
    _classify_entity_types,
)


class DevelopmentExtractor(BaseExtractor):
    """Extracts development-phase entities from natural language text.

    Detects and extracts IMPLEMENTATION, PR, and COMMIT entities.
    Shares classification keywords and priority ordering with
    PlanningExtractor for consistency.

    Attributes:
        GROUP_NAME:  Group identifier for registry lookup.
    """

    GROUP_NAME = "development"
    _ENTITY_TYPES = [IMPLEMENTATION, PR, COMMIT]

    def extract(self, text: str) -> list[BaseEntity]:
        """Extract development entities from natural language text.

        Pipeline matches PlanningExtractor.extract():
        1. Keyword classification
        2. AIHarness dispatch
        3. 3-signal confidence scoring (D-09)
        4. Priority sorting (D-10)

        Args:
            text: Natural language text.

        Returns:
            List of extracted entities with confidence >= threshold.
        """
        if not text or not text.strip():
            return []

        classification = _classify_entity_types(text)
        results: list[BaseEntity] = []

        for entity_type in self._ENTITY_TYPES:
            type_name = entity_type.__name__
            certainty = classification.get(type_name, 0.0)
            if certainty <= 0.0:
                continue

            harness = self._create_harness(entity_type)
            entity = harness.run(text)
            if entity is not None:
                entity_dict = (
                    entity.model_dump()
                    if hasattr(entity, "model_dump")
                    else {}
                )
                required_keys = {
                    field_name
                    for field_name, field_info in entity_type.model_fields.items()
                    if field_info.is_required()
                }
                entity.confidence = self._score_confidence(
                    instance=entity,
                    llm_confidence=1.0,
                    extracted_fields=entity_dict,
                    entity_certainty=certainty,
                    required_keys=required_keys,
                )
                if entity.confidence >= self.settings.extraction_confidence_threshold:
                    results.append(entity)

        results.sort(
            key=lambda e: (
                -e.confidence,
                _TYPE_PRIORITY.get(type(e).__name__, 999),
            )
        )
        return results