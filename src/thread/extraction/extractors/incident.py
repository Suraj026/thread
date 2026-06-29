"""IncidentExtractor — extracts incident-resolution entities.

Handles: INCIDENT, BUG, ROOT_CAUSE, FIX, ROLLBACK.
"""

from thread.extraction.base import BaseExtractor
from thread.extraction.models import (
    BUG,
    FIX,
    INCIDENT,
    ROLLBACK,
    ROOT_CAUSE,
    BaseEntity,
)
from thread.extraction.extractors.planning import (
    _TYPE_PRIORITY,
    _classify_entity_types,
)


class IncidentExtractor(BaseExtractor):
    """Extracts incident-resolution entities from natural language text.

    Detects and extracts INCIDENT, BUG, ROOT_CAUSE, FIX, and ROLLBACK
    entities. After extraction, applies a causality confidence bonus:
    if INCIDENT and ROOT_CAUSE are both extracted from the same text,
    ROOT_CAUSE's confidence gets a +0.05 boost.

    Attributes:
        GROUP_NAME:  Group identifier for registry lookup.
    """

    GROUP_NAME = "incident"
    _ENTITY_TYPES = [INCIDENT, BUG, ROOT_CAUSE, FIX, ROLLBACK]

    def extract(self, text: str) -> list[BaseEntity]:
        """Extract incident-resolution entities from natural language text.

        Pipeline:
        1. Keyword classification
        2. AIHarness dispatch per detected type
        3. 3-signal confidence scoring (D-09)
        4. Causality bonus: INCIDENT + ROOT_CAUSE co-occurrence (+0.05)
        5. Threshold filtering and priority sorting (D-10)

        Args:
            text: Natural language text.

        Returns:
            List of extracted entities with confidence >= threshold.
        """
        if not text or not text.strip():
            return []

        classification = _classify_entity_types(text)
        results: list[BaseEntity] = []
        has_incident = False
        has_root_cause = False

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

                if isinstance(entity, INCIDENT):
                    has_incident = True
                if isinstance(entity, ROOT_CAUSE):
                    has_root_cause = True

                if entity.confidence >= self.settings.extraction_confidence_threshold:
                    results.append(entity)

        # Causality bonus: +0.05 to ROOT_CAUSE if both INCIDENT and ROOT_CAUSE found
        if has_incident and has_root_cause:
            for entity in results:
                if isinstance(entity, ROOT_CAUSE):
                    entity.confidence = min(1.0, entity.confidence + 0.05)

        results.sort(
            key=lambda e: (
                -e.confidence,
                _TYPE_PRIORITY.get(type(e).__name__, 999),
            )
        )
        return results
