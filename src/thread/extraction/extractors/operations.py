"""OperationsExtractor — extracts operations-phase entities.

Handles: DEPLOYMENT, DEPLOYMENT_METRIC, WARNING.
"""

from thread.extraction.base import BaseExtractor
from thread.extraction.models import (
    DEPLOYMENT,
    DEPLOYMENT_METRIC,
    WARNING,
    BaseEntity,
)
from thread.extraction.extractors.planning import (
    _TYPE_PRIORITY,
    _classify_entity_types,
)


class OperationsExtractor(BaseExtractor):
    """Extracts operations-phase entities from natural language text.

    Detects and extracts DEPLOYMENT, DEPLOYMENT_METRIC, and WARNING
    entities. After extraction, applies field-level numeric validation
    to DEPLOYMENT_METRIC — failed validations reduce entity confidence
    by 0.2 per failure.

    Attributes:
        GROUP_NAME:  Group identifier for registry lookup.
    """

    GROUP_NAME = "operations"
    _ENTITY_TYPES = [DEPLOYMENT, DEPLOYMENT_METRIC, WARNING]

    def extract(self, text: str) -> list[BaseEntity]:
        """Extract operations entities from natural language text.

        Pipeline:
        1. Keyword classification
        2. AIHarness dispatch per detected type
        3. Field-level validation for DEPLOYMENT_METRIC
        4. 3-signal confidence scoring (D-09) with threshold filtering
        5. Priority sorting (D-10)

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
                # Apply field-level validators for DEPLOYMENT_METRIC
                confidence_penalty = 0.0
                if isinstance(entity, DEPLOYMENT_METRIC):
                    confidence_penalty = self._validate_metric(entity)

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
                entity.confidence = (
                    self._score_confidence(
                        instance=entity,
                        llm_confidence=1.0,
                        extracted_fields=entity_dict,
                        entity_certainty=certainty,
                        required_keys=required_keys,
                    )
                    - confidence_penalty
                )
                entity.confidence = max(0.0, entity.confidence)

                if entity.confidence >= self.settings.extraction_confidence_threshold:
                    results.append(entity)

        results.sort(
            key=lambda e: (
                -e.confidence,
                _TYPE_PRIORITY.get(type(e).__name__, 999),
            )
        )
        return results

    @staticmethod
    def _validate_metric(metric: DEPLOYMENT_METRIC) -> float:
        """Validate DEPLOYMENT_METRIC numeric fields.

        Checks:
        - p95_latency_ms >= 0 (if set)
        - error_rate in [0.0, 1.0] (if set)
        - throughput_rps >= 0 (if set)

        Each failure reduces confidence by 0.2.

        Args:
            metric: The DEPLOYMENT_METRIC instance to validate.

        Returns:
            Total confidence penalty (0.0, 0.2, 0.4, or 0.6).
        """
        penalty = 0.0

        if metric.p95_latency_ms is not None and metric.p95_latency_ms < 0:
            penalty += 0.2

        if metric.error_rate is not None and (
            metric.error_rate < 0.0 or metric.error_rate > 1.0
        ):
            penalty += 0.2

        if metric.throughput_rps is not None and metric.throughput_rps < 0:
            penalty += 0.2

        return penalty
