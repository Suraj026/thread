"""Batch extraction orchestrator — runs all extractors, aggregates, applies tiebreaker.

This module ties together all 6 extractors into a single pipeline:
1. Runs each extractor against input text
2. Deduplicates by entity type (keeps highest confidence)
3. Filters below confidence threshold
4. Sorts by D-10 priority tier
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from thread.extraction.models import BaseEntity
from thread.extraction.extractors import all_extractors, EXTRACTOR_REGISTRY
from thread.extraction.config import Settings


# D-10 Priority tiebreaker: feature > decision > session/people > collaboration
# Lower tier number = higher priority
TYPE_PRIORITY: dict[str, int] = {
    # Tier 1: Feature-oriented
    "GOAL": 1,
    "IMPLEMENTATION": 1,
    "ALTERNATIVE": 1,
    "CONSTRAINT": 1,
    "ASSUMPTION": 1,
    # Tier 2: Decision-oriented
    "DECISION": 2,
    "PR": 2,
    "COMMIT": 2,
    "DEPLOYMENT": 2,
    "DEPLOYMENT_METRIC": 2,
    # Tier 3: Session/people-oriented
    "INCIDENT": 3,
    "BUG": 3,
    "ROOT_CAUSE": 3,
    "FIX": 3,
    "ROLLBACK": 3,
    "LESSON": 3,
    "WARNING": 3,
    # Tier 4: Collaboration
    "ENGINEER": 4,
    "TEAM": 4,
    "MEETING": 4,
    "DISCUSSION": 4,
    "PLATFORM": 4,
    "SERVICE": 4,
    "PROJECT": 4,
    "TECHNOLOGY": 4,
}


@dataclass
class BatchResult:
    """Result of a batch extraction operation.

    Attributes:
        entities: Deduplicated, filtered, sorted entity list.
        extracted_at: Timestamp of extraction.
        input_length: Character length of input text.
    """

    entities: list[BaseEntity] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.now)
    input_length: int = 0

    @property
    def entity_count(self) -> int:
        """Total number of entities in the result."""
        return len(self.entities)


class BatchExtractor:
    """Orchestrates multiple extractors against a single input text.

    Runs all 6 extractors in sequence, deduplicates by entity type,
    applies confidence threshold filtering, and sorts by priority.

    Usage:
        extractor = BatchExtractor()
        result = extractor.run("Team decided to use Redis for caching")
        print(result.entity_count)  # 2 (DECISION + TECHNOLOGY)
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize with settings and create all extractor instances.

        Args:
            settings: Application settings. Creates default if None.
        """
        self.settings = settings or Settings()
        self.extractors = all_extractors()

    def run(self, text: str) -> BatchResult:
        """Run full extraction pipeline on input text.

        Steps:
        1. Run all 6 extractors in sequence
        2. Deduplicate by entity type (keep highest confidence)
        3. Filter entities below confidence threshold
        4. Sort by D-10 priority then descending confidence

        Args:
            text: Natural language input text.

        Returns:
            BatchResult with processed entity list and metadata.
        """
        if not text.strip():
            return BatchResult(input_length=len(text))

        # Step 1: Run all extractors, collect entities
        all_entities: list[BaseEntity] = []
        for extractor in self.extractors:
            try:
                extracted = extractor.extract(text)
                all_entities.extend(extracted)
            except Exception:
                # Isolate extractor failures — one failing extractor
                # should not crash the entire batch
                continue

        # Step 2: Deduplicate by entity type class name
        type_best: dict[str, BaseEntity] = {}
        for entity in all_entities:
            type_name = type(entity).__name__
            current = type_best.get(type_name)
            if current is None or entity.confidence > current.confidence:
                type_best[type_name] = entity

        # Step 3: Threshold filter
        threshold = self.settings.extraction_confidence_threshold
        filtered = [e for e in type_best.values() if e.confidence >= threshold]

        # Step 4: Priority sort (lower tier first, higher confidence first within tier)
        filtered.sort(
            key=lambda e: (
                TYPE_PRIORITY.get(type(e).__name__, 99),
                -e.confidence,
            )
        )

        return BatchResult(
            entities=filtered,
            extracted_at=datetime.now(),
            input_length=len(text),
        )

    def run_on_extractor_group(self, text: str, group_name: str) -> list[BaseEntity]:
        """Run only a single extractor group against input text.

        Args:
            text: Natural language input text.
            group_name: Extractor group name (e.g., 'planning', 'development').

        Returns:
            List of entities from the specified extractor group.

        Raises:
            ValueError: If group_name is not in the registry.
        """
        if group_name not in EXTRACTOR_REGISTRY:
            available = ", ".join(EXTRACTOR_REGISTRY.keys())
            raise ValueError(
                f"Unknown extractor group: '{group_name}'. "
                f"Available groups: {available}"
            )

        extractor_cls = EXTRACTOR_REGISTRY[group_name]
        extractor = extractor_cls(self.settings)
        return extractor.extract(text)

    def entity_summary(self, result: BatchResult) -> dict:
        """Generate a summary dict from a batch result.

        Args:
            result: BatchResult to summarize.

        Returns:
            Dict with total_entities, by_type counts, and avg_confidence.
        """
        if not result.entities:
            return {"total_entities": 0, "by_type": {}, "avg_confidence": 0.0}

        by_type: dict[str, int] = {}
        for entity in result.entities:
            type_name = type(entity).__name__
            by_type[type_name] = by_type.get(type_name, 0) + 1

        avg_conf = sum(e.confidence for e in result.entities) / len(result.entities)

        return {
            "total_entities": result.entity_count,
            "by_type": by_type,
            "avg_confidence": round(avg_conf, 4),
        }