"""Tests for batch extraction orchestration: dedup, tiebreaker, threshold."""

from datetime import datetime
from unittest.mock import patch

import pytest

from thread.extraction.batch_extractor import BatchExtractor, BatchResult, TYPE_PRIORITY
from thread.extraction.models import (
    GOAL, DECISION, DISCUSSION, MEETING, LESSON, BUG,
    ALTERNATIVE, ASSUMPTION, CONSTRAINT,
    BaseEntity,
)


def _make_entity(
    entity_class: type[BaseEntity],
    confidence: float = 0.8,
    **overrides,
) -> BaseEntity:
    """Create an entity with minimal required fields and given confidence.

    Args:
        entity_class: The Pydantic model class to instantiate.
        confidence: Extraction confidence score.
        **overrides: Any additional field overrides.

    Returns:
        An entity instance with confidence set.
    """
    # Minimal defaults per entity type (required fields only)
    defaults: dict = {}
    if entity_class == GOAL:
        defaults = {"id": "g1", "title": "Test Goal", "description": "A test goal"}
    elif entity_class == DECISION:
        defaults = {
            "id": "d1", "title": "Test Decision",
            "decision_type": "Architecture", "rationale": "Because",
        }
    elif entity_class == DISCUSSION:
        defaults = {"id": "ds1", "topic": "Test Discussion"}
    elif entity_class == MEETING:
        defaults = {"id": "m1", "meeting_type": "Standup"}
    elif entity_class == LESSON:
        defaults = {"id": "l1", "lesson_text": "Test", "category": "Process"}
    elif entity_class == BUG:
        defaults = {"id": "b1", "title": "Test Bug"}
    elif entity_class == ALTERNATIVE:
        defaults = {"id": "a1", "description": "Test alternative"}
    elif entity_class == ASSUMPTION:
        defaults = {"id": "as1", "assumption_text": "Test assumption"}
    elif entity_class == CONSTRAINT:
        defaults = {"id": "c1", "constraint_type": "Budget", "description": "Test constraint"}

    defaults.update(overrides)
    entity = entity_class(**defaults)
    entity.confidence = confidence
    return entity


class MockExtractor:
    """Mock extractor that returns predefined entities."""

    def __init__(self, entities: list[BaseEntity]):
        self._entities = entities

    def extract(self, text: str) -> list[BaseEntity]:
        return self._entities


# ---- BatchExtractor tests ----


def test_run_returns_entities():
    """Test 1: BatchExtractor.run() on a simple input returns entities."""
    mock_planning = MockExtractor([
        _make_entity(GOAL, confidence=0.9, id="g1"),
    ])
    with patch("thread.extraction.batch_extractor.all_extractors") as mock_factory:
        mock_factory.return_value = [mock_planning]
        extractor = BatchExtractor()
        result = extractor.run("reduce checkout latency")

    assert len(result.entities) == 1
    assert isinstance(result.entities[0], GOAL)
    assert result.entity_count == 1


def test_threshold_filtering():
    """Test 2: Entities below confidence threshold are filtered out."""
    mock_ext = MockExtractor([
        _make_entity(GOAL, confidence=0.9, id="g1"),
        _make_entity(DECISION, confidence=0.2, id="d1"),  # below default 0.3
    ])
    with patch("thread.extraction.batch_extractor.all_extractors") as mock_factory:
        mock_factory.return_value = [mock_ext]
        extractor = BatchExtractor()
        result = extractor.run("test text")

    assert len(result.entities) == 1
    assert type(result.entities[0]).__name__ == "GOAL"


def test_priority_tiebreaker():
    """Test 3: Entities with same confidence sorted by D-10 priority."""
    mock_ext = MockExtractor([
        _make_entity(GOAL, confidence=0.7, id="g1"),
        _make_entity(DECISION, confidence=0.7, id="d1"),   # tier 2
        _make_entity(DISCUSSION, confidence=0.7, id="ds1"),  # tier 4
    ])
    with patch("thread.extraction.batch_extractor.all_extractors") as mock_factory:
        mock_factory.return_value = [mock_ext]
        extractor = BatchExtractor()
        result = extractor.run("test text")

    # Should be sorted: GOAL (tier 1) → DECISION (tier 2) → DISCUSSION (tier 4)
    names = [type(e).__name__ for e in result.entities]
    assert names == ["GOAL", "DECISION", "DISCUSSION"]


def test_empty_input():
    """Test 4: Empty text returns empty BatchResult."""
    extractor = BatchExtractor()
    result = extractor.run("")
    assert result.entity_count == 0
    assert result.input_length == 0

    result2 = extractor.run("   ")
    assert result2.entity_count == 0


def test_deduplicate_by_type():
    """Test 5: When two extractors return same type, keep higher confidence."""
    mock_planning = MockExtractor([
        _make_entity(GOAL, confidence=0.9, id="g1"),
    ])
    # A second extractor that also returns a GOAL but with lower confidence
    mock_dev = MockExtractor([
        _make_entity(GOAL, confidence=0.5, id="g2"),
    ])
    with patch("thread.extraction.batch_extractor.all_extractors") as mock_factory:
        mock_factory.return_value = [mock_planning, mock_dev]
        extractor = BatchExtractor()
        result = extractor.run("test text")

    # Should keep only the higher-confidence GOAL
    assert len(result.entities) == 1
    assert result.entities[0].confidence == 0.9


def test_entity_summary():
    """Test 6: entity_summary returns correct counts and averages."""
    mock_ext = MockExtractor([
        _make_entity(GOAL, confidence=0.9, id="g1"),
        _make_entity(DECISION, confidence=0.7, id="d1"),
    ])
    with patch("thread.extraction.batch_extractor.all_extractors") as mock_factory:
        mock_factory.return_value = [mock_ext]
        extractor = BatchExtractor()
        result = extractor.run("test text")

    summary = extractor.entity_summary(result)
    assert summary["total_entities"] == 2
    assert summary["by_type"]["GOAL"] == 1
    assert summary["by_type"]["DECISION"] == 1
    assert summary["avg_confidence"] == 0.8  # (0.9 + 0.7) / 2


def test_extraction_timestamp():
    """Test 7: BatchResult includes extraction timestamp."""
    result = BatchResult(input_length=10)
    assert isinstance(result.extracted_at, datetime)
    assert result.input_length == 10