"""Tests for BaseExtractor abstract class and AIHarness wrapper."""

import pytest
from pydantic import ValidationError
from thread.extraction.models import GOAL, BaseEntity
from thread.extraction.config import Settings
from thread.extraction.base import BaseExtractor, AIHarness


class TestBaseExtractorAbstract:
    """BaseExtractor cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        """BaseExtractor is abstract and raises TypeError."""
        with pytest.raises(TypeError):
            BaseExtractor()  # type: ignore

    def test_cannot_instantiate_with_settings(self):
        """Even with settings, abstract class cannot be instantiated."""
        settings = Settings(openrouter_api_key="test")
        with pytest.raises(TypeError):
            BaseExtractor(settings=settings)  # type: ignore


class TestConfidenceScoring:
    """3-signal confidence scoring logic."""

    def test_average_of_three_signals(self):
        """LLM 0.8 + field 0.9 + entity 0.7 = 0.8."""
        # 9 of 10 required fields present -> field_completeness = 0.9
        fields = {f"f{i}": f"val{i}" for i in range(9)}
        required = {f"f{i}" for i in range(10)}
        result = BaseExtractor._score_confidence(
            None, 0.8, fields, 0.7, required_keys=required
        )
        assert abs(result - 0.8) < 0.01

    def test_all_zeros(self):
        """All signals zero = 0.0."""
        result = BaseExtractor._score_confidence(
            None, 0.0, {"a": None, "b": None}, 0.0, required_keys={"a", "b"}
        )
        assert result == 0.0

    def test_all_ones(self):
        """All signals 1.0 = 1.0."""
        result = BaseExtractor._score_confidence(
            None, 1.0, {"a": "x", "b": "y"}, 1.0, required_keys={"a", "b"}
        )
        assert result == 1.0

    def test_field_completeness_fraction(self):
        """3 of 5 required fields present = 0.6 completeness."""
        fields = {"id": "1", "name": "test", "title": "hello", "desc": None, "extra": None}
        required = {"id", "name", "title", "desc", "extra"}
        complete = BaseExtractor.confidence_field_completeness(fields, required)
        assert abs(complete - 0.6) < 0.01

    def test_confidence_clamped_above_one(self):
        """Clamp any result > 1.0 to 1.0."""
        result = BaseExtractor._score_confidence(
            None, 1.5, {"a": "x"}, 1.5, required_keys={"a"}
        )
        assert result == 1.0

    def test_confidence_clamped_below_zero(self):
        """Clamp any result < 0.0 to 0.0."""
        result = BaseExtractor._score_confidence(
            None, -0.5, {}, -0.5, required_keys={"a"}
        )
        assert result == 0.0


class TestTruncateInput:
    """Input truncation at word boundary."""

    def test_short_text_not_truncated(self):
        """Text under max length is returned as-is."""
        result = BaseExtractor._truncate_input("short text", max_length=100)
        assert result == "short text"

    def test_long_text_truncated_at_word_boundary(self):
        """Text over max length is truncated at last whole word with ellipsis."""
        text = "one two three four five six seven eight"
        result = BaseExtractor._truncate_input(text, max_length=15)
        assert result.endswith("...")
        assert result.startswith("one two three")

    def test_truncation_appends_ellipsis(self):
        """Truncated text ends with '...'."""
        text = "a long text that goes beyond the limit and should be cut"
        result = BaseExtractor._truncate_input(text, max_length=20)
        assert result.endswith("...")


class TestExtractionPrompt:
    """Prompt construction for entity extraction."""

    def test_prompt_contains_entity_type_name(self):
        """Built prompt includes the entity type name."""
        messages = BaseExtractor._build_extraction_prompt("GOAL", "some text about a goal")
        combined = " ".join(str(m) for m in messages)
        assert "GOAL" in combined

    def test_prompt_contains_input_text(self):
        """Built prompt includes the input text."""
        input_text = "we need to reduce latency"
        messages = BaseExtractor._build_extraction_prompt("DECISION", input_text)
        combined = " ".join(str(m) for m in messages)
        assert input_text in combined


class TestAIHarness:
    """AIHarness wrapper around Pydantic AI Agent."""

    def test_harness_creates_with_entity_type(self):
        """AIHarness can be instantiated with an entity type."""
        settings = Settings(openrouter_api_key="test")
        harness = AIHarness(GOAL, settings)
        assert harness.entity_type_name == "GOAL"

    def test_harness_entity_type_property(self):
        """Entity type name property returns correct string."""
        settings = Settings(openrouter_api_key="test")
        harness = AIHarness(GOAL, settings)
        assert harness.entity_type_name == "GOAL"


class TestFieldCompletenessStatic:
    """Static helper for field completeness calculation."""

    def test_all_fields_present(self):
        """All required fields present = 1.0."""
        fields = {"id": "1", "name": "test"}
        result = BaseExtractor.confidence_field_completeness(fields, {"id", "name"})
        assert result == 1.0

    def test_no_fields_present(self):
        """No required fields present = 0.0."""
        fields = {}
        result = BaseExtractor.confidence_field_completeness(fields, {"id", "name"})
        assert result == 0.0

    def test_empty_required_keys(self):
        """Empty required keys = 1.0 (nothing to check)."""
        fields = {"extra": "value"}
        result = BaseExtractor.confidence_field_completeness(fields, set())
        assert result == 1.0

    def test_mixed_presence(self):
        """2 of 4 fields present = 0.5."""
        fields = {"a": "yes", "b": "", "c": None, "d": "present"}
        result = BaseExtractor.confidence_field_completeness(fields, {"a", "b", "c", "d"})
        assert result == 0.5
