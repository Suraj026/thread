"""Tests for all 21 Pydantic entity models."""

import pytest
from pydantic import ValidationError
from datetime import datetime

from thread.extraction.models import (
    BaseEntity,
    BaseGraphEntity,
    GOAL,
    DECISION,
    ALTERNATIVE,
    ASSUMPTION,
    CONSTRAINT,
    IMPLEMENTATION,
    PR,
    COMMIT,
    DEPLOYMENT,
    DEPLOYMENT_METRIC,
    WARNING,
    INCIDENT,
    BUG,
    ROOT_CAUSE,
    FIX,
    ROLLBACK,
    LESSON,
    ENGINEER,
    PROJECT,
    TECHNOLOGY,
    SERVICE,
    MEETING,
    DISCUSSION,
    TEAM,
    PLATFORM,
    ENTITY_TYPE_MAP,
)

# --- Test 1: Every model instantiates with minimal fields ---

ALL_MODEL_TYPES = [
    GOAL, DECISION, ALTERNATIVE, ASSUMPTION, CONSTRAINT,
    IMPLEMENTATION, PR, COMMIT,
    DEPLOYMENT, DEPLOYMENT_METRIC, WARNING,
    INCIDENT, BUG, ROOT_CAUSE, FIX, ROLLBACK,
    LESSON,
    ENGINEER, PROJECT, TECHNOLOGY, SERVICE,
    MEETING, DISCUSSION, TEAM, PLATFORM,
]

REQUIRED_ID_FIELDS = {
    GOAL: ["id", "title", "description"],
    DECISION: ["id", "title", "decision_type", "rationale"],
    ALTERNATIVE: ["id", "description"],
    ASSUMPTION: ["id", "assumption_text"],
    CONSTRAINT: ["id", "constraint_type", "description"],
    IMPLEMENTATION: ["id", "component", "approach"],
    PR: ["id", "title"],
    COMMIT: ["id", "message"],
    DEPLOYMENT: ["id", "environment"],
    DEPLOYMENT_METRIC: ["id"],
    WARNING: ["id", "warning_type", "message"],
    INCIDENT: ["id", "summary"],
    BUG: ["id", "title"],
    ROOT_CAUSE: ["id", "cause_type", "description"],
    FIX: ["id", "description"],
    ROLLBACK: ["id", "reason"],
    LESSON: ["id", "lesson_text", "category"],
    ENGINEER: ["id", "name"],
    PROJECT: ["id", "project_key", "project_name"],
    TECHNOLOGY: ["id", "tech_name"],
    SERVICE: ["id", "service_name"],
    MEETING: ["id", "meeting_type"],
    DISCUSSION: ["id", "topic"],
    TEAM: ["id", "team_name"],
    PLATFORM: ["id", "name"],
}


class TestInstantiation:
    """Every entity model can be instantiated with minimal required fields."""

    @pytest.mark.parametrize("model_class", ALL_MODEL_TYPES)
    def test_minimal_instantiation(self, model_class):
        """Each model can be created with only required fields."""
        required = REQUIRED_ID_FIELDS[model_class]
        kwargs = {field: _sample_value(model_class, field) for field in required}
        instance = model_class(**kwargs)
        assert instance.confidence == 0.0  # defaults to 0.0
        for field in required:
            assert getattr(instance, field) is not None

    @pytest.mark.parametrize("model_class", ALL_MODEL_TYPES)
    def test_confidence_defaults_to_zero(self, model_class):
        """Confidence defaults to 0.0 and is always present."""
        required = REQUIRED_ID_FIELDS[model_class]
        kwargs = {field: _sample_value(model_class, field) for field in required}
        instance = model_class(**kwargs)
        assert instance.confidence == 0.0


class TestConfidenceValidator:
    """Confidence must be in [0.0, 1.0] range."""

    @pytest.mark.parametrize("model_class", ALL_MODEL_TYPES)
    def test_confidence_accepts_valid_range(self, model_class):
        """Confidence values in [0.0, 1.0] are accepted."""
        for val in [0.0, 0.3, 0.5, 0.7, 1.0]:
            instance = _make_minimal(model_class, confidence=val)
            assert instance.confidence == val

    @pytest.mark.parametrize("model_class", ALL_MODEL_TYPES)
    def test_confidence_rejects_below_zero(self, model_class):
        """Confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal(model_class, confidence=-0.1)

    @pytest.mark.parametrize("model_class", ALL_MODEL_TYPES)
    def test_confidence_rejects_above_one(self, model_class):
        """Confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal(model_class, confidence=1.1)


class TestModelValidate:
    """All models support model_validate() round-trip."""

    @pytest.mark.parametrize("model_class", ALL_MODEL_TYPES)
    def test_round_trip(self, model_class):
        """model_validate(model_dump()) returns identical model."""
        required = REQUIRED_ID_FIELDS[model_class]
        kwargs = {field: _sample_value(model_class, field) for field in required}
        kwargs["confidence"] = 0.85
        original = model_class(**kwargs)
        data = original.model_dump()
        restored = model_class.model_validate(data)
        assert restored.confidence == original.confidence
        for field in required:
            assert getattr(restored, field) == getattr(original, field)


class TestGraphEntityFields:
    """BaseGraphEntity subclasses have Cognee-compatible fields."""

    GRAPH_MODELS = [GOAL, DECISION, IMPLEMENTATION, INCIDENT, LESSON, TECHNOLOGY]

    @pytest.mark.parametrize("model_class", GRAPH_MODELS)
    def test_embedding_fields_default_none(self, model_class):
        """embedding and description_embedding default to None."""
        instance = _make_minimal(model_class)
        assert instance.embedding is None
        assert instance.description_embedding is None
        assert instance.similarity_cluster_id is None
        assert instance.tags == []

    @pytest.mark.parametrize("model_class", GRAPH_MODELS)
    def test_embedding_fields_accept_values(self, model_class):
        """Embedding fields accept valid values."""
        instance = _make_minimal(model_class,
                                 embedding=[0.1, 0.2, 0.3],
                                 tags=["performance", "checkout"])
        assert instance.embedding == [0.1, 0.2, 0.3]
        assert "performance" in instance.tags


class TestEntityTypeMap:
    """ENTITY_TYPE_MAP contains all 21+ entity types."""

    def test_minimum_count(self):
        """At least 21 entity types registered."""
        assert len(ENTITY_TYPE_MAP) >= 21

    def test_contains_goal(self):
        """GOAL is in the map."""
        assert ENTITY_TYPE_MAP["GOAL"] == GOAL

    def test_contains_decision(self):
        """DECISION is in the map."""
        assert ENTITY_TYPE_MAP["DECISION"] == DECISION


class TestExtraFieldsForbidden:
    """Extra fields should be rejected."""

    @pytest.mark.parametrize("model_class", ALL_MODEL_TYPES)
    def test_extra_field_rejected(self, model_class):
        """Passing an unknown field raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_minimal(model_class, nonexistent_field="boom")


# --- Helpers ---

def _make_minimal(model_class, **overrides):
    """Create an entity with minimal required fields + optional overrides."""
    base = REQUIRED_ID_FIELDS[model_class]
    kwargs = {field: _sample_value(model_class, field) for field in base}
    kwargs.update(overrides)
    return model_class(**kwargs)


def _sample_value(model_class, field_name):
    """Generate a valid sample value for a field."""
    field_info = model_class.model_fields[field_name]
    field_type = str(field_info.annotation)

    if field_name == "id":
        return "test-id-001"
    if field_name in ("title", "name", "team_name", "project_name",
                      "service_name", "tech_name", "entity_type_name"):
        return "test-value"
    if "str" in field_type and "list" not in field_type:
        return "test-string"
    if "int" in field_type:
        return 0
    if "float" in field_type:
        return 0.0
    if "datetime" in field_type:
        return datetime(2026, 6, 29)
    if "list" in field_type:
        return []
    return "test-value"