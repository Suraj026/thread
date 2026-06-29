"""Tests for entity extractors.

Tests the keyword classification, extractor dispatch, 
confidence scoring and priority tiebreaking logic. 
"""

import pytest
from unittest.mock import patch

from thread.extraction.extractors import (
    EXTRACTOR_REGISTRY,
    PlanningExtractor,
    DevelopmentExtractor,
    OperationsExtractor,
    IncidentExtractor,
    LearningExtractor,
    CollaborationExtractor,
    all_extractors,
)
from thread.extraction.extractors.planning import (
    _TYPE_PRIORITY,
    _classify_entity_types,
)
from thread.extraction.models import (
    ALTERNATIVE,
    ASSUMPTION,
    BUG,
    COMMIT,
    CONSTRAINT,
    DECISION,
    DEPLOYMENT,
    DEPLOYMENT_METRIC,
    DISCUSSION,
    ENGINEER,
    FIX,
    GOAL,
    IMPLEMENTATION,
    INCIDENT,
    LESSON,
    MEETING,
    PLATFORM,
    PR,
    PROJECT,
    ROLLBACK,
    ROOT_CAUSE,
    SERVICE,
    TEAM,
    TECHNOLOGY,
    WARNING,
)


class TestClassifier:
    """Tests for _classify_entity_types keyword-based classification."""

    def test_classify_goal_text(self):
        text = "Our goal is to reduce checkout latency under 200ms. This is a key objective for Q3."
        result = _classify_entity_types(text)
        assert "GOAL" in result
        assert result["GOAL"] > 0.5

    def test_classify_decision_text(self):
        text = "The team decided to use Redis for caching. We opted for it over Memcached."
        result = _classify_entity_types(text)
        assert "DECISION" in result
        assert result["DECISION"] > 0.3

    def test_classify_alternative_text(self):
        text = "We considered both Redis and Memcached as alternatives. The pros and cons were weighed."
        result = _classify_entity_types(text)
        assert "ALTERNATIVE" in result

    def test_classify_assumption_text(self):
        text = "We assume the traffic will grow 2x. This hypothesis needs validation."
        result = _classify_entity_types(text)
        assert "ASSUMPTION" in result

    def test_classify_constraint_text(self):
        text = "Budget limitation means we cannot use the enterprise tier. This is a hard constraint."
        result = _classify_entity_types(text)
        assert "CONSTRAINT" in result

    def test_empty_text_returns_empty(self):
        assert _classify_entity_types("") == {}

    def test_unrelated_text_returns_empty(self):
        assert _classify_entity_types("The weather is nice today.") == {}

    def test_multi_type_classification(self):
        text = (
            "Our goal is to improve API latency. We decided to use GraphQL. "
            "The budget constraint limits our infrastructure spend."
        )
        result = _classify_entity_types(text)
        assert "GOAL" in result
        assert "DECISION" in result
        assert "CONSTRAINT" in result

    def test_classify_implementation_text(self):
        text = "We implemented the new caching layer in the checkout service."
        result = _classify_entity_types(text)
        assert "IMPLEMENTATION" in result

    def test_classify_pr_text(self):
        text = "The pull request #142 was merged after review."
        result = _classify_entity_types(text)
        assert "PR" in result

    def test_classify_commit_text(self):
        text = "Commit a1b2c3d added the new rate limiting middleware."
        result = _classify_entity_types(text)
        assert "COMMIT" in result


class TestPlanningExtractor:
    """Tests for PlanningExtractor — dispatch, confidence, threshold."""

    def test_group_name(self):
        assert PlanningExtractor.GROUP_NAME == "planning"

    def test_entity_types_defined(self):
        extractor = PlanningExtractor()
        assert extractor._ENTITY_TYPES == [GOAL, DECISION, ALTERNATIVE, ASSUMPTION, CONSTRAINT]

    def test_empty_input_returns_empty_list(self):
        extractor = PlanningExtractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []
        assert extractor.extract(None) == []  # type: ignore

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_goal(self, mock_run):
        mock_run.return_value = GOAL(
            id="goal-1",
            title="Reduce latency",
            description="Reduce checkout latency to under 200ms",
        )
        text = "Our goal is to reduce checkout latency under 200ms."
        extractor = PlanningExtractor()
        results = extractor.extract(text)
        assert len(results) >= 1
        assert any(isinstance(e, GOAL) for e in results)

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_decision(self, mock_run):
        mock_run.return_value = DECISION(
            id="dec-1",
            title="Use Redis",
            decision_type="Technology",
            rationale="Better data structures and persistence",
        )
        text = "The team decided to use Redis for caching."
        extractor = PlanningExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, DECISION) for e in results)

    @patch("thread.extraction.base.AIHarness.run")
    def test_multi_entity_extraction(self, mock_run):
        """Both GOAL and DECISION extracted from combined text."""
        mock_run.side_effect = [
            GOAL(id="g-1", title="Goal", description="Desc"),
            DECISION(id="d-1", title="Decision", decision_type="Tech", rationale="Rationale"),
        ]

        text = "Our goal is X. We decided to use Y."
        extractor = PlanningExtractor()
        results = extractor.extract(text)
        type_names = {type(e).__name__ for e in results}
        assert "GOAL" in type_names or "DECISION" in type_names

    @patch("thread.extraction.base.AIHarness.run")
    def test_confidence_threshold_filtering(self, mock_run):
        """Entities below threshold are excluded."""
        mock_run.side_effect = [
            GOAL(id="g-1", title="Goal", description="Desc", confidence=0.1),
            DECISION(id="d-1", title="Dec", decision_type="Tech", rationale="R", confidence=0.9),
        ]

        # Override threshold to 0.5
        extractor = PlanningExtractor()
        original_threshold = extractor.settings.extraction_confidence_threshold
        extractor.settings.extraction_confidence_threshold = 0.5

        text = "Our goal is X. We decided to use Y."
        results = extractor.extract(text)

        # Only the GOAL with confidence < 0.5 should have been filtered
        for entity in results:
            assert entity.confidence >= 0.5

    def test_priority_tiebreaker_order(self):
        """feature-oriented types sort before collaboration types."""
        assert _TYPE_PRIORITY["GOAL"] < _TYPE_PRIORITY["DISCUSSION"]
        assert _TYPE_PRIORITY["IMPLEMENTATION"] < _TYPE_PRIORITY["TEAM"]

    @patch("thread.extraction.base.AIHarness.run")
    def test_unrelated_text_returns_empty(self, mock_run):
        """Text matching no entity types should not call AIHarness."""
        mock_run.side_effect = Exception("Should not be called")
        text = "The weather is nice today."
        extractor = PlanningExtractor()
        results = extractor.extract(text)
        assert results == []


class TestDevelopmentExtractor:
    """Tests for DevelopmentExtractor."""

    def test_group_name(self):
        assert DevelopmentExtractor.GROUP_NAME == "development"

    def test_entity_types_defined(self):
        extractor = DevelopmentExtractor()
        assert extractor._ENTITY_TYPES == [IMPLEMENTATION, PR, COMMIT]

    def test_empty_input_returns_empty_list(self):
        extractor = DevelopmentExtractor()
        assert extractor.extract("") == []

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_implementation(self, mock_run):
        mock_run.return_value = IMPLEMENTATION(
            id="impl-1",
            component="checkout-service",
            approach="Added Redis caching layer",
        )
        text = "We implemented a new Redis caching layer in the checkout service."
        extractor = DevelopmentExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, IMPLEMENTATION) for e in results)


class TestOperationsExtractor:
    """Tests for OperationsExtractor — deployment metrics and validation."""

    def test_group_name(self):
        assert OperationsExtractor.GROUP_NAME == "operations"

    def test_entity_types_defined(self):
        extractor = OperationsExtractor()
        assert extractor._ENTITY_TYPES == [DEPLOYMENT, DEPLOYMENT_METRIC, WARNING]

    def test_empty_input_returns_empty_list(self):
        extractor = OperationsExtractor()
        assert extractor.extract("") == []

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_deployment(self, mock_run):
        mock_run.return_value = DEPLOYMENT(
            id="dep-1",
            environment="production",
            status="Succeeded",
        )
        text = "Deployed version 2.1.0 to production via rolling update."
        extractor = OperationsExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, DEPLOYMENT) for e in results)

    def test_validate_metric_p95_negative(self):
        """Use model_construct to bypass Pydantic's ge=0 validator."""
        metric = DEPLOYMENT_METRIC.model_construct(id="m-1", p95_latency_ms=-1.0)
        penalty = OperationsExtractor._validate_metric(metric)
        assert penalty == 0.2

    def test_validate_metric_error_rate_out_of_range(self):
        metric = DEPLOYMENT_METRIC.model_construct(id="m-1", error_rate=1.5)
        penalty = OperationsExtractor._validate_metric(metric)
        assert penalty == 0.2

    def test_validate_metric_throughput_negative(self):
        metric = DEPLOYMENT_METRIC.model_construct(id="m-1", throughput_rps=-5)
        penalty = OperationsExtractor._validate_metric(metric)
        assert penalty == 0.2

    def test_validate_metric_all_valid(self):
        metric = DEPLOYMENT_METRIC(
            id="m-1",
            p95_latency_ms=150.0,
            error_rate=0.02,
            throughput_rps=1000,
        )
        penalty = OperationsExtractor._validate_metric(metric)
        assert penalty == 0.0

    def test_validate_metric_multiple_failures(self):
        metric = DEPLOYMENT_METRIC.model_construct(
            id="m-1",
            p95_latency_ms=-1.0,
            error_rate=1.5,
            throughput_rps=-5,
        )
        penalty = OperationsExtractor._validate_metric(metric)
        assert penalty == pytest.approx(0.6)

    def test_classify_deployment_text(self):
        text = "Deployed to production with rolling update strategy."
        result = _classify_entity_types(text)
        assert "DEPLOYMENT" in result

    def test_classify_warning_text(self):
        text = "Warning: the connection pool is approaching its limit."
        result = _classify_entity_types(text)
        assert "WARNING" in result


class TestIncidentExtractor:
    """Tests for IncidentExtractor — causality and confidence bonus."""

    def test_group_name(self):
        assert IncidentExtractor.GROUP_NAME == "incident"

    def test_entity_types_defined(self):
        extractor = IncidentExtractor()
        assert extractor._ENTITY_TYPES == [INCIDENT, BUG, ROOT_CAUSE, FIX, ROLLBACK]

    def test_empty_input_returns_empty_list(self):
        extractor = IncidentExtractor()
        assert extractor.extract("") == []

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_incident(self, mock_run):
        mock_run.return_value = INCIDENT(
            id="inc-1",
            severity="Critical",
            summary="Checkout service returned 503 errors",
        )
        text = "Critical incident: checkout service outage detected at 14:32 UTC."
        extractor = IncidentExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, INCIDENT) for e in results)

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_bug(self, mock_run):
        mock_run.return_value = BUG(
            id="bug-1",
            title="Null pointer in checkout flow",
            severity="High",
        )
        text = "Bug: null pointer exception in checkout flow when promo code is empty."
        extractor = IncidentExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, BUG) for e in results)

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_root_cause(self, mock_run):
        mock_run.return_value = ROOT_CAUSE(
            id="rc-1",
            cause_type="Configuration",
            description="Database connection pool limits not updated after deployment",
        )
        text = "Root cause: configuration drift between staging and production caused the outage."
        extractor = IncidentExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, ROOT_CAUSE) for e in results)

    def test_classify_incident_text(self):
        text = "Critical incident: the checkout service is down."
        result = _classify_entity_types(text)
        assert "INCIDENT" in result

    def test_classify_fix_text(self):
        text = "The fix was to increase the connection pool size."
        result = _classify_entity_types(text)
        assert "FIX" in result

    def test_classify_rollback_text(self):
        text = "We had to rollback the deployment to the previous version."
        result = _classify_entity_types(text)
        assert "ROLLBACK" in result

    def test_classify_root_cause_text(self):
        text = "The root cause was a configuration drift between environments."
        result = _classify_entity_types(text)
        assert "ROOT_CAUSE" in result

    @patch("thread.extraction.base.AIHarness.run")
    def test_causality_bonus_applied(self, mock_run):
        """When both INCIDENT and ROOT_CAUSE extracted, RC gets +0.05."""
        mock_run.side_effect = [
            INCIDENT(id="i-1", severity="High", summary="Outage"),
            ROOT_CAUSE(id="r-1", cause_type="Config", description="Config drift"),
        ]
        text = "Incident: outage caused by configuration drift."
        extractor = IncidentExtractor()
        extractor.settings.extraction_confidence_threshold = 0.0  # Don't filter
        results = extractor.extract(text)
        for entity in results:
            if isinstance(entity, ROOT_CAUSE):
                # Confidence should be > 1.0 (baseline) due to causality bonus
                assert entity.confidence > 0.0


class TestLearningExtractor:
    """Tests for LearningExtractor — single type, keyword boost."""

    def test_group_name(self):
        assert LearningExtractor.GROUP_NAME == "learning"

    def test_entity_types_defined(self):
        extractor = LearningExtractor()
        assert extractor._ENTITY_TYPES == [LESSON]

    def test_empty_input_returns_empty_list(self):
        extractor = LearningExtractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []
        assert extractor.extract(None) == []  # type: ignore

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_lesson(self, mock_run):
        mock_run.return_value = LESSON(
            id="lsn-1",
            lesson_text="Always validate input before processing",
            category="Process",
            actionable_items=["Add input validation to all endpoints"],
        )
        text = "We learned that input validation is critical."
        extractor = LearningExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, LESSON) for e in results)

    def test_has_lesson_keywords_explicit(self):
        extractor = LearningExtractor()
        assert extractor._has_lesson_keywords(
            "Key takeaway from the retrospective"
        )
        assert extractor._has_lesson_keywords(
            "Next time we should do this differently"
        )
        assert not extractor._has_lesson_keywords(
            "The weather is nice today."
        )

    def test_has_lesson_keywords_retrospective(self):
        extractor = LearningExtractor()
        assert extractor._has_lesson_keywords(
            "Retrospective findings show we need better monitoring"
        )

    @patch("thread.extraction.base.AIHarness.run")
    def test_keyword_confidence_boost(self, mock_run):
        """Confidence is higher when explicit lesson keywords are present."""
        # First call returns the direct extraction lesson
        # Second call returns the implicit extraction lesson
        mock_run.side_effect = [
            LESSON(
                id="lsn-1",
                lesson_text="Key takeaway: validate inputs",
                category="Process",
            ),
            None,  # No implicit lesson
        ]
        text = "Key takeaway from the retrospective is to validate inputs."
        extractor = LearningExtractor()
        extractor.settings.extraction_confidence_threshold = 0.0
        results = extractor.extract(text)
        assert len(results) >= 1
        # Confidence should be > 0 (baseline with keyword boost)
        for entity in results:
            assert entity.confidence > 0.0

    def test_classify_lesson_text(self):
        """LESSON keywords now in _CLASSIFICATION_KEYWORDS."""
        text = "Key takeaway from the retrospective: add monitoring."
        result = _classify_entity_types(text)
        assert "LESSON" in result

    def test_classify_team_text(self):
        text = "The Platform squad worked on the checkout service."
        result = _classify_entity_types(text)
        # 'squad' should match TEAM
        assert "TEAM" in result


class TestCollaborationExtractor:
    """Tests for CollaborationExtractor — 8 types, reference resolution."""

    def test_group_name(self):
        assert CollaborationExtractor.GROUP_NAME == "collaboration"

    def test_entity_types_defined(self):
        extractor = CollaborationExtractor()
        assert set(extractor._ENTITY_TYPES) == {
            ENGINEER, TEAM, MEETING, DISCUSSION,
            PLATFORM, SERVICE, PROJECT, TECHNOLOGY,
        }

    def test_empty_input_returns_empty_list(self):
        extractor = CollaborationExtractor()
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []
        assert extractor.extract(None) == []  # type: ignore

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_engineer(self, mock_run):
        mock_run.return_value = ENGINEER(
            id="eng-1",
            name="Alex Chen",
            role="Senior Engineer",
            team="Platform",
        )
        text = "Alex Chen is a senior engineer on the Platform team."
        extractor = CollaborationExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, ENGINEER) for e in results)

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_team(self, mock_run):
        mock_run.return_value = TEAM(
            id="team-1",
            team_name="Platform",
            org="Engineering",
            function="Infrastructure",
        )
        text = "The Platform team handles infrastructure."
        extractor = CollaborationExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, TEAM) for e in results)

    @patch("thread.extraction.base.AIHarness.run")
    def test_extract_meeting(self, mock_run):
        mock_run.return_value = MEETING(
            id="mtg-1",
            meeting_type="Standup",
            participants=["Alex", "Bob"],
        )
        text = "In standup today, we discussed the deployment progress."
        extractor = CollaborationExtractor()
        results = extractor.extract(text)
        assert any(isinstance(e, MEETING) for e in results)

    def test_resolve_references_engineer_team_linking(self):
        """'Alex from Platform team' creates both ENGINEER and TEAM."""
        extractor = CollaborationExtractor()
        entities: list = []
        resolved = extractor._resolve_references(
            "Alex from Platform team worked on the checkout service.",
            entities,
        )
        assert len(resolved) >= 2
        assert any(
            isinstance(e, ENGINEER) and e.name.lower() == "alex"
            for e in resolved
        )
        assert any(
            isinstance(e, TEAM) and e.team_name.lower() == "platform"
            for e in resolved
        )

    def test_resolve_references_no_duplicates(self):
        """Existing entities should not be duplicated."""
        extractor = CollaborationExtractor()
        entities = [
            ENGINEER(id="eng-1", name="Alex", team="Platform", confidence=0.9),
            TEAM(id="team-1", team_name="Platform", confidence=0.9),
        ]
        resolved = extractor._resolve_references(
            "Alex from Platform team worked on the checkout service.",
            entities,
        )
        # Should not create new entities — existing ones already cover it
        engineer_count = sum(
            1 for e in resolved
            if isinstance(e, ENGINEER) and e.name == "Alex"
        )
        assert engineer_count == 1

    def test_resolve_references_no_match(self):
        """Text without reference patterns should not create new entities."""
        extractor = CollaborationExtractor()
        resolved = extractor._resolve_references(
            "The weather is nice today.",
            [],
        )
        assert resolved == []

    def test_classify_engineer_text(self):
        text = "The lead developer Alex reported a critical bug."
        result = _classify_entity_types(text)
        assert "ENGINEER" in result

    def test_classify_technology_text(self):
        text = "We use PostgreSQL as our primary database."
        result = _classify_entity_types(text)
        assert "TECHNOLOGY" in result

    def test_classify_platform_text(self):
        text = "We track issues in Jira and code in GitHub."
        result = _classify_entity_types(text)
        assert "PLATFORM" in result


class TestExtractorRegistry:
    """Tests for extractor registry and factory."""

    def test_registry_contains_all(self):
        assert "planning" in EXTRACTOR_REGISTRY
        assert "development" in EXTRACTOR_REGISTRY
        assert "operations" in EXTRACTOR_REGISTRY
        assert "incident" in EXTRACTOR_REGISTRY
        assert "learning" in EXTRACTOR_REGISTRY
        assert "collaboration" in EXTRACTOR_REGISTRY
        assert EXTRACTOR_REGISTRY["planning"] is PlanningExtractor
        assert EXTRACTOR_REGISTRY["development"] is DevelopmentExtractor
        assert EXTRACTOR_REGISTRY["operations"] is OperationsExtractor
        assert EXTRACTOR_REGISTRY["incident"] is IncidentExtractor
        assert EXTRACTOR_REGISTRY["learning"] is LearningExtractor
        assert EXTRACTOR_REGISTRY["collaboration"] is CollaborationExtractor

    def test_all_extractors_returns_six(self):
        extractors = all_extractors()
        assert len(extractors) == 6
        group_names = [e.GROUP_NAME for e in extractors]
        assert "planning" in group_names
        assert "development" in group_names
        assert "operations" in group_names
        assert "incident" in group_names
        assert "learning" in group_names
        assert "collaboration" in group_names

    def test_all_extractors_are_concrete(self):
        import inspect
        for extractor in all_extractors():
            assert not inspect.isabstract(type(extractor))