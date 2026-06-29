"""PlanningExtractor — extracts planning-phase entities.

Handles: GOAL, DECISION, ALTERNATIVE, ASSUMPTION, CONSTRAINT.
"""

from thread.extraction.base import BaseExtractor
from thread.extraction.models import (
    ALTERNATIVE,
    ASSUMPTION,
    CONSTRAINT,
    DECISION,
    GOAL,
    BaseEntity,
)

# Keyword patterns for entity type classification
# More specific terms first to reduce false positives
_CLASSIFICATION_KEYWORDS: dict[str, list[str]] = {
    "GOAL": [
        "goal", "objective", "target", "outcome", "success criteria",
        "need to", "should achieve", "metric", "kpi", "milestone",
        "aspire", "strive", "purpose",
    ],
    "DECISION": [
        "decided", "chose", "selected", "elected", "decision",
        "we will use", "we opted", "settled on", "went with", "picked",
        "resolution", "conclusion", "chosen",
    ],
    "ALTERNATIVE": [
        "alternative", "option", "considered", "evaluated", "compared",
        "instead of", "versus", "vs", "trade-off", "pros and cons",
        "weighing", "candidate",
    ],
    "ASSUMPTION": [
        "assume", "assumption", "assuming", "presume", "presumption",
        "hypothesis", "believe", "belief", "expectation", "expected",
        "speculate", "guess",
    ],
    "CONSTRAINT": [
        "constraint", "limitation", "restriction", "cannot",
        "limited", "boundary", "requirement", "mandate", "budget",
        "deadline", "compliance", "regulation", "policy",
    ],
    "IMPLEMENTATION": [
        "implemented", "implementation", "built", "developed",
        "created", "added", "released", "launched", "rolled out",
        "wrote", "configured", "set up",
    ],
    "PR": [
        "pull request", "pr #", "code review", "merged", "branch",
        "opened a pr", "submitted", "review",
    ],
    "COMMIT": [
        "commit", "committed", "commit message", "hash", "sha",
        "change set", "patch",
    ],
    "DEPLOYMENT": [
        "deployment", "deploy", "released to", "environment",
        "canary", "blue/green", "staging", "production",
        "rolled out to", "promoted",
    ],
    "DEPLOYMENT_METRIC": [
        "p95", "latency", "error rate", "throughput", "rps",
        "uptime", "sla", "slo", "metric", "performance",
    ],
    "WARNING": [
        "warning", "deprecated", "notice", "caution",
        "anomaly", "non-fatal", "advisory",
    ],
    "INCIDENT": [
        "incident", "outage", "degradation", "downtime",
        "sev", "pager", "alert", "critical", "unavailable",
    ],
    "BUG": [
        "bug", "defect", "issue #", "regression", "error",
        "broken", "bug fix",
    ],
    "ROOT_CAUSE": [
        "root cause", "caused by", "triggered by",
        "underlying", "why this happened",
    ],
    "FIX": [
        "fix", "fixed", "patched", "resolution", "resolved",
        "hotfix", "corrected",
    ],
    "ROLLBACK": [
        "rollback", "revert", "rolled back", "reverted",
        "undo", "back out",
    ],
    "LESSON": [
        "lesson learned", "key takeaway", "retrospective",
        "next time", "never again", "we learned", "postmortem",
        "improvement", "action item",
    ],
    "ENGINEER": [
        "engineer", "developer", "contributor", "author",
        "assigned to", "reported by", "lead",
    ],
    "TEAM": [
        "team", "squad", "pod", "team lead",
        "team member", "cross-team",
    ],
    "MEETING": [
        "meeting", "standup", "sync", "retro",
        "design review", "sprint planning", "huddle",
    ],
    "DISCUSSION": [
        "discussion", "thread", "conversation",
        "slack thread", "async", "chat",
    ],
    "PLATFORM": [
        "platform", "tool", "jira", "slack",
        "github", "confluence", "notion",
    ],
    "SERVICE": [
        "service", "api", "microservice", "endpoint",
        "backend", "frontend",
    ],
    "PROJECT": [
        "project", "initiative", "epic",
        "workstream", "track",
    ],
    "TECHNOLOGY": [
        "technology", "framework", "library", "stack",
        "language", "database", "cache", "queue",
    ],
}

# Priority order for tiebreaking (D-10) — covers all 25 entity types
_TYPE_PRIORITY: dict[str, int] = {
    # Feature-oriented (Tier 1)
    "GOAL": 1,
    "IMPLEMENTATION": 2,
    "ALTERNATIVE": 3,
    "CONSTRAINT": 4,
    "ASSUMPTION": 5,
    # Decision-oriented (Tier 2)
    "DECISION": 6,
    "PR": 7,
    "COMMIT": 8,
    "DEPLOYMENT": 9,
    "DEPLOYMENT_METRIC": 10,
    # Session/people-oriented (Tier 3)
    "INCIDENT": 11,
    "BUG": 12,
    "ROOT_CAUSE": 13,
    "FIX": 14,
    "ROLLBACK": 15,
    "LESSON": 16,
    "WARNING": 17,
    # Collaboration (Tier 4)
    "ENGINEER": 18,
    "TEAM": 19,
    "MEETING": 20,
    "DISCUSSION": 21,
    "PLATFORM": 22,
    "SERVICE": 23,
    "PROJECT": 24,
    "TECHNOLOGY": 25,
}


class PlanningExtractor(BaseExtractor):
    """Extracts planning-phase entities from natural language text.

    Detects and extracts GOAL, DECISION, ALTERNATIVE, ASSUMPTION,
    and CONSTRAINT entities using keyword-based classification
    followed by LLM-powered structured extraction.

    Attributes:
        GROUP_NAME:  Group identifier for registry lookup.
    """

    GROUP_NAME = "planning"
    _ENTITY_TYPES = [GOAL, DECISION, ALTERNATIVE, ASSUMPTION, CONSTRAINT]

    def extract(self, text: str) -> list[BaseEntity]:
        """Extract planning entities from natural language text.

        Pipeline:
        1. Keyword classification to detect which entities are present
        2. AIHarness dispatch per detected type
        3. Confidence scoring (D-09) with threshold filtering
        4. Sort by confidence descending, ties by D-10 priority

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


def _classify_entity_types(text: str) -> dict[str, float]:
    """Classify which entity types are described in the text.

    Uses keyword matching against curated domain-specific keyword lists.
    Returns certainty scores in [0.0, 1.0] for matching types.

    The scoring formula:
    - Base = matches / len(keywords) * 2 (capped at 1.0)
    - Boost = +0.3 per keyword match (capped at +0.6)
    - Final = min(1.0, base + boost)

    This means 1 keyword match ~ 0.3-0.5, 2+ matches ~ 0.6-1.0.

    Args:
        text: Input text.

    Returns:
        Dict of {type_name: certainty} for types above zero.
    """
    text_lower = text.lower()
    result: dict[str, float] = {}

    for type_name, keywords in _CLASSIFICATION_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        if matches == 0:
            continue

        base = min(1.0, (matches / max(len(keywords), 1)) * 2.0)
        boost = 0.3 * min(matches, 2)
        certainty = min(1.0, base + boost)
        result[type_name] = round(certainty, 2)

    return result