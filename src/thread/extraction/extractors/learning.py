"""LearningExtractor — extracts learning-phase entities.

Handles: LESSON (single entity type — direct extraction, no classification).
"""

from thread.extraction.base import AIHarness, BaseExtractor
from thread.extraction.models import LESSON, BaseEntity
from thread.extraction.extractors.planning import _TYPE_PRIORITY


class LearningExtractor(BaseExtractor):
    """Extracts LESSON entities from natural language text.

    LESSON is a single entity type, so no classification step is needed.
    Uses two extraction passes:
    1. Direct extraction for explicit lesson statements
    2. Implicit extraction via a fresh harness to catch embedded learnings

    Boosts confidence by 0.1 if explicit lesson-language keywords are present.
    """

    GROUP_NAME = "learning"
    _ENTITY_TYPES = [LESSON]

    # Keywords that signal explicit lesson statements
    _LESSON_KEYWORDS = [
        "lesson learned",
        "key takeaway",
        "retrospective",
        "next time",
        "never again",
        "we learned",
        "postmortem",
        "improvement",
        "action item",
    ]

    def extract(self, text: str) -> list[BaseEntity]:
        """Extract LESSON entities from natural language text.

        Pipeline:
        1. Direct extraction via cached AIHarness
        2. Implicit extraction via separate harness
        3. 3-signal confidence scoring (D-09) with keyword boost
        4. Threshold filtering

        Args:
            text: Natural language text.

        Returns:
            List of LESSON entities with confidence >= threshold.
        """
        if not text or not text.strip():
            return []

        results: list[BaseEntity] = []

        # Pass 1: Direct extraction
        harness = self._create_harness(LESSON)
        lesson_direct = harness.run(text)
        if lesson_direct is not None:
            self._score_and_filter(lesson_direct, text, results)

        # Pass 2: Implicit learning extraction
        # Create a fresh harness so a new Agent is built with a different prompt
        implicit_harness = AIHarness(LESSON, self.settings)
        # Force the agent to rebuild with implicit-learning system prompt
        # by running on a slightly different input to trigger lazy init
        implicit_text = f"Identify embedded lessons in this text:\n\n{text}"
        lesson_implicit = implicit_harness.run(implicit_text)
        if lesson_implicit is not None:
            self._score_and_filter(lesson_implicit, text, results)

        results.sort(
            key=lambda e: (
                -e.confidence,
                _TYPE_PRIORITY.get(type(e).__name__, 999),
            )
        )
        return results

    def _score_and_filter(
        self, entity: LESSON, text: str, results: list[BaseEntity]
    ) -> None:
        """Score confidence for a LESSON entity and append if above threshold.

        Args:
            entity: Extracted LESSON entity.
            text: Original input text (for keyword detection).
            results: Mutable list to append to if above threshold.
        """
        entity_dict = entity.model_dump() if hasattr(entity, "model_dump") else {}
        required_keys = {
            field_name
            for field_name, field_info in LESSON.model_fields.items()
            if field_info.is_required()
        }
        entity.confidence = self._score_confidence(
            instance=entity,
            llm_confidence=1.0,
            extracted_fields=entity_dict,
            entity_certainty=1.0,
            required_keys=required_keys,
        )

        # Confidence boost for explicit lesson-language keywords
        if self._has_lesson_keywords(text):
            entity.confidence = min(1.0, entity.confidence + 0.1)

        if entity.confidence >= self.settings.extraction_confidence_threshold:
            results.append(entity)

    def _has_lesson_keywords(self, text: str) -> bool:
        """Check if text contains explicit lesson-language keywords.

        Args:
            text: Input text to check.

        Returns:
            True if any lesson keyword is present.
        """
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._LESSON_KEYWORDS)
