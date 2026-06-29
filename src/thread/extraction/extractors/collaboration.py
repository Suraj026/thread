"""CollaborationExtractor — extracts collaboration-phase entities.

Handles: ENGINEER, TEAM, MEETING, DISCUSSION, PLATFORM, SERVICE,
         PROJECT, TECHNOLOGY.

Special features:
- _resolve_references() post-processing for linked entities
"""

import re

from thread.extraction.base import BaseExtractor
from thread.extraction.models import (
    DISCUSSION,
    ENGINEER,
    MEETING,
    PLATFORM,
    PROJECT,
    SERVICE,
    TEAM,
    TECHNOLOGY,
    BaseEntity,
)
from thread.extraction.extractors.planning import (
    _TYPE_PRIORITY,
    _classify_entity_types,
)


class CollaborationExtractor(BaseExtractor):
    """Extracts collaboration-phase entities from natural language text.

    Detects and extracts ENGINEER, TEAM, MEETING, DISCUSSION, PLATFORM,
    SERVICE, PROJECT, and TECHNOLOGY entities using keyword-based
    classification followed by LLM extraction.

    After initial extraction, _resolve_references() scans the original
    text for patterns like "X from Y team" and creates linked entities.
    """

    GROUP_NAME = "collaboration"
    _ENTITY_TYPES = [
        ENGINEER,
        TEAM,
        MEETING,
        DISCUSSION,
        PLATFORM,
        SERVICE,
        PROJECT,
        TECHNOLOGY,
    ]

    def extract(self, text: str) -> list[BaseEntity]:
        """Extract collaboration entities from natural language text.

        Pipeline:
        1. Keyword classification
        2. AIHarness dispatch per detected type
        3. 3-signal confidence scoring (D-09) with threshold filtering
        4. _resolve_references() for cross-entity linking
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

        # Post-processing: resolve cross-references
        results = self._resolve_references(text, results)

        results.sort(
            key=lambda e: (
                -e.confidence,
                _TYPE_PRIORITY.get(type(e).__name__, 999),
            )
        )
        return results

    def _resolve_references(
        self, text: str, entities: list[BaseEntity]
    ) -> list[BaseEntity]:
        """Resolve cross-references and create linked entities.

        Looks for patterns like "X from Y team" in the original text.
        If the pattern matches and no corresponding ENGINEER or TEAM
        entity already exists, creates new entities with moderate confidence.

        Args:
            text: Original input text.
            entities: Entities extracted so far.

        Returns:
            Extended list of entities with linked references added.
        """
        text_lower = text.lower()
        new_entities: list[BaseEntity] = []

        # Collect existing names/teams to avoid duplicates
        existing_names: set[str] = set()
        existing_teams: set[str] = set()
        for e in entities:
            if isinstance(e, ENGINEER) and e.name:
                existing_names.add(e.name.lower())
            if isinstance(e, TEAM) and e.team_name:
                existing_teams.add(e.team_name.lower())

        # Pattern: "{Name} from {Team Name} team"  or  "{Name} from the {Team Name} team"
        pattern = r"(\w+(?:\s+\w+)?)\s+from\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+team"
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            name = match.group(1).strip()
            team = match.group(2).strip()

            # Create ENGINEER if not already present
            if name not in existing_names:
                safe_id = re.sub(r"[^a-z0-9]", "-", name.lower())
                new_entities.append(
                    ENGINEER(
                        id=f"eng-{safe_id}",
                        name=name.title(),
                        team=team.title(),
                        confidence=0.6,
                    )
                )
                existing_names.add(name)

            # Create TEAM if not already present
            if team not in existing_teams:
                safe_id = re.sub(r"[^a-z0-9]", "-", team.lower())
                new_entities.append(
                    TEAM(
                        id=f"team-{safe_id}",
                        team_name=team.title(),
                        confidence=0.5,
                    )
                )
                existing_teams.add(team)

        return entities + new_entities
