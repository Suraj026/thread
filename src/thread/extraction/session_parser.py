"""Session parser — converts session logs into extractable conversation turns.

Parsing strategy:
1. Primary delimiter: double newline (\n\n)
2. Secondary: timestamp prefix: ^\[.*\] (User|Assistant|System):
3. Tertiary: role prefix: ^(User|Assistant|System):
4. Fallback: entire input as one unknown turn

No cross-turn entity linking (deferred to Phase 2 per D-12).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from thread.extraction.batch_extractor import BatchExtractor
from thread.extraction.models import BaseEntity


# Regex patterns for turn boundary detection
TIMESTAMP_PATTERN = re.compile(
    r"^\[.*?\]\s*(User|Assistant|System):", re.MULTILINE
)
ROLE_PATTERN = re.compile(r"^(User|Assistant|System):", re.MULTILINE)


@dataclass
class SessionTurn:
    """A single conversation turn extracted from a session log.

    Attributes:
        index: 0-based turn position in the session.
        speaker: Detected speaker role ('user', 'assistant', 'system', 'unknown').
        text: The raw text content of this turn.
        timestamp: Optional extracted timestamp string.
    """

    index: int
    speaker: str = "unknown"
    text: str = ""
    timestamp: Optional[str] = None


@dataclass
class SessionTurnResult:
    """Result of extracting entities from a single turn.

    Attributes:
        turn: The session turn metadata.
        entities: List of extracted entities (may be empty).
    """

    turn: SessionTurn
    entities: list[BaseEntity] = field(default_factory=list)


class SessionParser:
    """Parses session logs and extracts entities per conversation turn."""

    def __init__(self, batch_extractor: Optional[BatchExtractor] = None):
        """Initialize with an optional BatchExtractor."""
        self.batch_extractor = batch_extractor or BatchExtractor()

    def split_turns(self, session_text: str) -> list[SessionTurn]:
        """Split session text into individual conversation turns.

        Detection priority:
        1. Double newline (\n\n) segments
        2. Timestamp-prefixed role lines: [2026-01-01] User: ...
        3. Role-prefixed lines: User: ... / Assistant: ...
        4. Fallback: entire input as one unknown turn

        Args:
            session_text: Raw session log text.

        Returns:
            List of SessionTurn objects in order.
        """
        if not session_text.strip():
            return []

        turns: list[SessionTurn] = []

        # Strategy 1: Split on double-newlines first
        blocks = re.split(r"\n\s*\n", session_text.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Try to detect turns within this block using patterns
            detected_turns = self._detect_turns_from_block(block)
            turns.extend(detected_turns)

        # If no turns were detected, create one unknown turn
        if not turns:
            turns.append(SessionTurn(index=0, speaker="unknown", text=session_text.strip()))

        # Re-index to ensure 0-based sequential indices
        for i, turn in enumerate(turns):
            turn.index = i

        return turns

    def _detect_turns_from_block(self, block: str) -> list[SessionTurn]:
        """Try to detect multiple turns within a single text block.

        Uses timestamp and role patterns to split further.

        Args:
            block: A single text block (from double-newline split).

        Returns:
            List of SessionTurn objects.
        """
        turns: list[SessionTurn] = []

        # Try timestamp pattern first: [timestamp] Role: text
        timestamp_matches = list(TIMESTAMP_PATTERN.finditer(block))
        if len(timestamp_matches) >= 2:
            # Multiple timestamp-prefixed turns in this block
            for i, match in enumerate(timestamp_matches):
                start = match.start()
                end = timestamp_matches[i + 1].start() if i + 1 < len(timestamp_matches) else len(block)
                turn_text = block[start:end].strip()

                # Extract timestamp
                ts_match = re.match(r"\[(.*?)\]", match.group(0))
                timestamp = ts_match.group(1) if ts_match else None

                speaker = match.group(1).lower() if match.group(1) else "unknown"
                # Ensure valid speaker value
                if speaker not in ("user", "assistant", "system"):
                    speaker = "unknown"

                turns.append(SessionTurn(
                    index=0,  # will be re-indexed
                    speaker=speaker,
                    text=turn_text,
                    timestamp=timestamp,
                ))
            return turns

        # Try role pattern: User:/Assistant:/System: at line start
        role_matches = list(ROLE_PATTERN.finditer(block))
        if len(role_matches) >= 2:
            for i, match in enumerate(role_matches):
                start = match.start()
                end = role_matches[i + 1].start() if i + 1 < len(role_matches) else len(block)
                turn_text = block[start:end].strip()

                speaker = match.group(1).lower()
                if speaker not in ("user", "assistant", "system"):
                    speaker = "unknown"

                turns.append(SessionTurn(
                    index=0,
                    speaker=speaker,
                    text=turn_text,
                ))
            return turns

        # Single timestamp-prefixed turn
        if timestamp_matches:
            match = timestamp_matches[0]
            ts_match = re.match(r"\[(.*?)\]", match.group(0))
            timestamp = ts_match.group(1) if ts_match else None
            speaker = match.group(1).lower() if match.group(1) else "unknown"
            if speaker not in ("user", "assistant", "system"):
                speaker = "unknown"
            turns.append(SessionTurn(
                index=0,
                speaker=speaker,
                text=block,
                timestamp=timestamp,
            ))
            return turns

        # Single role-prefixed turn
        if role_matches:
            match = role_matches[0]
            speaker = match.group(1).lower()
            if speaker not in ("user", "assistant", "system"):
                speaker = "unknown"
            turns.append(SessionTurn(
                index=0,
                speaker=speaker,
                text=block,
            ))
            return turns

        # No patterns matched — return as unknown turn
        return []

    def extract_all(self, session_text: str) -> list[SessionTurnResult]:
        """Extract entities from all turns in a session log.

        Each turn is independently evaluated with no cross-turn
        entity linking (D-12).

        Args:
            session_text: Raw session log text.

        Returns:
            List of SessionTurnResult objects, one per turn, in order.
        """
        turns = self.split_turns(session_text)

        results: list[SessionTurnResult] = []
        for turn in turns:
            try:
                batch_result = self.batch_extractor.run(turn.text)
                entities = batch_result.entities
            except Exception:
                entities = []

            results.append(SessionTurnResult(turn=turn, entities=entities))

        return results

    def extract_turn(self, turn_text: str) -> SessionTurnResult:
        """Extract entities from a single turn text.

        Convenience wrapper for non-session callers.
        Treats the entire input as one unknown turn.

        Args:
            turn_text: Single turn text.

        Returns:
            SessionTurnResult with one turn and its extracted entities.
        """
        turn = SessionTurn(index=0, speaker="unknown", text=turn_text.strip())
        try:
            batch_result = self.batch_extractor.run(turn.text)
            entities = batch_result.entities
        except Exception:
            entities = []

        return SessionTurnResult(turn=turn, entities=entities)