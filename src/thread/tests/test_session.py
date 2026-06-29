"""Tests for session parsing: turn splitting, per-turn extraction."""

from unittest.mock import patch

import pytest

from thread.extraction.session_parser import (
    SessionParser,
    SessionTurn,
    SessionTurnResult,
)
from thread.extraction.models import GOAL, BaseEntity


def _make_mock_batch_extractor(entities_by_text: dict):
    """Create a mock BatchExtractor that returns entities based on input text.

    Args:
        entities_by_text: Dict mapping text substrings to entity lists.
                         If text contains substring, return those entities.

    Returns:
        A configured MagicMock.
    """
    from unittest.mock import MagicMock

    mock = MagicMock()

    def mock_run(text: str):
        from threading import RLock
        from thread.extraction.batch_extractor import BatchResult

        for substr, entities in entities_by_text.items():
            if substr in text:
                return BatchResult(entities=entities)
        return BatchResult(entities=[])

    mock.run = mock_run
    return mock


def test_split_turns_double_newline():
    """Test 1: split_turns splits multi-turn session on double-newline."""
    parser = SessionParser()
    text = "User: first turn\n\nAssistant: second turn\n\nUser: third turn"
    turns = parser.split_turns(text)
    assert len(turns) == 3
    assert turns[0].speaker.lower() == "user"
    assert turns[1].speaker.lower() == "assistant"
    assert turns[2].speaker.lower() == "user"


def test_split_turns_single_line():
    """Test 2: Single-line input produces one turn."""
    parser = SessionParser()
    turns = parser.split_turns("User: just one turn here")
    assert len(turns) == 1
    assert "just one turn" in turns[0].text


def test_split_turns_empty_input():
    """Test 3: Empty input returns empty list."""
    parser = SessionParser()
    assert parser.split_turns("") == []
    assert parser.split_turns("   ") == []
    assert parser.split_turns("\n\n\n") == []


def test_split_turns_timestamp_boundary():
    """Test 4: Timestamp-prefixed lines as turn boundaries."""
    parser = SessionParser()
    text = (
        "[2026-06-29 10:00] User: let's design the API\n"
        "[2026-06-29 10:01] Assistant: I suggest using GraphQL\n"
        "[2026-06-29 10:02] User: agreed, let's proceed"
    )
    turns = parser.split_turns(text)
    assert len(turns) == 3
    assert turns[0].timestamp is not None
    assert "2026-06-29 10:00" in turns[0].timestamp
    assert "GraphQL" in turns[1].text


def test_split_turns_role_prefix():
    """Test 5: Role-prefixed lines as turn boundaries."""
    parser = SessionParser()
    text = "User: what's the latency?\nAssistant: currently 200ms\nUser: that's too high"
    turns = parser.split_turns(text)
    assert len(turns) == 3
    assert "latency" in turns[0].text
    assert "200ms" in turns[1].text
    assert "too high" in turns[2].text


def test_extract_all_returns_turn_results():
    """Test 6: extract_all returns SessionTurnResult with correct structure."""
    mock_extractor = _make_mock_batch_extractor({
        "caching": [
            GOAL(id="g1", title="Add caching", description="Reduce latency", confidence=0.9),
        ],
    })
    parser = SessionParser(batch_extractor=mock_extractor)
    text = "User: we need caching\n\nAssistant: good idea"
    results = parser.extract_all(text)

    assert len(results) == 2
    for result in results:
        assert isinstance(result, SessionTurnResult)
        assert isinstance(result.turn, SessionTurn)
        assert isinstance(result.entities, list)


def test_extract_all_skips_empty_turns():
    """Test 7: Turns with no entities still produce a result with empty list."""
    mock_extractor = _make_mock_batch_extractor({})
    parser = SessionParser(batch_extractor=mock_extractor)
    text = "User: hello\n\nAssistant: hi"
    results = parser.extract_all(text)

    assert len(results) == 2
    for result in results:
        assert result.entities == []  # Not None, not skipped


def test_extract_all_preserves_order():
    """Test 8: Turn results maintain the original session order."""
    mock_extractor = _make_mock_batch_extractor({
        "first": [GOAL(id="g1", title="First", description="first goal", confidence=0.8)],
        "second": [GOAL(id="g2", title="Second", description="second goal", confidence=0.8)],
        "third": [GOAL(id="g3", title="Third", description="third goal", confidence=0.8)],
    })
    parser = SessionParser(batch_extractor=mock_extractor)
    text = "User: first turn\n\nAssistant: second turn\n\nUser: third turn"
    results = parser.extract_all(text)

    assert len(results) == 3
    # Order should match input: first, second, third
    assert results[0].turn.index == 0
    assert results[1].turn.index == 1
    assert results[2].turn.index == 2