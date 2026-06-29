"""Test configuration and shared fixtures."""

import os

import pytest
from thread.extraction.config import Settings


def pytest_configure():
    """Set test environment variables before any tests run."""
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def settings() -> Settings:
    """Create a Settings instance with test API key."""
    return Settings(openrouter_api_key="test-key")


@pytest.fixture
def sample_goal_text() -> str:
    """Realistic natural language description of a GOAL."""
    return (
        "We need to reduce checkout page latency to under 200ms p95. "
        "This is a high-priority initiative for the Platform team (project PLAT-42). "
        "Success will be measured by improved conversion rates and reduced cart abandonment."
    )


@pytest.fixture
def sample_decision_text() -> str:
    """Realistic natural language description of a DECISION."""
    return (
        "After evaluating Redis and Memcached, the team decided to use Redis for caching "
        "the product catalog. Redis was chosen because of its better support for complex "
        "data structures and built-in persistence options. The migration is planned for Q3."
    )


@pytest.fixture
def sample_incident_text() -> str:
    """Realistic natural language description of an INCIDENT."""
    return (
        "At 14:32 UTC, the checkout service started returning 503 errors. "
        "The incident was triggered by a database connection pool exhaustion caused by "
        "a deployment that increased connection counts without updating pool limits. "
        "The fix was to rollback the deployment and increase the connection pool size. "
        "Root cause: configuration drift between staging and production."
    )