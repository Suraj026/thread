"""Entity extractors package — one extractor per SDLC phase group.

Provides:
- EXTRACTOR_REGISTRY: mapping of group names to extractor classes
- all_extractors(): factory function returning one instance of each extractor
"""

from .planning import PlanningExtractor
from .development import DevelopmentExtractor
from .operations import OperationsExtractor
from .incident import IncidentExtractor
from .learning import LearningExtractor
from .collaboration import CollaborationExtractor

EXTRACTOR_REGISTRY: dict[str, type] = {
    "planning": PlanningExtractor,
    "development": DevelopmentExtractor,
    "operations": OperationsExtractor,
    "incident": IncidentExtractor,
    "learning": LearningExtractor,
    "collaboration": CollaborationExtractor,
}

__all__ = [
    "PlanningExtractor",
    "DevelopmentExtractor",
    "OperationsExtractor",
    "IncidentExtractor",
    "LearningExtractor",
    "CollaborationExtractor",
    "EXTRACTOR_REGISTRY",
    "all_extractors",
]


def all_extractors():
    """Create one of each registered extractor with default settings.

    Returns:
        list of BaseExtractor instances, one per registered group.
    """
    from thread.extraction.config import Settings

    settings = Settings()
    return [cls(settings) for cls in EXTRACTOR_REGISTRY.values()]