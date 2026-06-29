"""Pydantic v2 entity models mirroring the SDLC graph ontology.

All 21 entity types from the reference schema are defined here.
Each model uses model_config dict.
BaseGraphEntity subclasses include Cognee-compatible fields.
Every model carries a confidence: float field.
"""

from datetime import datetime
from typing import Any, Optional, get_args

from pydantic import BaseModel, Field, field_validator


# Base classes

class BaseEntity(BaseModel):
    """Abstract base for all entity types.

    Carries extraction metadata (confidence) separate from graph properties.
    Extra fields are forbidden to catch schema mismatches early.
    """

    model_config = {"from_attributes": True, "extra": "forbid"}

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score (0.0-1.0), distinct from graph schema confidence",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is clamped to [0.0, 1.0]."""
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class BaseGraphEntity(BaseEntity):
    """Base for entities persisted to the graph with Cognee-compatible fields.

    These fields support vector search (embedding), filtering (tags),
    and cluster analysis (similarity_cluster_id).
    """

    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding for semantic search",
    )
    description_embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding for description field",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for filtering and categorization",
    )
    similarity_cluster_id: Optional[str] = Field(
        default=None,
        description="Cluster identifier for similar entity grouping",
    )


# PLANNING GROUP

class GOAL(BaseGraphEntity):
    """Desired business or technical outcome.

    From the SDLC schema: a GOAL describes what the team aims to achieve,
    with priority, owner, and measurable success criteria.
    """

    id: str = Field(..., min_length=1, description="Unique goal identifier")
    title: str = Field(..., min_length=1, description="Short goal title")
    description: str = Field(..., min_length=1, description="Detailed goal description")
    priority: Optional[str] = Field(default=None, description="Priority level (High/Medium/Low)")
    created_at: Optional[datetime] = Field(default=None, description="When the goal was defined")
    owner_team: Optional[str] = Field(default=None, description="Owning team name")
    project_id: Optional[str] = Field(default=None, description="Associated project ID")
    outcome_summary: Optional[str] = Field(
        default=None, description="Summary of achieved outcome"
    )


class DECISION(BaseGraphEntity):
    """A chosen path with rationale.

    Note: the graph schema has its own 'confidence' field on DECISION.
    We name it 'graph_confidence' to avoid collision with the extraction
    confidence inherited from BaseEntity (D-06).
    """

    id: str = Field(..., min_length=1, description="Unique decision identifier")
    title: str = Field(..., min_length=1, description="Decision title")
    decision_type: str = Field(..., min_length=1, description="Type of decision (Architecture/Process/Etc)")
    rationale: str = Field(..., min_length=1, description="Rationale behind the decision")
    created_at: Optional[datetime] = Field(default=None, description="When the decision was made")
    status: Optional[str] = Field(default=None, description="Decision status (Proposed/Accepted/Rejected)")
    graph_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Graph schema's own confidence score (separate from extraction confidence)",
    )


class ALTERNATIVE(BaseEntity):
    """Considered option that was not chosen."""

    id: str = Field(..., min_length=1, description="Unique alternative identifier")
    description: str = Field(..., min_length=1, description="Description of the alternative")
    pros: list[str] = Field(default_factory=list, description="Advantages of this alternative")
    cons: list[str] = Field(default_factory=list, description="Disadvantages of this alternative")
    score: Optional[float] = Field(default=None, description="Evaluation score")


class ASSUMPTION(BaseEntity):
    """Hypothesis underpinning decisions or plans."""

    id: str = Field(..., min_length=1, description="Unique assumption identifier")
    assumption_text: str = Field(..., min_length=1, description="The assumption statement")
    risk_if_wrong: Optional[str] = Field(default=None, description="Risk if the assumption is incorrect")
    belief_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Subjective belief in assumption correctness (separate from extraction confidence)")
    created_at: Optional[datetime] = Field(default=None, description="When the assumption was recorded")
    owner_team: Optional[str] = Field(default=None, description="Owning team")


class CONSTRAINT(BaseEntity):
    """Requirement or limit shaping solutions."""

    id: str = Field(..., min_length=1, description="Unique constraint identifier")
    constraint_type: str = Field(..., min_length=1, description="Type (Budget/Time/Resource/Legal/Etc)")
    description: str = Field(..., min_length=1, description="Detailed constraint description")
    severity: Optional[str] = Field(default=None, description="Severity level")
    created_at: Optional[datetime] = Field(default=None, description="When the constraint was identified")
    imposed_by: Optional[str] = Field(default=None, description="Who or what imposed the constraint")


# DEVELOPMENT GROUP

class IMPLEMENTATION(BaseGraphEntity):
    """Concrete design/build unit realizing a decision."""

    id: str = Field(..., min_length=1, description="Unique implementation identifier")
    component: str = Field(..., min_length=1, description="Affected component or module")
    approach: str = Field(..., min_length=1, description="Implementation approach description")
    started_at: Optional[datetime] = Field(default=None, description="When implementation started")
    completed_at: Optional[datetime] = Field(default=None, description="When implementation completed")
    status: Optional[str] = Field(default=None, description="Implementation status")
    version: Optional[str] = Field(default=None, description="Version number")
    outcome_summary: Optional[str] = Field(default=None, description="Summary of implementation outcome")


class PR(BaseEntity):
    """Code change proposal (Pull Request)."""

    id: str = Field(..., min_length=1, description="Unique PR identifier")
    pr_number: Optional[int] = Field(default=None, description="PR number in the repository")
    title: str = Field(..., min_length=1, description="PR title")
    branch: Optional[str] = Field(default=None, description="Source branch")
    repository: Optional[str] = Field(default=None, description="Repository name")
    created_at: Optional[datetime] = Field(default=None, description="When PR was created")
    merged_at: Optional[datetime] = Field(default=None, description="When PR was merged")
    status: Optional[str] = Field(default=None, description="PR status (Open/Merged/Closed/Draft)")


class COMMIT(BaseEntity):
    """Atomic code change."""

    id: str = Field(..., min_length=1, description="Commit hash")
    message: str = Field(..., min_length=1, description="Commit message")
    authored_at: Optional[datetime] = Field(default=None, description="When the commit was authored")
    repository: Optional[str] = Field(default=None, description="Repository name")
    files_changed: Optional[int] = Field(default=None, ge=0, description="Number of files changed")
    additions: Optional[int] = Field(default=None, ge=0, description="Lines added")
    deletions: Optional[int] = Field(default=None, ge=0, description="Lines deleted")


# OPERATIONS GROUP

class DEPLOYMENT(BaseEntity):
    """Delivery of changes to a target environment."""

    id: str = Field(..., min_length=1, description="Unique deployment identifier")
    environment: str = Field(..., min_length=1, description="Target environment (prod/staging/dev)")
    deployed_at: Optional[datetime] = Field(default=None, description="When the deployment occurred")
    status: Optional[str] = Field(default=None, description="Deployment status (Succeeded/Failed/Rolling)")
    strategy: Optional[str] = Field(default=None, description="Deployment strategy (Rolling/BlueGreen/Canary)")
    version: Optional[str] = Field(default=None, description="Deployed version")
    build_id: Optional[str] = Field(default=None, description="CI build identifier")


class DEPLOYMENT_METRIC(BaseEntity):
    """Metric captured during or after deployment."""

    id: str = Field(..., min_length=1, description="Unique metric identifier")
    p95_latency_ms: Optional[float] = Field(default=None, ge=0.0, description="p95 latency in milliseconds")
    error_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Error rate (0.0-1.0)")
    throughput_rps: Optional[int] = Field(default=None, ge=0, description="Throughput in requests per second")
    window: Optional[str] = Field(default=None, description="Measurement window (e.g. '5m')")
    success: Optional[bool] = Field(default=None, description="Whether the deployment was successful")


class WARNING(BaseEntity):
    """Non-fatal anomaly or risk detected during build/deploy/run."""

    id: str = Field(..., min_length=1, description="Unique warning identifier")
    warning_type: str = Field(..., min_length=1, description="Warning category")
    message: str = Field(..., min_length=1, description="Warning message")
    created_at: Optional[datetime] = Field(default=None, description="When the warning was raised")
    severity: Optional[str] = Field(default=None, description="Severity (Low/Medium/High/Critical)")
    source: Optional[str] = Field(default=None, description="Warning source")


# INCIDENT GROUP

class INCIDENT(BaseGraphEntity):
    """Service degradation or outage requiring response."""

    id: str = Field(..., min_length=1, description="Unique incident identifier")
    incident_type: Optional[str] = Field(
        default=None, description="Type (Latency/Outage/ErrorBurst/Security/Performance)"
    )
    severity: Optional[str] = Field(default=None, description="Severity (Low/Medium/High/Critical)")
    detected_at: Optional[datetime] = Field(default=None, description="When the incident was detected")
    resolved_at: Optional[datetime] = Field(default=None, description="When the incident was resolved")
    status: Optional[str] = Field(default=None, description="Incident status")
    summary: str = Field(..., min_length=1, description="Summary of the incident")
    outcome_summary: Optional[str] = Field(default=None, description="Post-incident outcome summary")


class BUG(BaseEntity):
    """Defect tracked in issue management."""

    id: str = Field(..., min_length=1, description="Unique bug identifier")
    key: Optional[str] = Field(default=None, description="Issue tracking key (e.g. 'PROJ-123')")
    title: str = Field(..., min_length=1, description="Bug title")
    severity: Optional[str] = Field(default=None, description="Severity level")
    status: Optional[str] = Field(default=None, description="Bug status")
    created_at: Optional[datetime] = Field(default=None, description="When the bug was reported")
    resolved_at: Optional[datetime] = Field(default=None, description="When the bug was resolved")


class ROOT_CAUSE(BaseEntity):
    """Underlying cause of an incident or bug."""

    id: str = Field(..., min_length=1, description="Unique root cause identifier")
    cause_type: str = Field(
        ..., min_length=1,
        description="Cause type (Design/Code/Configuration/Infrastructure/Process/External)",
    )
    description: str = Field(..., min_length=1, description="Root cause description")
    identified_at: Optional[datetime] = Field(default=None, description="When the root cause was identified")


class FIX(BaseEntity):
    """Change addressing a root cause or bug."""

    id: str = Field(..., min_length=1, description="Unique fix identifier")
    description: str = Field(..., min_length=1, description="Fix description")
    created_at: Optional[datetime] = Field(default=None, description="When the fix was created")
    applied_at: Optional[datetime] = Field(default=None, description="When the fix was applied")
    status: Optional[str] = Field(default=None, description="Fix status (Applied/Pending/Rejected)")


class ROLLBACK(BaseEntity):
    """Revert of a deployment to a previous state."""

    id: str = Field(..., min_length=1, description="Unique rollback identifier")
    reason: str = Field(..., min_length=1, description="Reason for the rollback")
    enacted_at: Optional[datetime] = Field(default=None, description="When the rollback was enacted")
    environment: Optional[str] = Field(default=None, description="Affected environment")
    status: Optional[str] = Field(default=None, description="Rollback status (Completed/InProgress/Failed)")


# LEARNING GROUP

class LESSON(BaseGraphEntity):
    """Institutionalized learning informing future work."""

    id: str = Field(..., min_length=1, description="Unique lesson identifier")
    lesson_text: str = Field(..., min_length=1, description="The core learning")
    category: str = Field(
        ..., min_length=1,
        description="Category (Reliability/Performance/Security/Process/Architecture/Collaboration/Other)",
    )
    documented_at: Optional[datetime] = Field(default=None, description="When the lesson was documented")
    actionable_items: list[str] = Field(default_factory=list, description="Specific action items")


# COLLABORATION GROUP

class ENGINEER(BaseEntity):
    """Individual contributor or operator."""

    id: str = Field(..., min_length=1, description="Unique engineer identifier")
    name: str = Field(..., min_length=1, description="Engineer name")
    role: Optional[str] = Field(default=None, description="Job role or title")
    team: Optional[str] = Field(default=None, description="Team name")
    email: Optional[str] = Field(default=None, description="Email address")


class TEAM(BaseEntity):
    """Group of engineers with shared ownership."""

    id: str = Field(..., min_length=1, description="Unique team identifier")
    team_name: str = Field(..., min_length=1, description="Team name")
    org: Optional[str] = Field(default=None, description="Organization name")
    function: Optional[str] = Field(default=None, description="Team function")


class MEETING(BaseEntity):
    """Scheduled discussion that can lead to decisions."""

    id: str = Field(..., min_length=1, description="Unique meeting identifier")
    meeting_type: str = Field(..., min_length=1, description="Meeting type (Standup/Retro/DesignReview/Etc)")
    occurred_at: Optional[datetime] = Field(default=None, description="When the meeting occurred")
    recording_url: Optional[str] = Field(default=None, description="URL to meeting recording")
    participants: list[str] = Field(default_factory=list, description="List of participant names")


class DISCUSSION(BaseEntity):
    """Asynchronous conversation."""

    id: str = Field(..., min_length=1, description="Unique discussion identifier")
    topic: str = Field(..., min_length=1, description="Discussion topic")
    started_at: Optional[datetime] = Field(default=None, description="When the discussion started")
    url: Optional[str] = Field(default=None, description="URL to the discussion thread")
    channel: Optional[str] = Field(default=None, description="Channel or platform where discussion occurred")


class PLATFORM(BaseEntity):
    """Tool where discussions and documents are captured."""

    id: str = Field(..., min_length=1, description="Unique platform identifier")
    name: str = Field(..., min_length=1, description="Platform name")
    platform_type: Optional[str] = Field(default=None, description="Type (Chat/Docs/Ticketing/CI)")
    url: Optional[str] = Field(default=None, description="Platform URL")


class SERVICE(BaseEntity):
    """Deployable system component owned by a team."""

    id: str = Field(..., min_length=1, description="Unique service identifier")
    service_name: str = Field(..., min_length=1, description="Service name")
    owner_team: Optional[str] = Field(default=None, description="Owning team")
    tier: Optional[str] = Field(default=None, description="Service tier (Tier-1/Tier-2/Tier-3)")


class PROJECT(BaseEntity):
    """Organizational container for related work."""

    id: str = Field(..., min_length=1, description="Unique project identifier")
    project_key: str = Field(..., min_length=1, description="Short project code (e.g. 'PLAT-42')")
    project_name: str = Field(..., min_length=1, description="Full project name")
    owner_team: Optional[str] = Field(default=None, description="Owning team")
    status: Optional[str] = Field(default=None, description="Project status (Active/Completed/OnHold)")


class TECHNOLOGY(BaseGraphEntity):
    """Library, service, or platform dependency."""

    id: str = Field(..., min_length=1, description="Unique technology identifier")
    tech_name: str = Field(..., min_length=1, description="Technology name")
    version: Optional[str] = Field(default=None, description="Version string")
    category: Optional[str] = Field(
        default=None,
        description="Category (Cache/Database/Queue/Language/Framework/Observability/Infra)",
    )
    vendor: Optional[str] = Field(default=None, description="Vendor or maintainer")


# Utility exports

# Build EntityType literal from all model class names
ENTITY_TYPE_NAMES = [
    "GOAL", "DECISION", "ALTERNATIVE", "ASSUMPTION", "CONSTRAINT",
    "IMPLEMENTATION", "PR", "COMMIT",
    "DEPLOYMENT", "DEPLOYMENT_METRIC", "WARNING",
    "INCIDENT", "BUG", "ROOT_CAUSE", "FIX", "ROLLBACK",
    "LESSON",
    "ENGINEER", "PROJECT", "TECHNOLOGY", "SERVICE",
    "MEETING", "DISCUSSION", "TEAM", "PLATFORM",
]

# Import typing for Literal
from typing import Literal  # noqa: E402

EntityType = Literal[
    "GOAL", "DECISION", "ALTERNATIVE", "ASSUMPTION", "CONSTRAINT",
    "IMPLEMENTATION", "PR", "COMMIT",
    "DEPLOYMENT", "DEPLOYMENT_METRIC", "WARNING",
    "INCIDENT", "BUG", "ROOT_CAUSE", "FIX", "ROLLBACK",
    "LESSON",
    "ENGINEER", "PROJECT", "TECHNOLOGY", "SERVICE",
    "MEETING", "DISCUSSION", "TEAM", "PLATFORM",
]

ENTITY_TYPE_MAP: dict[str, type[BaseEntity]] = {
    "GOAL": GOAL,
    "DECISION": DECISION,
    "ALTERNATIVE": ALTERNATIVE,
    "ASSUMPTION": ASSUMPTION,
    "CONSTRAINT": CONSTRAINT,
    "IMPLEMENTATION": IMPLEMENTATION,
    "PR": PR,
    "COMMIT": COMMIT,
    "DEPLOYMENT": DEPLOYMENT,
    "DEPLOYMENT_METRIC": DEPLOYMENT_METRIC,
    "WARNING": WARNING,
    "INCIDENT": INCIDENT,
    "BUG": BUG,
    "ROOT_CAUSE": ROOT_CAUSE,
    "FIX": FIX,
    "ROLLBACK": ROLLBACK,
    "LESSON": LESSON,
    "ENGINEER": ENGINEER,
    "PROJECT": PROJECT,
    "TECHNOLOGY": TECHNOLOGY,
    "SERVICE": SERVICE,
    "MEETING": MEETING,
    "DISCUSSION": DISCUSSION,
    "TEAM": TEAM,
    "PLATFORM": PLATFORM,
}

__all__ = [
    "BaseEntity",
    "BaseGraphEntity",
    "GOAL", "DECISION", "ALTERNATIVE", "ASSUMPTION", "CONSTRAINT",
    "IMPLEMENTATION", "PR", "COMMIT",
    "DEPLOYMENT", "DEPLOYMENT_METRIC", "WARNING",
    "INCIDENT", "BUG", "ROOT_CAUSE", "FIX", "ROLLBACK",
    "LESSON",
    "ENGINEER", "PROJECT", "TECHNOLOGY", "SERVICE",
    "MEETING", "DISCUSSION", "TEAM", "PLATFORM",
    "EntityType",
    "ENTITY_TYPE_MAP",
]