import uuid
import json
from datetime import datetime
from enum import Enum
from sqlalchemy import String, ForeignKey, Float, Text, Boolean, JSON, Integer, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.models.base import Base
from typing import TYPE_CHECKING, Dict, Any, List

if TYPE_CHECKING:
    from .user import User
    from .garden import Garden

class ProjectPermission(str, Enum):
    """Project permission levels."""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

class ProjectStatus(str, Enum):
    """Project status."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class Project(Base):
    """Project model with versioning and collaboration support."""
    __tablename__ = 'projects'

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ProjectStatus] = mapped_column(String(50), default=ProjectStatus.DRAFT, nullable=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    
    # Metadata
    location: Mapped[str | None] = mapped_column(String(500), nullable=True)
    climate_zone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    soil_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    garden_size: Mapped[float | None] = mapped_column(Float, nullable=True)  # in square meters
    
    # Layout and data
    layout_data: Mapped[Dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    plant_data: Mapped[Dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    irrigation_data: Mapped[Dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    
    # Versioning
    current_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_modified_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    
    # Collaboration
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    allow_comments: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    allow_forking: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Relationships
    owner: Mapped["User"] = relationship("User", foreign_keys=[owner_id], back_populates="owned_projects")
    last_modified_user: Mapped["User | None"] = relationship("User", foreign_keys=[last_modified_by])
    members: Mapped[List["ProjectMember"]] = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    versions: Mapped[List["ProjectVersion"]] = relationship("ProjectVersion", back_populates="project", cascade="all, delete-orphan")
    comments: Mapped[List["ProjectComment"]] = relationship("ProjectComment", back_populates="project", cascade="all, delete-orphan")
    gardens: Mapped[List["Garden"]] = relationship("Garden", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', status='{self.status}')>"

    @validates('name')
    def validate_name(self, key, name):
        if not name or len(name.strip()) == 0:
            raise ValueError("Project name cannot be empty")
        if len(name) > 255:
            raise ValueError("Project name cannot exceed 255 characters")
        return name.strip()

class ProjectMember(Base):
    """Project member with permissions."""
    __tablename__ = 'project_members'

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    permission: Mapped[ProjectPermission] = mapped_column(String(50), default=ProjectPermission.VIEWER, nullable=False)
    invited_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    invited_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="members")
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    inviter: Mapped["User | None"] = relationship("User", foreign_keys=[invited_by])

    def __repr__(self):
        return f"<ProjectMember(project_id={self.project_id}, user_id={self.user_id}, permission='{self.permission}')>"

class ProjectVersion(Base):
    """Project version for git-like versioning."""
    __tablename__ = 'project_versions'

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True, nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    
    # Version data
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    layout_data: Mapped[Dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    plant_data: Mapped[Dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    irrigation_data: Mapped[Dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    
    # Version metadata
    is_tagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    tag_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("project_versions.id"), nullable=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="versions")
    creator: Mapped["User"] = relationship("User", foreign_keys=[created_by])
    parent_version: Mapped["ProjectVersion | None"] = relationship("ProjectVersion", remote_side=[id])

    def __repr__(self):
        return f"<ProjectVersion(project_id={self.project_id}, version={self.version_number}, name='{self.name}')>"

class ProjectComment(Base):
    """Project comments for collaboration."""
    __tablename__ = 'project_comments'

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True, nullable=False)
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    parent_comment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("project_comments.id"), nullable=True)
    
    # Comment data
    content: Mapped[str] = mapped_column(Text, nullable=False)
    position_x: Mapped[float | None] = mapped_column(Float, nullable=True)  # For inline comments
    position_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    element_id: Mapped[str | None] = mapped_column(String(100), nullable=True)  # For element-specific comments
    
    # Metadata
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="comments")
    author: Mapped["User"] = relationship("User", foreign_keys=[author_id])
    resolver: Mapped["User | None"] = relationship("User", foreign_keys=[resolved_by])
    parent_comment: Mapped["ProjectComment | None"] = relationship("ProjectComment", remote_side=[id])
    replies: Mapped[List["ProjectComment"]] = relationship("ProjectComment", back_populates="parent_comment")

    def __repr__(self):
        return f"<ProjectComment(id={self.id}, project_id={self.project_id}, author_id={self.author_id})>"

class ProjectActivity(Base):
    """Project activity log for audit trail."""
    __tablename__ = 'project_activities'

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    
    # Activity data
    activity_type: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'project_created', 'layout_updated'
    description: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project")
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])

    def __repr__(self):
        return f"<ProjectActivity(project_id={self.project_id}, user_id={self.user_id}, type='{self.activity_type}')>"

# Update User model to include project relationships
if TYPE_CHECKING:
    from .user import User

# Add indexes for better performance
Index('idx_project_owner_status', Project.owner_id, Project.status)
Index('idx_project_member_user', ProjectMember.user_id, ProjectMember.project_id)
Index('idx_project_version_number', ProjectVersion.project_id, ProjectVersion.version_number)
Index('idx_project_activity_project', ProjectActivity.project_id, ProjectActivity.created_at) 