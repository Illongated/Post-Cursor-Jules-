# This file makes the 'models' directory a Python package.
# It also helps Alembic discover the models.

from .base import Base
from .user import User
from .garden import Garden
from .plant import Plant
from .plant_catalog import PlantCatalog
from .project import Project, ProjectMember, ProjectVersion, ProjectComment, ProjectActivity, ProjectPermission, ProjectStatus
from .irrigation import (
    IrrigationEquipment, IrrigationZone, IrrigationZoneEquipment, IrrigationPipe,
    IrrigationSchedule, WeatherData, IrrigationProject, EquipmentType, PipeMaterial,
    ZoneStatus, ScheduleType
)

__all__ = [
    "Base",
    "User", 
    "Garden",
    "Plant",
    "PlantCatalog",
    "Project",
    "ProjectMember", 
    "ProjectVersion",
    "ProjectComment",
    "ProjectActivity",
    "ProjectPermission",
    "ProjectStatus",
    "IrrigationEquipment",
    "IrrigationZone",
    "IrrigationZoneEquipment", 
    "IrrigationPipe",
    "IrrigationSchedule",
    "WeatherData",
    "IrrigationProject",
    "EquipmentType",
    "PipeMaterial",
    "ZoneStatus",
    "ScheduleType"
]
