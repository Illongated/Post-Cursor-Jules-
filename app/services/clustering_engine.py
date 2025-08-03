import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from app.schemas.irrigation import (
    ClusteringInput, ClusteringResult, IrrigationZone, EquipmentType,
    EquipmentSelectionInput, EquipmentSelectionResult, IrrigationEquipment
)

logger = logging.getLogger(__name__)


class ClusteringMethod(str, Enum):
    """Clustering methods available."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    OPTIMIZATION = "optimization"


@dataclass
class PlantData:
    """Plant data for clustering."""
    plant_id: int
    x: float
    y: float
    water_needs: str  # low, moderate, high
    area_m2: float
    growth_stage: str
    crop_coefficient: float
    
    @property
    def water_needs_numeric(self) -> float:
        """Convert water needs to numeric value."""
        mapping = {"low": 1.0, "moderate": 2.0, "high": 3.0}
        return mapping.get(self.water_needs.lower(), 2.0)


class ClusteringEngine:
    """Advanced clustering engine for irrigation zone optimization."""
    
    def __init__(self):
        """Initialize the clustering engine."""
        self.logger = logging.getLogger(__name__)
        
        # Equipment database (simplified - in real implementation this would come from database)
        self.equipment_database = {
            EquipmentType.DRIP: [
                {"name": "Drip Emitter 2LPH", "flow_rate_lph": 2.0, "spacing_m": 0.3, "cost_per_unit": 0.5},
                {"name": "Drip Emitter 4LPH", "flow_rate_lph": 4.0, "spacing_m": 0.4, "cost_per_unit": 0.6},
                {"name": "Drip Emitter 8LPH", "flow_rate_lph": 8.0, "spacing_m": 0.5, "cost_per_unit": 0.8},
            ],
            EquipmentType.SPRINKLER: [
                {"name": "Spray Head 360°", "flow_rate_lph": 50.0, "spacing_m": 3.0, "cost_per_unit": 2.5},
                {"name": "Spray Head 180°", "flow_rate_lph": 30.0, "spacing_m": 2.5, "cost_per_unit": 2.0},
                {"name": "Spray Head 90°", "flow_rate_lph": 20.0, "spacing_m": 2.0, "cost_per_unit": 1.8},
            ],
            EquipmentType.MICROJET: [
                {"name": "Microjet 15LPH", "flow_rate_lph": 15.0, "spacing_m": 1.5, "cost_per_unit": 1.2},
                {"name": "Microjet 25LPH", "flow_rate_lph": 25.0, "spacing_m": 2.0, "cost_per_unit": 1.5},
                {"name": "Microjet 40LPH", "flow_rate_lph": 40.0, "spacing_m": 2.5, "cost_per_unit": 2.0},
            ]
        }
    
    def preprocess_plant_data(self, plants: List[Dict[str, Any]]) -> List[PlantData]:
        """
        Preprocess plant data for clustering.
        
        Args:
            plants: List of plant dictionaries
            
        Returns:
            List of PlantData objects
        """
        plant_data = []
        for plant in plants:
            plant_data.append(PlantData(
                plant_id=plant["plant_id"],
                x=plant["x"],
                y=plant["y"],
                water_needs=plant.get("water_needs", "moderate"),
                area_m2=plant.get("area_m2", 1.0),
                growth_stage=plant.get("growth_stage", "vegetative"),
                crop_coefficient=plant.get("crop_coefficient", 0.8)
            ))
        return plant_data
    
    def create_feature_matrix(self, plants: List[PlantData]) -> np.ndarray:
        """
        Create feature matrix for clustering.
        
        Args:
            plants: List of plant data
            
        Returns:
            Feature matrix
        """
        features = []
        for plant in plants:
            # Normalize coordinates to 0-1 range
            features.append([
                plant.x,  # X coordinate
                plant.y,  # Y coordinate
                plant.water_needs_numeric,  # Water needs (1-3)
                plant.area_m2,  # Area
                plant.crop_coefficient  # Crop coefficient
            ])
        
        return np.array(features)
    
    def kmeans_clustering(
        self, 
        plants: List[PlantData], 
        max_zones: int,
        min_plants_per_zone: int = 1
    ) -> Tuple[List[List[PlantData]], List[np.ndarray]]:
        """
        Perform K-means clustering on plants.
        
        Args:
            plants: List of plant data
            max_zones: Maximum number of zones
            min_plants_per_zone: Minimum plants per zone
            
        Returns:
            Tuple of (clusters, cluster_centers)
        """
        if len(plants) < min_plants_per_zone:
            # Single cluster if not enough plants
            return [plants], [np.array([0, 0])]
        
        # Determine optimal number of clusters
        max_clusters = min(max_zones, len(plants) // min_plants_per_zone)
        if max_clusters < 1:
            max_clusters = 1
        
        # Create feature matrix
        features = self.create_feature_matrix(plants)
        
        # Try different numbers of clusters and find the best one
        best_score = -1
        best_clusters = None
        best_centers = None
        
        for n_clusters in range(1, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # Check if all clusters have minimum plants
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                if np.all(counts >= min_plants_per_zone):
                    # Calculate silhouette score for quality
                    if n_clusters > 1:
                        score = silhouette_score(features, cluster_labels)
                    else:
                        score = 0
                    
                    if score > best_score:
                        best_score = score
                        best_clusters = cluster_labels
                        best_centers = kmeans.cluster_centers_
                        
            except Exception as e:
                self.logger.warning(f"K-means failed for {n_clusters} clusters: {e}")
                continue
        
        if best_clusters is None:
            # Fallback to single cluster
            return [plants], [np.array([0, 0])]
        
        # Group plants by clusters
        clusters = [[] for _ in range(len(best_centers))]
        for i, label in enumerate(best_clusters):
            clusters[label].append(plants[i])
        
        return clusters, best_centers
    
    def dbscan_clustering(
        self, 
        plants: List[PlantData], 
        eps: float = 2.0,
        min_samples: int = 2
    ) -> Tuple[List[List[PlantData]], List[np.ndarray]]:
        """
        Perform DBSCAN clustering on plants.
        
        Args:
            plants: List of plant data
            eps: Maximum distance between points in the same cluster
            min_samples: Minimum number of samples in a cluster
            
        Returns:
            Tuple of (clusters, cluster_centers)
        """
        if len(plants) < min_samples:
            return [plants], [np.array([0, 0])]
        
        # Create feature matrix (only spatial features for DBSCAN)
        features = np.array([[plant.x, plant.y] for plant in plants])
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Group plants by clusters
        unique_labels = np.unique(cluster_labels)
        clusters = []
        centers = []
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_plants = [plants[i] for i, l in enumerate(cluster_labels) if l == label]
            if len(cluster_plants) >= min_samples:
                clusters.append(cluster_plants)
                
                # Calculate cluster center
                cluster_coords = np.array([[p.x, p.y] for p in cluster_plants])
                center = np.mean(cluster_coords, axis=0)
                centers.append(center)
        
        # Handle noise points as separate clusters
        noise_plants = [plants[i] for i, l in enumerate(cluster_labels) if l == -1]
        if noise_plants:
            clusters.extend([[p] for p in noise_plants])
            centers.extend([[p.x, p.y] for p in noise_plants])
        
        return clusters, centers
    
    def optimize_equipment_selection(
        self, 
        zone_area_m2: float, 
        water_needs: str,
        budget_constraint: Optional[float] = None,
        preferred_equipment_type: Optional[EquipmentType] = None
    ) -> EquipmentSelectionResult:
        """
        Optimize equipment selection for a zone.
        
        Args:
            zone_area_m2: Zone area in square meters
            water_needs: Water needs (low, moderate, high)
            budget_constraint: Budget constraint in USD
            preferred_equipment_type: Preferred equipment type
            
        Returns:
            Equipment selection result
        """
        # Determine water needs factor
        water_needs_mapping = {"low": 0.7, "moderate": 1.0, "high": 1.3}
        water_factor = water_needs_mapping.get(water_needs.lower(), 1.0)
        
        # Calculate required flow rate based on area and water needs
        base_flow_per_m2 = 2.0  # LPH per m² for moderate needs
        required_flow_lph = zone_area_m2 * base_flow_per_m2 * water_factor
        
        best_equipment = None
        best_cost = float('inf')
        best_coverage = 0.0
        
        # Try different equipment types
        equipment_types = [preferred_equipment_type] if preferred_equipment_type else list(EquipmentType)
        
        for eq_type in equipment_types:
            if eq_type not in self.equipment_database:
                continue
                
            for equipment in self.equipment_database[eq_type]:
                # Calculate number of units needed
                coverage_per_unit = equipment["spacing_m"] ** 2
                units_needed = math.ceil(zone_area_m2 / coverage_per_unit)
                
                # Calculate total flow and cost
                total_flow = units_needed * equipment["flow_rate_lph"]
                total_cost = units_needed * equipment["cost_per_unit"]
                
                # Check budget constraint
                if budget_constraint and total_cost > budget_constraint:
                    continue
                
                # Check if flow meets requirements
                if total_flow >= required_flow_lph * 0.8:  # Allow 20% tolerance
                    coverage_efficiency = min(1.0, (units_needed * coverage_per_unit) / zone_area_m2)
                    
                    if total_cost < best_cost or (total_cost == best_cost and coverage_efficiency > best_coverage):
                        best_equipment = equipment
                        best_cost = total_cost
                        best_coverage = coverage_efficiency
        
        if best_equipment is None:
            raise ValueError("No suitable equipment found for given constraints")
        
        # Create equipment selection result
        coverage_per_unit = best_equipment["spacing_m"] ** 2
        units_needed = math.ceil(zone_area_m2 / coverage_per_unit)
        
        return EquipmentSelectionResult(
            recommended_equipment=IrrigationEquipment(
                id="temp_id",  # Will be replaced by actual equipment from database
                name=best_equipment["name"],
                equipment_type=eq_type,
                manufacturer="Generic",
                model="Standard",
                flow_rate_lph=best_equipment["flow_rate_lph"],
                pressure_range_min=1.0,
                pressure_range_max=3.0,
                coverage_radius_m=best_equipment["spacing_m"] / 2,
                spacing_m=best_equipment["spacing_m"],
                cost_per_unit=best_equipment["cost_per_unit"],
                specifications={},
                created_at=None,
                updated_at=None
            ),
            quantity_needed=units_needed,
            total_cost=best_cost,
            coverage_efficiency=best_coverage,
            justification=f"Selected {best_equipment['name']} for optimal coverage and cost efficiency"
        )
    
    def calculate_zone_properties(
        self, 
        plants: List[PlantData], 
        cluster_center: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate properties for an irrigation zone.
        
        Args:
            plants: Plants in the zone
            cluster_center: Cluster center coordinates
            
        Returns:
            Zone properties
        """
        if not plants:
            return {}
        
        # Calculate total area
        total_area = sum(plant.area_m2 for plant in plants)
        
        # Calculate average water needs
        avg_water_needs = np.mean([plant.water_needs_numeric for plant in plants])
        water_needs_str = "moderate"
        if avg_water_needs < 1.5:
            water_needs_str = "low"
        elif avg_water_needs > 2.5:
            water_needs_str = "high"
        
        # Calculate required flow rate
        base_flow_per_m2 = 2.0  # LPH per m²
        water_factor = {"low": 0.7, "moderate": 1.0, "high": 1.3}[water_needs_str]
        required_flow_lph = total_area * base_flow_per_m2 * water_factor
        
        # Calculate operating pressure (based on equipment type and flow)
        operating_pressure_bar = 1.5  # Default pressure
        
        return {
            "total_area_m2": total_area,
            "water_needs": water_needs_str,
            "required_flow_lph": required_flow_lph,
            "operating_pressure_bar": operating_pressure_bar,
            "cluster_center_x": float(cluster_center[0]),
            "cluster_center_y": float(cluster_center[1]),
            "plant_ids": [plant.plant_id for plant in plants]
        }
    
    def perform_clustering(
        self, 
        input_data: ClusteringInput
    ) -> ClusteringResult:
        """
        Perform comprehensive clustering analysis.
        
        Args:
            input_data: Clustering input data
            
        Returns:
            Clustering results
        """
        # Preprocess plant data
        plants = self.preprocess_plant_data(input_data.plants)
        
        if not plants:
            return ClusteringResult(
                zones=[],
                cluster_centers=[],
                total_cost=0.0,
                efficiency_score=0.0
            )
        
        # Perform clustering
        clusters, centers = self.kmeans_clustering(
            plants, 
            input_data.max_zones, 
            input_data.min_plants_per_zone
        )
        
        # Create zones from clusters
        zones = []
        total_cost = 0.0
        
        for i, (cluster_plants, center) in enumerate(zip(clusters, centers)):
            # Calculate zone properties
            zone_props = self.calculate_zone_properties(cluster_plants, center)
            
            if not zone_props:
                continue
            
            # Optimize equipment selection
            try:
                equipment_result = self.optimize_equipment_selection(
                    zone_props["total_area_m2"],
                    zone_props["water_needs"]
                )
                total_cost += equipment_result.total_cost
                
                # Create zone
                zone = IrrigationZone(
                    id=f"zone_{i}",  # Temporary ID
                    name=f"Zone {i+1}",
                    description=f"Automatically generated zone with {len(cluster_plants)} plants",
                    status="active",
                    required_flow_lph=zone_props["required_flow_lph"],
                    operating_pressure_bar=zone_props["operating_pressure_bar"],
                    total_area_m2=zone_props["total_area_m2"],
                    cluster_center_x=zone_props["cluster_center_x"],
                    cluster_center_y=zone_props["cluster_center_y"],
                    plant_ids=zone_props["plant_ids"],
                    estimated_cost=equipment_result.total_cost,
                    garden_id=None,  # Will be set by caller
                    created_at=None,
                    updated_at=None
                )
                zones.append(zone)
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize equipment for zone {i}: {e}")
                continue
        
        # Calculate efficiency score
        if zones:
            total_area = sum(zone.total_area_m2 for zone in zones)
            efficiency_score = min(1.0, total_area / (len(zones) * 10))  # Normalize by zone count
        else:
            efficiency_score = 0.0
        
        return ClusteringResult(
            zones=zones,
            cluster_centers=[{"x": float(c[0]), "y": float(c[1])} for c in centers],
            total_cost=total_cost,
            efficiency_score=efficiency_score
        )
    
    def optimize_zone_layout(
        self, 
        zones: List[IrrigationZone],
        garden_bounds: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Optimize the layout of zones within garden bounds.
        
        Args:
            zones: List of irrigation zones
            garden_bounds: Garden boundaries {x_min, x_max, y_min, y_max}
            
        Returns:
            Optimized zone layouts
        """
        layouts = []
        
        for zone in zones:
            # Calculate optimal zone shape based on area
            area = zone.total_area_m2
            aspect_ratio = 1.0  # Square zones by default
            
            # Calculate zone dimensions
            zone_width = math.sqrt(area * aspect_ratio)
            zone_height = area / zone_width
            
            # Position zone around cluster center
            center_x = zone.cluster_center_x
            center_y = zone.cluster_center_y
            
            # Ensure zone fits within garden bounds
            x_min = max(garden_bounds.get("x_min", 0), center_x - zone_width / 2)
            y_min = max(garden_bounds.get("y_min", 0), center_y - zone_height / 2)
            x_max = min(garden_bounds.get("x_max", 100), center_x + zone_width / 2)
            y_max = min(garden_bounds.get("y_max", 100), center_y + zone_height / 2)
            
            layouts.append({
                "zone_id": str(zone.id),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "center_x": center_x,
                "center_y": center_y,
                "area_m2": area
            })
        
        return layouts 