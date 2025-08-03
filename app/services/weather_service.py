import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from dataclasses import dataclass

from app.schemas.irrigation import (
    WeatherData, WeatherForecastInput, WeatherForecastResult,
    WeatherDataCreate
)

logger = logging.getLogger(__name__)


@dataclass
class WeatherParameters:
    """Weather parameters for calculations."""
    temperature_c: float
    humidity_percent: float
    wind_speed_kmh: float
    solar_radiation_mj_m2: float
    rainfall_mm: float
    atmospheric_pressure_hpa: float = 1013.25


class WeatherService:
    """Weather service for irrigation scheduling and evapotranspiration calculations."""
    
    def __init__(self):
        """Initialize the weather service."""
        self.logger = logging.getLogger(__name__)
        
        # Weather API configuration (OpenWeatherMap example)
        self.api_key = "your_api_key_here"  # Should be in environment variables
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Evapotranspiration calculation constants
        self.SOLAR_CONSTANT = 0.082  # MJ/m²/min
        self.LATENT_HEAT_VAPORIZATION = 2.45  # MJ/kg
        self.PSYCHROMETRIC_CONSTANT = 0.000665  # kPa/°C
        
        # Crop coefficients for different growth stages
        self.CROP_COEFFICIENTS = {
            "initial": 0.3,
            "vegetative": 0.7,
            "flowering": 1.0,
            "fruiting": 1.1,
            "mature": 0.8,
            "harvest": 0.6
        }
    
    async def fetch_weather_data(
        self, 
        latitude: float, 
        longitude: float,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Fetch weather data from external API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            date: Specific date for historical data (optional)
            
        Returns:
            Weather data dictionary
        """
        try:
            if date:
                # Historical data (requires different endpoint)
                url = f"{self.base_url}/onecall/timemachine"
                params = {
                    "lat": latitude,
                    "lon": longitude,
                    "dt": int(date.timestamp()),
                    "appid": self.api_key,
                    "units": "metric"
                }
            else:
                # Current weather
                url = f"{self.base_url}/weather"
                params = {
                    "lat": latitude,
                    "lon": longitude,
                    "appid": self.api_key,
                    "units": "metric"
                }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self.parse_weather_data(data, date or datetime.now())
                    else:
                        self.logger.error(f"Weather API error: {response.status}")
                        return self.get_mock_weather_data(latitude, longitude, date)
                        
        except Exception as e:
            self.logger.error(f"Error fetching weather data: {e}")
            return self.get_mock_weather_data(latitude, longitude, date)
    
    def parse_weather_data(self, api_data: Dict[str, Any], date: datetime) -> Dict[str, Any]:
        """
        Parse weather data from API response.
        
        Args:
            api_data: Raw API response
            date: Date of the weather data
            
        Returns:
            Parsed weather data
        """
        try:
            # Extract basic weather parameters
            main = api_data.get("main", {})
            weather = api_data.get("weather", [{}])[0]
            wind = api_data.get("wind", {})
            
            temperature_c = main.get("temp", 20.0)
            humidity_percent = main.get("humidity", 60.0)
            pressure_hpa = main.get("pressure", 1013.25)
            wind_speed_kmh = wind.get("speed", 5.0) * 3.6  # Convert m/s to km/h
            
            # Estimate solar radiation (not always available in free API)
            solar_radiation = self.estimate_solar_radiation(
                date, 
                api_data.get("coord", {}).get("lat", 0),
                weather.get("main", "Clear")
            )
            
            # Estimate rainfall
            rainfall_mm = api_data.get("rain", {}).get("1h", 0.0)
            
            return {
                "date": date,
                "temperature_c": temperature_c,
                "humidity_percent": humidity_percent,
                "wind_speed_kmh": wind_speed_kmh,
                "solar_radiation_mj_m2": solar_radiation,
                "rainfall_mm": rainfall_mm,
                "atmospheric_pressure_hpa": pressure_hpa,
                "weather_condition": weather.get("main", "Clear"),
                "description": weather.get("description", "")
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing weather data: {e}")
            return self.get_mock_weather_data(0, 0, date)
    
    def estimate_solar_radiation(
        self, 
        date: datetime, 
        latitude: float, 
        weather_condition: str
    ) -> float:
        """
        Estimate solar radiation based on date, latitude, and weather conditions.
        
        Args:
            date: Date for calculation
            latitude: Latitude coordinate
            weather_condition: Weather condition (Clear, Cloudy, etc.)
            
        Returns:
            Estimated solar radiation in MJ/m²
        """
        # Calculate day of year
        day_of_year = date.timetuple().tm_yday
        
        # Calculate solar declination
        solar_declination = 23.45 * math.sin(math.radians(360/365 * (day_of_year - 80)))
        
        # Calculate sunset hour angle
        sunset_hour_angle = math.acos(-math.tan(math.radians(latitude)) * math.tan(math.radians(solar_declination)))
        
        # Calculate extraterrestrial radiation
        extraterrestrial_radiation = self.SOLAR_CONSTANT * (1 + 0.033 * math.cos(math.radians(360/365 * day_of_year))) * \
                                   (math.cos(math.radians(latitude)) * math.cos(math.radians(solar_declination)) * \
                                    math.sin(sunset_hour_angle) + sunset_hour_angle * math.sin(math.radians(latitude)) * \
                                    math.sin(math.radians(solar_declination)))
        
        # Apply weather condition factor
        weather_factors = {
            "Clear": 0.8,
            "Clouds": 0.6,
            "Rain": 0.4,
            "Snow": 0.3,
            "Thunderstorm": 0.3
        }
        
        weather_factor = weather_factors.get(weather_condition, 0.6)
        
        return extraterrestrial_radiation * weather_factor
    
    def calculate_evapotranspiration(self, weather_params: WeatherParameters) -> float:
        """
        Calculate reference evapotranspiration using FAO Penman-Monteith equation.
        
        Args:
            weather_params: Weather parameters
            
        Returns:
            Reference evapotranspiration in mm/day
        """
        # Convert units
        temperature_k = weather_params.temperature_c + 273.15
        wind_speed_ms = weather_params.wind_speed_kmh / 3.6
        
        # Calculate saturation vapor pressure
        saturation_vapor_pressure = 0.6108 * math.exp(17.27 * weather_params.temperature_c / 
                                                     (weather_params.temperature_c + 237.3))
        
        # Calculate actual vapor pressure
        actual_vapor_pressure = saturation_vapor_pressure * weather_params.humidity_percent / 100
        
        # Calculate vapor pressure deficit
        vapor_pressure_deficit = saturation_vapor_pressure - actual_vapor_pressure
        
        # Calculate slope of vapor pressure curve
        slope_vapor_pressure = 4098 * saturation_vapor_pressure / (temperature_k ** 2)
        
        # Calculate psychrometric constant
        psychrometric_constant = self.PSYCHROMETRIC_CONSTANT * weather_params.atmospheric_pressure_hpa / 1000
        
        # Calculate net radiation (simplified)
        net_radiation = weather_params.solar_radiation_mj_m2 * 0.77 - 0.26 * (1 - weather_params.humidity_percent/100)
        
        # Calculate soil heat flux (simplified)
        soil_heat_flux = 0.1 * net_radiation
        
        # FAO Penman-Monteith equation
        numerator = 0.408 * slope_vapor_pressure * (net_radiation - soil_heat_flux) + \
                   psychrometric_constant * (900 / temperature_k) * wind_speed_ms * vapor_pressure_deficit
        
        denominator = slope_vapor_pressure + psychrometric_constant * (1 + 0.34 * wind_speed_ms)
        
        et0 = numerator / denominator
        
        return max(0, et0)  # Ensure non-negative
    
    def calculate_irrigation_need(
        self, 
        et0: float, 
        crop_coefficient: float,
        rainfall_mm: float,
        soil_moisture_deficit: float = 0.0
    ) -> float:
        """
        Calculate irrigation need based on evapotranspiration and rainfall.
        
        Args:
            et0: Reference evapotranspiration in mm/day
            crop_coefficient: Crop coefficient
            rainfall_mm: Rainfall in mm
            soil_moisture_deficit: Soil moisture deficit in mm
            
        Returns:
            Irrigation need in mm
        """
        # Calculate crop evapotranspiration
        etc = et0 * crop_coefficient
        
        # Calculate effective rainfall (assume 80% efficiency)
        effective_rainfall = rainfall_mm * 0.8
        
        # Calculate irrigation need
        irrigation_need = etc - effective_rainfall + soil_moisture_deficit
        
        return max(0, irrigation_need)
    
    def get_irrigation_recommendations(
        self, 
        weather_data: List[WeatherData],
        crop_type: str = "general",
        growth_stage: str = "vegetative"
    ) -> List[Dict[str, Any]]:
        """
        Generate irrigation recommendations based on weather forecast.
        
        Args:
            weather_data: List of weather data
            crop_type: Type of crop
            growth_stage: Growth stage of the crop
            
        Returns:
            List of irrigation recommendations
        """
        recommendations = []
        crop_coefficient = self.CROP_COEFFICIENTS.get(growth_stage, 0.7)
        
        for weather in weather_data:
            # Create weather parameters
            weather_params = WeatherParameters(
                temperature_c=weather.temperature_c,
                humidity_percent=weather.humidity_percent,
                wind_speed_kmh=weather.wind_speed_kmh,
                solar_radiation_mj_m2=weather.solar_radiation_mj_m2,
                rainfall_mm=weather.rainfall_mm
            )
            
            # Calculate evapotranspiration
            et0 = self.calculate_evapotranspiration(weather_params)
            
            # Calculate irrigation need
            irrigation_need = self.calculate_irrigation_need(
                et0, crop_coefficient, weather.rainfall_mm
            )
            
            # Generate recommendation
            if irrigation_need > 2.0:  # More than 2mm needed
                recommendation = {
                    "date": weather.date,
                    "irrigation_need_mm": irrigation_need,
                    "recommendation": "irrigate",
                    "duration_minutes": int(irrigation_need * 10),  # Rough conversion
                    "reason": f"High evapotranspiration ({et0:.1f} mm) and low rainfall ({weather.rainfall_mm:.1f} mm)"
                }
            elif irrigation_need > 0.5:  # Moderate need
                recommendation = {
                    "date": weather.date,
                    "irrigation_need_mm": irrigation_need,
                    "recommendation": "monitor",
                    "duration_minutes": 0,
                    "reason": f"Moderate water need ({irrigation_need:.1f} mm)"
                }
            else:
                recommendation = {
                    "date": weather.date,
                    "irrigation_need_mm": irrigation_need,
                    "recommendation": "skip",
                    "duration_minutes": 0,
                    "reason": f"Sufficient rainfall ({weather.rainfall_mm:.1f} mm) or low ET ({et0:.1f} mm)"
                }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    async def get_weather_forecast(
        self, 
        input_data: WeatherForecastInput
    ) -> WeatherForecastResult:
        """
        Get weather forecast and irrigation recommendations.
        
        Args:
            input_data: Weather forecast input data
            
        Returns:
            Weather forecast result
        """
        # For demo purposes, use mock data
        # In production, this would fetch from actual weather API
        forecast_data = []
        irrigation_recommendations = []
        total_water_savings = 0.0
        
        # Generate mock forecast data
        for i in range(input_data.days_ahead):
            date = datetime.now() + timedelta(days=i)
            
            # Mock weather data
            weather_data = WeatherData(
                id=f"weather_{i}",
                garden_id=input_data.garden_id,
                date=date,
                temperature_c=20 + 5 * math.sin(i * 0.5),  # Varying temperature
                humidity_percent=60 + 10 * math.sin(i * 0.3),
                rainfall_mm=max(0, 5 * math.sin(i * 0.7)),  # Some rainfall
                wind_speed_kmh=5 + 3 * math.sin(i * 0.4),
                solar_radiation_mj_m2=15 + 5 * math.sin(i * 0.6),
                evapotranspiration_mm=3 + 2 * math.sin(i * 0.5),
                irrigation_need_mm=2 + 1.5 * math.sin(i * 0.5),
                source="mock_api",
                raw_data={},
                created_at=date,
                updated_at=date
            )
            
            forecast_data.append(weather_data)
        
        # Generate irrigation recommendations
        recommendations = self.get_irrigation_recommendations(forecast_data)
        
        # Calculate potential water savings
        for rec in recommendations:
            if rec["recommendation"] == "skip":
                total_water_savings += rec["irrigation_need_mm"]
        
        return WeatherForecastResult(
            forecast_data=forecast_data,
            irrigation_recommendations=recommendations,
            water_savings_potential=total_water_savings
        )
    
    def get_mock_weather_data(
        self, 
        latitude: float, 
        longitude: float, 
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate mock weather data for testing.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            date: Date for weather data
            
        Returns:
            Mock weather data
        """
        if date is None:
            date = datetime.now()
        
        # Generate realistic mock data
        day_of_year = date.timetuple().tm_yday
        
        # Seasonal temperature variation
        base_temp = 20 + 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        temperature = base_temp + 5 * math.sin(day_of_year * 0.1)
        
        # Humidity inversely related to temperature
        humidity = max(30, min(90, 80 - temperature * 1.5))
        
        # Rainfall probability
        rainfall = max(0, 10 * math.sin(day_of_year * 0.2)) if math.random() > 0.7 else 0
        
        return {
            "date": date,
            "temperature_c": temperature,
            "humidity_percent": humidity,
            "wind_speed_kmh": 5 + 3 * math.sin(day_of_year * 0.1),
            "solar_radiation_mj_m2": 15 + 5 * math.sin(day_of_year * 0.1),
            "rainfall_mm": rainfall,
            "atmospheric_pressure_hpa": 1013.25,
            "weather_condition": "Clear" if rainfall == 0 else "Rain",
            "description": "Mock weather data"
        }
    
    def calculate_water_efficiency_score(
        self, 
        weather_data: List[WeatherData],
        irrigation_schedules: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate water efficiency score based on weather and irrigation schedules.
        
        Args:
            weather_data: Historical weather data
            irrigation_schedules: Irrigation schedules
            
        Returns:
            Water efficiency metrics
        """
        total_irrigation = sum(schedule.get("duration_minutes", 0) for schedule in irrigation_schedules)
        total_rainfall = sum(weather.rainfall_mm for weather in weather_data)
        
        # Calculate efficiency metrics
        water_use_efficiency = total_rainfall / (total_irrigation + total_rainfall) if (total_irrigation + total_rainfall) > 0 else 1.0
        irrigation_efficiency = min(1.0, total_rainfall / total_irrigation) if total_irrigation > 0 else 1.0
        
        return {
            "water_use_efficiency": water_use_efficiency,
            "irrigation_efficiency": irrigation_efficiency,
            "total_irrigation_minutes": total_irrigation,
            "total_rainfall_mm": total_rainfall
        } 