import os
import json
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeatherDataProcessor:
    """Handle weather and CSV data processing"""
    
    def __init__(self):
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
    
    def get_current_weather(self, city: str = "London") -> Optional[Dict]:
        """Get current weather data from OpenWeatherMap API or return sample data if API fails"""
        
        def _get_sample_weather_data():
            """Provide sample weather data for testing"""
            print("Using sample weather data for testing purposes")
            return {
                "solar_radiation": 750,  # Good conditions for solar
                "sunlight_hours": 8.5,   # Typical sunny day
                "wind_speed": 15.0,      # Good conditions for wind power
                "rainfall": 25.0         # Moderate rainfall
            }
        
        try:
            if not self.weather_api_key:
                print("Warning: WEATHER_API_KEY not found, using sample weather data")
                return _get_sample_weather_data()
                
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract and process weather data
            cloud_coverage = data.get("clouds", {}).get("all", 50)
            wind_speed = data.get("wind", {}).get("speed", 0)
            rain_data = data.get("rain", {})
            rainfall = rain_data.get("1h", 0) if rain_data else 0
            
            estimated_solar_radiation = 1000 - (cloud_coverage * 9)
            base_daylight_hours = 12
            estimated_sunlight = base_daylight_hours * (1 - (cloud_coverage / 100))
            
            return {
                "solar_radiation": estimated_solar_radiation,
                "sunlight_hours": estimated_sunlight,
                "wind_speed": wind_speed,
                "rainfall": rainfall
            }
            
        except Exception as e:
            print(f"Warning: API error ({str(e)}), using sample weather data")
            return _get_sample_weather_data()
        except Exception as e:
            print(f"Error getting weather data: {e}")
            return None

    @staticmethod
    def read_historical_data(file_path: str) -> pd.DataFrame:
        """Read and validate the CSV file"""
        try:
            df = pd.read_csv(file_path)
            required_columns = [
                'Day', 'Solar_Radiation', 'Direct_Sunlight_Hours',
                'Wind_Speed', 'Rainfall_mm', 'Energy_Type'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return df
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

class EnergyRecommender:
    """Energy recommendation system using ChatGroq"""
    
    def __init__(self):
        # Initialize ChatGroq
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
        
        # Initialize JSON output parser
        self.parser = JsonOutputParser()
        
        # Create prompt template for current weather
        self.prompt = PromptTemplate(
            template="""Analyze the current weather metrics and historical patterns to recommend the most appropriate renewable energy source:

Current Weather Metrics:
- Solar Radiation: {solar_radiation} W/m²
- Sunlight Hours: {sunlight_hours} hours
- Wind Speed: {wind_speed} m/s
- Rainfall: {rainfall} mm

Historical Pattern Summary:
{historical_pattern}

Based on both current conditions and historical patterns, provide your recommendation in the following JSON format:
{{
    "recommended_energy_source": "solar|wind|hydro",
    "confidence_score": <float between 0 and 1>,
    "reasoning": "detailed explanation of why this source is recommended",
    "alternative_source": "solar|wind|hydro|none",
    "weather_analysis": {{
        "solar_viability": <float between 0 and 1>,
        "wind_viability": <float between 0 and 1>,
        "hydro_viability": <float between 0 and 1>
    }}
}}""",
            input_variables=["solar_radiation", "wind_speed", "rainfall", 
                           "sunlight_hours", "historical_pattern"]
        )

    def analyze_historical_patterns(self, df: pd.DataFrame, 
                                  current_weather: Dict) -> str:
        """Analyze historical data for similar weather patterns"""
        conditions = (
            (abs(df['Solar_Radiation'] - current_weather['solar_radiation']) < 50) &
            (abs(df['Direct_Sunlight_Hours'] - current_weather['sunlight_hours']) < 2) &
            (abs(df['Wind_Speed'] - current_weather['wind_speed']) < 2) &
            (abs(df['Rainfall_mm'] - current_weather['rainfall']) < 10)
        )
        
        similar_conditions = df[conditions]
        pattern_summary = {}
        
        if len(similar_conditions) > 0:
            pattern_summary = {
                'most_common_energy': similar_conditions['Energy_Type'].mode().iloc[0],
                'success_rate': len(similar_conditions[similar_conditions['Energy_Type'] == 
                                  similar_conditions['Energy_Type'].mode().iloc[0]]) / len(similar_conditions)
            }
            
        return f"""In similar weather conditions historically:
- Most commonly used energy source: {pattern_summary.get('most_common_energy', 'No direct match')}
- Historical success rate: {pattern_summary.get('success_rate', 0):.2%}"""

    def get_recommendation(self, current_weather: Dict, 
                         historical_data: pd.DataFrame) -> Dict:
        """Generate energy recommendation based on current weather and historical data"""
        try:
            # Analyze historical patterns
            historical_pattern = self.analyze_historical_patterns(historical_data, 
                                                               current_weather)
            
            # Format prompt with all data
            formatted_prompt = self.prompt.format(
                **current_weather,
                historical_pattern=historical_pattern
            )
            
            # Get response from ChatGroq
            response = self.llm.predict(formatted_prompt)
            
            # Parse JSON response
            recommendation = self.parser.parse(response)
            
            return recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return self._get_error_response(f"Error: {str(e)}")

    def _get_error_response(self, error_message: str) -> Dict:
        """Generate error response in the correct format"""
        return {
            "recommended_energy_source": "unknown",
            "confidence_score": 0.0,
            "reasoning": error_message,
            "alternative_source": "none",
            "weather_analysis": {
                "solar_viability": 0.0,
                "wind_viability": 0.0,
                "hydro_viability": 0.0
            }
        }

def main():
    try:
        # Initialize components
        weather_processor = WeatherDataProcessor()
        recommender = EnergyRecommender()
        
        # Get current weather
        current_weather = weather_processor.get_current_weather()
        if not current_weather:
            print("Error: Could not fetch current weather data")
            return
            
        # Load historical data
        historical_data = weather_processor.read_historical_data("energy_type_dataset_updated.csv")
        if historical_data is None:
            print("Error: Could not load historical data")
            return
            
        # Get recommendation
        recommendation = recommender.get_recommendation(current_weather, historical_data)
        
        # Print results
        print("\nCurrent Weather Conditions:")
        print(f"Solar Radiation: {current_weather['solar_radiation']:.1f} W/m²")
        print(f"Sunlight Hours: {current_weather['sunlight_hours']:.1f} hours")
        print(f"Wind Speed: {current_weather['wind_speed']:.1f} m/s")
        print(f"Rainfall: {current_weather['rainfall']:.1f} mm")
        
        print(f"\nRecommended Energy Source: {recommendation['recommended_energy_source'].upper()}")
        print(f"Confidence Score: {recommendation['confidence_score']:.2%}")
        print(f"Reasoning: {recommendation['reasoning']}")
        
        if recommendation['alternative_source'] != 'none':
            print(f"Alternative Source: {recommendation['alternative_source'].upper()}")
            
        print("\nViability Scores:")
        for source, score in recommendation['weather_analysis'].items():
            print(f"{source.replace('_viability', '').title()}: {score:.2%}")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()