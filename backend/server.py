from flask import Flask, jsonify
from flask_cors import CORS
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

def generate_weather_data():
    return {
        "solar_radiation": round(random.uniform(100, 1000), 1),
        "sunlight_hours": round(random.uniform(4, 12), 1),
        "wind_speed": round(random.uniform(0, 25), 1),
        "rainfall": round(random.uniform(0, 100), 1),
        "timestamp": datetime.now().isoformat()
    }

def calculate_viability(weather):
    solar_viability = min((weather["solar_radiation"] / 1000) * (weather["sunlight_hours"] / 12), 1)
    wind_viability = min(weather["wind_speed"] / 15, 1)
    hydro_viability = min(weather["rainfall"] / 50, 1)
    
    return {
        "solar_viability": round(solar_viability, 2),
        "wind_viability": round(wind_viability, 2),
        "hydro_viability": round(hydro_viability, 2)
    }

def get_recommendation(weather_analysis):
    sources = ["solar", "wind", "hydro"]
    viabilities = [
        weather_analysis["solar_viability"],
        weather_analysis["wind_viability"],
        weather_analysis["hydro_viability"]
    ]
    
    recommended = sources[viabilities.index(max(viabilities))]
    alternatives = [s for s, v in zip(sources, viabilities) 
                   if v > 0.3 and s != recommended]
    
    return {
        "recommended_energy_source": recommended,
        "confidence_score": max(viabilities),
        "alternative_source": alternatives[0] if alternatives else "none",
        "reasoning": generate_reasoning(recommended, max(viabilities)),
        "weather_analysis": weather_analysis
    }

def generate_reasoning(source, confidence):
    reasons = {
        "solar": "Based on current solar radiation levels and daylight hours, solar power generation conditions are optimal.",
        "wind": "Current wind speeds are favorable for wind energy harvesting, making it the most efficient choice.",
        "hydro": "Recent rainfall patterns and water flow rates indicate hydro power would be most effective."
    }
    return reasons.get(source, "")

@app.route('/run-script', methods=['GET'])
def get_energy_recommendation():
    try:
        weather_data = generate_weather_data()
        weather_analysis = calculate_viability(weather_data)
        recommendation = get_recommendation(weather_analysis)
        
        return jsonify({
            "weather": weather_data,
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)