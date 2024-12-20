import os
import pandas as pd
import json
import time
import random
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import logging
import httpx
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ImprovedRateLimiter:
    """Enhanced rate limiter with better backoff strategy"""
    def __init__(self, base_delay=1, max_delay=120, max_retries=5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.current_retry = 0
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum time between requests in seconds

    def wait(self, status_code: Optional[int] = None) -> bool:
        """
        Implements improved exponential backoff with status code awareness
        Returns: bool indicating whether to retry
        """
        if self.current_retry >= self.max_retries:
            return False

        # Calculate time since last request
        time_since_last = time.time() - self.last_request_time
        
        # If we're retrying due to a rate limit
        if status_code == 429:
            delay = min(
                self.base_delay * (2 ** self.current_retry) + random.uniform(0, 1),
                self.max_delay
            )
            # Add extra delay for rate limit errors
            delay *= 1.5
        else:
            # Ensure minimum interval between requests
            delay = max(0, self.min_request_interval - time_since_last)

        if delay > 0:
            logger.info(f"Rate limiting: Waiting {delay:.2f} seconds before {'retry' if status_code else 'next request'}...")
            time.sleep(delay)

        self.last_request_time = time.time()
        self.current_retry += 1
        return True

    def reset(self):
        """Resets the retry counter but maintains last request time"""
        self.current_retry = 0

class TowerTrafficPredictor:
    def __init__(self, api_key=None):
        try:
            logger.debug("Initializing TowerTrafficPredictor...")
            
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            
            if not self.api_key:
                raise ValueError("No API key provided. Please set GROQ_API_KEY environment variable or pass api_key to constructor.")
            
            logger.debug("API key found, initializing LLM...")
            
            self.llm = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0,
                groq_api_key=self.api_key,
                max_retries=3,  # Add max retries to ChatGroq
            )
            self.historical_patterns = []
            self.rate_limiter = ImprovedRateLimiter()
            logger.info("LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    def _call_llm_with_retry(self, chain, input_data):
        """Enhanced helper method to call LLM with improved retry logic"""
        while True:
            try:
                # Wait before making request (implements rate limiting)
                if not self.rate_limiter.wait():
                    raise Exception("Max retries exceeded")
                
                res = chain.invoke(input=input_data)
                self.rate_limiter.reset()
                return res
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if not self.rate_limiter.wait(status_code=429):
                        raise Exception("Rate limit retries exceeded")
                    continue
                raise
            except Exception as e:
                if "429" in str(e):
                    if not self.rate_limiter.wait(status_code=429):
                        raise Exception("Rate limit retries exceeded")
                    continue
                raise

    def load_and_pretrain_historical_data(self, csv_file_path, chunk_size=25):
        """Load and process historical data in chunks with improved rate limiting"""
        try:
            logger.debug(f"Loading historical data from {csv_file_path}")
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
            
            df = pd.read_csv(csv_file_path)
            logger.debug(f"Loaded CSV with shape: {df.shape}")
            
            chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            logger.info(f"Split data into {len(chunks)} chunks")

            for i, chunk in enumerate(chunks, 1):
                logger.debug(f"Processing chunk {i}/{len(chunks)}")
                chunk_data = chunk.to_string(index=False)

                prompt_train = PromptTemplate.from_template(
                    """
                    ### HISTORICAL TOWER DATA CHUNK
                    {historical_data}
                    ### INSTRUCTION
                    Analyze this chunk of historical tower data and extract key patterns:
                    1. Typical traffic ranges for this time period
                    2. Any notable anomalies
                    3. Peak usage patterns
                    
                    Provide a concise summary of patterns found in this exact JSON format:
                    {{
                        "typical_traffic_range": {{"min": <number>, "max": <number>}},
                        "peak_hours": ["<hour1>", "<hour2>"],
                        "anomalies_detected": <boolean>
                    }}
                    """
                )

                try:
                    chain_train = prompt_train | self.llm
                    res = self._call_llm_with_retry(chain_train, {"historical_data": chunk_data})
                    
                    try:
                        pattern = json.loads(res.content.strip())
                        self.historical_patterns.append(pattern)
                        logger.debug(f"Successfully processed chunk {i}")
                    except json.JSONDecodeError as je:
                        logger.warning(f"Invalid JSON response for chunk {i}: {str(je)}")
                        continue
                    
                except Exception as e:
                    logger.warning(f"Error processing chunk {i}: {str(e)}")
                    continue

            logger.info(f"Successfully processed {len(self.historical_patterns)} chunks of historical data")
            
        except Exception as e:
            logger.error(f"Error during pretraining: {str(e)}")
            raise

    # ... rest of the TowerTrafficPredictor class remains the same ...

    def predict_realtime_traffic(self, realtime_data_json):
        """Predict traffic patterns based on real-time data"""
        try:
            logger.debug("Starting real-time prediction...")
            if not self.historical_patterns:
                raise Exception("Historical patterns not loaded. Pretrain the model first.")

            realtime_data_str = pd.DataFrame([realtime_data_json]).to_string(index=False)
            patterns_summary = json.dumps(self.historical_patterns[-3:])  # Use last 3 patterns

            prompt_analyze = PromptTemplate.from_template(
                """
                ### REAL-TIME TOWER DATA
                {realtime_data}
                ### HISTORICAL PATTERNS SUMMARY
                {patterns_summary}
                ### INSTRUCTION
                Compare real-time data with the historical patterns. Determine if:
                - Any two out of three towers (tower1, tower2, tower3) have low network traffic.
                - Low traffic is defined as below 1000 active connections.
                Return the result in this exact JSON format:
                {{
                    "timestamp": "{timestamp}",
                    "tower1": <boolean>,
                    "tower2": <boolean>,
                    "tower3": <boolean>
                }}
                """
            )

            logger.debug("Invoking LLM for prediction...")
            chain_analyze = prompt_analyze | self.llm
            res = self._call_llm_with_retry(
                chain_analyze,
                {
                    "realtime_data": realtime_data_str,
                    "patterns_summary": patterns_summary,
                    "timestamp": realtime_data_json["timestamp"]
                }
            )
            
            try:
                result = json.loads(res.content.strip())
                logger.info(f"Prediction complete: {result}")
                return result
            except json.JSONDecodeError as je:
                raise Exception(f"Invalid JSON response from LLM: {str(je)}")
            
        except Exception as e:
            logger.error(f"Error during real-time prediction: {str(e)}")
            raise Exception(f"Error during real-time prediction: {str(e)}")


if __name__ == "__main__":
    try:
        logger.info("Starting main execution...")
        
        # Initialize with API key
        api_key = "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"  # Replace with your actual API key
        predictor = TowerTrafficPredictor(api_key=api_key)
        
        # Load and process historical data
        historical_csv_file = "./historical_tower_data.csv"
        logger.info(f"Loading historical data from: {historical_csv_file}")
        predictor.load_and_pretrain_historical_data(historical_csv_file, chunk_size=5)

        # Example real-time data for prediction
        realtime_data_json = {
            "towerId": "tower1",
            "latitude": 28.7041,
            "longitude": 77.1025,
            "region": "Region1",
            "activeConnections": 1200,
            "trafficVolumeMB": 500,
            "callTraffic": 150,
            "peakUsage": 1800,
            "blockedRequests": 5,
            "droppedCalls": 2,
            "timestamp": "2024-12-19 10:30:00"
        }

        # Make prediction
        logger.info("Making prediction with real-time data...")
        prediction = predictor.predict_realtime_traffic(realtime_data_json)
        logger.info(f"Final prediction result: {prediction}")
        print("Prediction Result:", prediction)
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"Error: {e}")