"""
LLM Insights Engine
"""

import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# define base prompt
BASE_PROMPT = """
You are a helpful plant disease expert providing insights about plant health analysis.
"""

class InsightsEngine:
    """
    A class to generate conversational insights from model predictions using LLM
    """
    def __init__(self):
        """Initialize the InsightsEngine"""
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_GPT_API_KEY'))
        
    def generate_insights(self, predictions: List[Dict], image_paths: List[str]) -> str:
        """
        Generate natural language insights from model predictions
        
        Args:
            predictions (List[Dict]): List of prediction dictionaries containing class labels and confidence scores
            image_paths (List[str]): List of paths to the analyzed images
            
        Returns:
            str: Natural language insights about the predictions
        """
        # Construct the prompt
        prompt = self._construct_prompt(predictions, image_paths)
        
        try:
            # Get completion from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": BASE_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def _construct_prompt(self, predictions: List[Dict], image_paths: List[str]) -> str:
        """
        Construct a prompt for the LLM based on predictions
        
        Args:
            predictions (List[Dict]): List of prediction dictionaries
            image_paths (List[str]): List of image paths
            
        Returns:
            str: Constructed prompt
        """
        prompt = "Based on the analysis of plant images, here are the findings:\n\n"
        
        for i, (pred, img_path) in enumerate(zip(predictions, image_paths)):
            prompt += f"Image {i+1} ({os.path.basename(img_path)}):\n"
            prompt += f"- Predicted class: {pred['class']}\n"
            prompt += f"- Confidence: {pred['confidence']:.2f}%\n\n"
            
        prompt += """
        Please provide insights about these findings in a conversational way. Include:
        1. A summary of the plant health status
        2. Potential concerns or issues identified
        3. Recommended actions or treatments if problems are detected
        4. General advice for plant care

        Please keep the response informative but friendly and easy to understand.
        """
        
        return prompt
