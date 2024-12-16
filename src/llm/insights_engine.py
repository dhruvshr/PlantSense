"""
LLM Insights Engine
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# define base prompt
BASE_PROMPT = """
You are PlantSense, an expert in plant health and care. A user has uploaded a picture of their plant, and you have identified an issue with it. Your task is to:
1. Provide a friendly introduction.
2. Name the identified disease.
3. Explain what the disease is in simple, relatable terms.
4. Talk about the disease that has been detected with some words around it. Explain to the user what the disease is and how it affects the plant.
5. Offer specific and actionable advice to manage or treat the condition.
6. Ask a follow-up question to keep the conversation going (e.g., "Would you like to learn more about prevention?" or "Is there anything else you'd like help with?").
7. Casually mention the diease that has been detected with some words around it. Express your confidence in the prediction with some words around it.
MAX_WORDS: 275
Be empathetic, helpful, and engaging in your responses.
"""

FEEDBACK_PROMPT = """
1. You have detected the disease.
2. The user has asked some follow up questions and/or concerns. Or the user as answered your follow-up question, indicating that they want to know more.
3. Respond to their question with additional information and/or clarifying information.
4. Talk about the disease that has been detected within the context of the user's question. Affirm the user's question and answer it.
Be empathetic, helpful, and engaging in your responses.
MAX_WORDS: 275
"""

class InsightsEngine:
    """
    A class to generate conversational insights from model predictions using LLM
    """
    def __init__(self):
        """Initialize the InsightsEngine"""
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_GPT_API_KEY'))

    def _generate_prompt(self, predicted_class, confidence, user_feedback=None):
        """
        Generate a prompt for the LLM based on predictions
        
        Args:
            predicted_class (str): Predicted class label
            confidence (float): Confidence score of the prediction
            user_feedback (str, optional): user's follow up question
        Returns:
            str: Constructed prompt
        """
        # print(predicted_class)
        # print(confidence)
        prompt = ''

        if predicted_class:
            if "background" in str(predicted_class).lower():
                if confidence > 0.8:  # high confidence that there's no plant
                    prompt += f"""
                        1. Politely inform the user that no plant or leaves were detected in the image with {confidence:.2f}% confidence. But don't explicitly mention the confidence score.
                        2. Ask them to ensure their plant is clearly visible in the frame.
                        3. Provide tips for taking better plant photos (good lighting, clear view of leaves, minimal background clutter).
                        4. Encourage them to try again with a new photo.
                        5. End with a supportive message like "I'm here to help once you have a clear photo of your plant!"
                        Example starting point:
                        "I don't seem to detect any plants or leaves in this image. This could mean the plant is out of frame or the image might need to be clearer..."
                        """
                else:  # lower confidence might indicate a misclassification
                    prompt += f"""
                        1. Acknowledge potential uncertainty in the analysis with {confidence:.2f}% confidence. But don't explicitly mention the confidence score.
                        2. Mention that while the system is having trouble detecting the plant clearly, you'd be happy to try again.
                        3. Provide tips for getting a better analysis (different angle, better lighting, closer shot of leaves).
                        4. If they confirm there is a plant in the image, apologize for the misclassification.
                        5. End with an encouraging message to try again with a different photo.
                        Example starting point:
                        "I'm having a bit of trouble getting a clear read on your plant in this image. While I see something that might be a plant..."
                        """
            elif "healthy" in str(predicted_class).lower():
                prompt += f"""
                    1. Share the good news about the plant's good health in a friendly tone.
                    2. Offer general plant care advice to keep the plant thriving.
                    3. Consider your language and tone with a degree of certainty depending on the confidence score {confidence:.2f}%.
                    4. Provide fun or interesting facts about plants to engage the user.
                    5. End with an open-ended question like, "Would you like tips to make your plant even happier?"
                    Example starting point:
                    "Hi there! Your plant looks fantastic—healthy and happy! Keep up the great care. Here's how you can maintain this..."
                    """

            else:
                prompt += f"""
                    1. Explain the detected condition '{predicted_class}' in simple, clear terms with {confidence:.2f}% confidence.
                    2. Provide specific care instructions and treatment recommendations for this condition.
                    3. Mention preventive measures to avoid this issue in the future.
                    4. Reassure the user while being honest about the severity.
                    5. End with an open question like "Would you like more specific details about treating this condition?"
                    Example starting point:
                    "I've analyzed your plant and detected signs of {predicted_class}. While this is concerning, don't worry - with proper care, we can help your plant recover. Here's what you need to know..."
                    """

        if user_feedback:
             prompt = f"""
             Avoid directly using the predicted class and confidence in your response.
             1. The plant has been detected with the condition '{predicted_class}'.
             2. The confidence is {confidence:.2f}%.
             3. User has asked the follow-up: '{user_feedback}'.
             4. Provide additional information or clarification regarding the user's query or concern.
             """

        return prompt
         
        
    def generate_insights(self, predicted_class, confidence, user_feedback=None) -> str:
        """
        Generate natural language insights from model predictions
        
        Args:
            predicted_class (str): Predicted class label (e.g., disease name).
            confidence (float): Confidence score of the prediction.
            user_feedback (str, optional): User-provided follow-up question or comment
            
        Returns:
            str: Natural language insights about the predictions
        """
        # Construct the prompt
        prompt = self._generate_prompt(predicted_class, confidence, user_feedback)
        
        if user_feedback:
             try:
                  # get completion from openai
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": FEEDBACK_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
             except Exception as e:
                 return f"Error generating follow up insights: {str(e)}"

        try:
            # get completion from openai
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