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
2. Name the identified disease without technical jargon.
3. Explain the disease in simple, relatable terms.
4. Offer specific and actionable advice to manage or treat the condition.
5. Ask a follow-up question to keep the conversation going (e.g., "Would you like to learn more about prevention?" or "Is there anything else you'd like help with?").
6. Avoid mentioning technical details like "confidence scores" or "model predictions."

Be empathetic, helpful, and engaging in your responses.
"""

FEEDBACK_PROMPT = """
1. You have detected the disease.
2. The user has asked some follow up questions and/or concerns. Or the user as answered your follow-up question, indicating that they want to know more.
3. Respond to their question with additional information and/or clarifying information.
4. Avoid mentioning technical details like "confidence scores" or "model predictions."
Be empathetic, helpful, and engaging in your responses.
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
        prompt = BASE_PROMPT

        if predicted_class:
            if "healthy" in predicted_class:
                    prompt += f"""
                    1. Share the good news about the plant's health in a friendly tone.
                    2. Offer general plant care advice to keep the plant thriving.
                    3. Consider your language and tone with a degree of certainty depending on the confidence score {confidence:.2f}%.
                    4. Provide fun or interesting facts about plants to engage the user.
                    5. End with an open-ended question like, "Would you like tips to make your plant even happier?"
                
                    Example starting point:
                    "Hi there! Your plant looks fantastic—healthy and happy! Keep up the great care. Here's how you can maintain this..."
                    """

            else:
                prompt += f"""
                You are PlantSense, an intelligent and friendly plant health assistant. A user has uploaded a picture of their plant, and you've identified signs of {predicted_class}. Your task is to:
                
                1. Greet the user in a friendly way and confirm the plant's condition conversationally.
                2. Explain {predicted_class} in simple terms, including what it is and how it affects the plant.
                3. Consider your language and tone with a degree of certainty depending on the confidence score {confidence:.2f}%.
                4. Offer practical, step-by-step advice to manage or treat the condition.
                5. Suggest preventive measures to avoid similar issues in the future.
                6. End with an open-ended question like, "Is there anything else you'd like help with?"
                
                Example starting point:
                "Hi there! Based on your plant's condition, it seems to have {str(predicted_class).replace('_', ' ')}. Don't worry—I'm here to help! Here's what you need to know..."
                """

        if user_feedback:
             prompt = f"""
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
    
    # def _construct_prompt(self, predictions: List[Dict], image_paths: List[str]) -> str:
    #     """
    #     Construct a prompt for the LLM based on predictions
        
    #     Args:
    #         predictions (List[Dict]): List of prediction dictionaries
    #         image_paths (List[str]): List of image paths
            
    #     Returns:
    #         str: Constructed prompt
    #     """
    #     prompt = generate_prompt(predictions, confidence)
        
    #     for i, (pred, img_path) in enumerate(zip(predictions, image_paths)):
    #         prompt += f"Image {i+1} ({os.path.basename(img_path)}):\n"
    #         prompt += f"- Predicted class: {pred['class']}\n"
    #         prompt += f"- Confidence: {pred['confidence']:.2f}%\n\n"
            
    #     prompt += """
    #     Please provide insights about these findings in a conversational way. Include:
    #     1. A summary of the plant health status
    #     2. Potential concerns or issues identified
    #     3. Recommended actions or treatments if problems are detected
    #     4. General advice for plant care

    #     Please keep the response informative but friendly and easy to understand.
    #     """
        
    #     return prompt
