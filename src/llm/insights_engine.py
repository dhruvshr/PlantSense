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
MAX_WORDS: 100
Be empathetic, helpful, and engaging in your responses.
"""

FEEDBACK_PROMPT = """
1. You have detected the disease.
2. The user has asked some follow up questions and/or concerns. Or the user as answered your follow-up question, indicating that they want to know more.
3. Respond to their question with additional information and/or clarifying information.
4. Talk about the disease that has been detected within the context of the user's question. Affirm the user's question and answer it.
Be empathetic, helpful, and engaging in your responses.
MAX_WORDS: 100
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
            if "healthy" in str(predicted_class).lower():
                    prompt += f"""
                    1. Share the good news about the plant's good health in a friendly tone.
                    2. Offer general plant care advice to keep the plant thriving.
                    3. Consider your language and tone with a degree of certainty depending on the confidence score {confidence:.2f}%.
                    4. Provide fun or interesting facts about plants to engage the user.
                    5. End with an open-ended question like, "Would you like tips to make your plant even happier?"
                    MAX_WORDS: 100
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
                    MAX_WORDS: 100
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
             MAX_WORDS: 100
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
