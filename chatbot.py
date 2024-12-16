import re
import joblib
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import gradio as gr


# Load the pre-trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Initialize the OpenAI LLM via LangChain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="")

# Define the Prompt Template for LangChain
prompt = PromptTemplate(
    input_variables=["parsed_input", "user_input", "model_prediction"],
    template=(
        "You are a bowling strategy assistant. The user provided the following input: {user_input}.\n\n"
        "Here is the extracted information from their input: {parsed_input}.\n\n"
        "The model predicted the following insights: {model_prediction}.\n\n"
        "Based on this, provide strategies to improve their bowling performance."
    )
)

# Create a LangChain LLM Chain
strategy_chain = LLMChain(llm=llm, prompt=prompt)

# Function to Parse User Input
def parse_user_input(user_input):
    """
    Extract bowling details from the user's input using regex.
    """
    # Extract details using regex patterns
    pins = re.search(r"(\d+)\s*pins", user_input, re.IGNORECASE)
    angle = re.search(r"angle\s*of\s*(\d+)°", user_input, re.IGNORECASE)
    speed = re.search(r"speed\s*of\s*(\d+)\s*m/s", user_input, re.IGNORECASE)

    # Parse results or default to "not specified"
    parsed_details = {
        "pins": int(pins.group(1)) if pins else None,
        "angle": int(angle.group(1)) if angle else None,
        "speed": int(speed.group(1)) if speed else None,
    }
    return parsed_details

# Function to Generate Model Prediction
def generate_model_prediction(parsed_details):
    """
    Use the loaded model and scaler to generate predictions based on user inputs.
    """
    try:
        # Prepare input for the model
        input_data = np.array([[parsed_details['pins'], parsed_details['angle'], parsed_details['speed']]])
        input_data = scaler.transform(input_data)  # Scale the input data

        # Predict using the model
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        return f"Error generating prediction: {str(e)}"

# Function to Generate Chatbot Response
def generate_response(user_input):
    """
    Generate a response based on parsed details, model prediction, and user input.
    """
    try:
        # Parse user input for details
        parsed_details = parse_user_input(user_input)
        
        # Ensure all necessary details are present
        if None in parsed_details.values():
            return "Error: Please provide details about pins, angle, and speed in your input."

        parsed_input = (
            f"Pins knocked down: {parsed_details['pins']}, "
            f"Angle: {parsed_details['angle']}°, "
            f"Speed: {parsed_details['speed']} m/s"
        )

        # Generate model prediction
        model_prediction = generate_model_prediction(parsed_details)

        # Generate strategy using LangChain
        langchain_response = strategy_chain.run(
            parsed_input=parsed_input,
            user_input=user_input,
            model_prediction=model_prediction
        )

        # Combine Parsed Details, Model Prediction, and LangChain Output
        return (
            f"Parsed Details:\n{parsed_input}\n\n"
            f"Model Prediction:\n{model_prediction}\n\n"
            f"Strategy:\n{langchain_response}"
        )
    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"

# Gradio Chatbot Interface
def chatbot(user_input):
    """
    Interface function for Gradio.
    """
    return generate_response(user_input)

# Set up Gradio Interface
interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="User Bowling Description"),  # Free text input
    outputs=gr.Textbox(label="Recommendations:"),  # Text output
    title="Bowling Strategy Chatbot with Model Insights",
    description=(
        "Ask the chatbot for bowling strategies! For example, type:\n"
        "'I knocked down 5 pins with an angle of 30° and a speed of 15 m/s.'\n"
        "The chatbot will analyze, provide predictions, and suggest improvements."
    )
)

# Run the Gradio App
if __name__ == "__main__":
    interface.launch(share=True)
