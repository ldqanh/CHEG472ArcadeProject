# CHEG472ArcadeProject
Project Overview
This project involves collecting, analyzing, and modeling data from mini-bowling and arcade games to derive actionable insights. It includes exploratory data analysis (EDA), predictive modeling, and conversational AI development using LangChain.

Project Structure
Data Collection

Mini-Bowling:
Number of pins knocked down per round.
Estimated throw angle.
Bowling speed categorized as "slow," "medium," or "fast."
Total score per player.
Arcade Games:
Game name.
Final score per player.
Number of attempts/lives used.
Observed player strategies.
Data Preprocessing

Cleaning and organizing data to address missing or inconsistent values.
Separating features (inputs) and labels (outputs) for modeling.
Exploratory Data Analysis (EDA)

Visualizations using Python libraries such as Pandas and Matplotlib.
Identifying trends (e.g., how throw angle affects pins knocked down) and anomalies.
Machine Learning Model

Splitting data into training and testing sets.
Developing a predictive model using Scikit-learn.
Evaluating model performance with metrics like accuracy or R² score.
LangChain Chatbot

Designing a chatbot that suggests strategies based on collected data.
Utilizing pre-trained large language models via LangChain.
Deploying the chatbot (e.g., Streamlit, Gradio).
Technologies Used
Python Libraries: Pandas, Matplotlib, Scikit-learn
Machine Learning Frameworks: Scikit-learn
Conversational AI: LangChain
Hosting Tools (optional): Streamlit, Gradio
How to Run the Project
Data Collection:

Use the template provided in the instructions to collect game data.
Save the structured data in an Excel or CSV file.
Preprocessing and EDA:

Run the provided Jupyter Notebook/Google Colab file.
Ensure you install all dependencies: pandas, matplotlib, and scikit-learn.
Machine Learning Model:

Use the dataset generated from preprocessing to train the predictive model.
Run the Python script or notebook provided to evaluate model performance.
LangChain Chatbot:

Use the LangChain script to deploy the chatbot.
Test inputs (e.g., game strategies) to receive suggestions.
Deliverables
Cleaned dataset in Excel or CSV format.
Jupyter Notebook/Google Colab file with:
Preprocessing steps.
EDA visualizations.
Machine learning implementation and results.
LangChain script or hosted chatbot link.
