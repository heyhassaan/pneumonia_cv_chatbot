# Pediatric Chest X-Ray Pneumonia Classification Using CNN

This project implements a Convolutional Neural Network (CNN) to classify pediatric chest X-ray images as either **Pneumonia** or **Normal**. The dataset contains 5,863 anterior-posterior grayscale X-ray images. To enhance model performance and address class imbalance, images were resized, normalized, and augmented during preprocessing.


## Project Files

- **Medical_Image_Detection_Using_CNNs.ipynb**  
  Jupyter notebook with the complete model implementation and training pipeline.

- **Medical_Image_Detection_Using_CNNs.html**  
  HTML export of the notebook, including detailed Markdown explanations and visualizations.

- **Cnn_model91**  
  Exported trained CNN model file, used within the Streamlit application.

- **streamlit.py**  
  Streamlit application code for interactive use of the model.  
  *Note:* To run locally, place your LLM token file (used for chatbot responses) in a secure `.env` file for the app to function properly on localhost.


## Installation

1. Clone the repository.

2. Create and activate a virtual environment.

3. Install required dependencies.

4. Add your LLM token to a `.env` file in the root directory:

