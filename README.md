# Diagno-AI: Disease Prediction Application

Diagno-AI is a Streamlit-based web application designed to predict the likelihood of various diseases using machine learning models. This application provides an easy-to-use interface for users to input relevant medical data and receive predictions for diseases such as Parkinson's Disease, Lung Cancer, Heart Disease, and Diabetes.
- link for the app [Diagno-AI](https://diagno-ai.streamlit.app/)
## Work Demo
![Screenshot 2025-03-25 215039](https://github.com/user-attachments/assets/47e935f0-ad30-4007-8952-3c4ccda1237f)
![Screenshot 2025-03-25 215057](https://github.com/user-attachments/assets/f52ef4e1-04a5-4244-b53f-a4a41e80b326)

## Features

- **Parkinson's Disease Prediction:** Predicts the likelihood of Parkinson's Disease based on vocal frequency and amplitude features.
- **Lung Cancer Prediction:** Predicts the likelihood of Lung Cancer using personal, primary, and secondary symptom data.
- **Heart Disease Prediction:** Predicts the likelihood of Heart Disease using patient information, cholesterol levels, and exercise-related factors.
- **Diabetes Prediction:** Predicts the likelihood of Diabetes using patient demographics, blood test results, and lifestyle factors.
- **Customizable Background:** The app includes a visually appealing background with a semi-transparent overlay for better readability.
- **Responsive Design:** The app is designed to work seamlessly across devices.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Parneet-Sandhu/Diagno-AI.git
    cd Diagno-AI
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the following directory structure exists:

    ```
    Diagno-AI/
    ├── app.py
    ├── Models/
    │    ├── parkinsons_model.sav
    │    ├── lungs_disease_model.sav
    │    ├── heart_disease_model.sav
    │    └── diabetes_prediction_model.sav
    ├── assets/
    └── requirements.txt
    ```

4. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the application in your browser.
2. Select a disease from the dropdown menu.
3. Fill in the required input fields based on the selected disease.
4. Click the **Predict** button to get the prediction result.
5. View the prediction result, which will indicate whether the patient is likely to have the disease or not.

## Models Used

The application uses pre-trained machine learning models stored in the `Models` directory. These models are trained on relevant datasets for each disease:

- **Parkinson's Disease:** `parkinsons_model.sav`
- **Lung Cancer:** `lungs_disease_model.sav`
- **Heart Disease:** `heart_disease_model.sav`
- **Diabetes:** `diabetes_prediction_model.sav`

## File Structure

- `app.py`: Main application file containing the Streamlit code.
- `Models/`: Directory containing pre-trained machine learning models.
- `assets/`: Directory containing static assets like the background image.
- `requirements.txt`: File listing all the Python dependencies required for the project.

## Dependencies

The application requires the following Python libraries:

- `streamlit`
- `pandas`
- `pickle`
- `base64`
- `os`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Disclaimer

This application is a prototype and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The machine learning models used in this application were trained on publicly available datasets.
- Special thanks to the Streamlit community for providing an excellent framework for building interactive web applications.
