import dash
from dash import dcc, html
import tensorflow as tf
import numpy as np
import pandas as pd
import os

interpreter = tf.lite.Interpreter("../artifacts/student_performance.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
train_mean = pd.read_csv("../artifacts/train_mean.csv", index_col=0).squeeze("columns")
train_std = pd.read_csv("../artifacts/train_std.csv", index_col=0).squeeze("columns")
app = dash.Dash(__name__)
server = app.server
feature_names = [
    "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering", "GPA"
]
app.layout = html.Div([
    html.H1("BrightPath Student Grade Predictor"),
    *[dcc.Input(
        id=feature.lower(), 
        type="number", 
        placeholder=feature, 
        style={'margin': '5px'}
        ) 
        for feature in feature_names
        ],
    html.Button("Predict", id ="submit"),
    html.Div(id = "output")
])
@app.callback(
    dash.Output("output", "children"),
    dash.Input("submit", "n_clicks"),
    [dash.State(feature.lower(), "value") for feature in feature_names]
)
def predict_grade(n_clicks, *inputs):
    if n_clicks:
        if None in inputs:
            return "Please fill in all input fields."
        input_data = pd.DataFrame([inputs], columns=feature_names)
        input_scaled = (input_data - train_mean)/train_std
        input_array = np.array(input_scaled, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        grade_mapping = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'F'
        }
        return f"Predicted grade: {grade_mapping[predicted_class]}."
    return ""
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug = True, host = '0.0.0.0', port = port)
    
