import dash
from dash import dcc, html
import tensorflow as tf
import numpy as np
import pandas as pd
import os

interpreter = tf.lite.Interpreter("artifacts/student_performance.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
train_mean = pd.read_csv("artifacts/train_mean.csv", index_col=0).squeeze("columns")
train_std = pd.read_csv("artifacts/train_std.csv", index_col=0).squeeze("columns")
app = dash.Dash(__name__)
server = app.server
feature_names = [
    "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering", "GPA"
]
app.layout = html.Div([
    html.H1("BrightPath Student Grade Predictor"),
    dcc.Input(id='age', type='number', placeholder='Age', style={'margin': '5px'}),

    dcc.Dropdown(
        id='gender',
        options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}],
        placeholder='Select Gender', style={'margin': '5px'}
    ),

    dcc.Dropdown(
        id='ethnicity',
        options=[
            {'label': 'Group A', 'value': 0},
            {'label': 'Group B', 'value': 1},
            {'label': 'Group C', 'value': 2},
            {'label': 'Group D', 'value': 3},
            {'label': 'Group E', 'value': 4}
        ],
        placeholder='Select Ethnicity', style={'margin': '5px'}
    ),

    dcc.Dropdown(
        id='parentaleducation',
        options=[
            {'label': 'High School', 'value': 0},
            {'label': "Bachelor's", 'value': 1},
            {'label': "Master's", 'value': 2}
        ],
        placeholder='Parental Education', style={'margin': '5px'}
    ),

    dcc.Input(id='studytimeweekly', type='number', placeholder='Study Time Weekly', style={'margin': '5px'}),
    dcc.Input(id='absences', type='number', placeholder='Absences', style={'margin': '5px'}),

    dcc.Dropdown(
        id='tutoring',
        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
        placeholder='Tutoring', style={'margin': '5px'}
    ),

    dcc.Dropdown(
        id='parentalsupport',
        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
        placeholder='Parental Support', style={'margin': '5px'}
    ),

    dcc.Dropdown(
        id='extracurricular',
        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
        placeholder='Extracurricular', style={'margin': '5px'}
    ),

    dcc.Dropdown(
        id='sports',
        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
        placeholder='Sports', style={'margin': '5px'}
    ),

    dcc.Dropdown(
        id='music',
        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
        placeholder='Music', style={'margin': '5px'}
    ),

    dcc.Dropdown(
        id='volunteering',
        options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
        placeholder='Volunteering', style={'margin': '5px'}
    ),

    dcc.Input(id='gpa', type='number', placeholder='GPA', style={'margin': '5px'}),
    html.Button("Predict", id ="submit"),
    html.Div(id="output", style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "18px"})
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
        feature_names = [
            "Age", "Gender", "Ethnicity", "ParentalEducation",
            "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
            "Extracurricular", "Sports", "Music", "Volunteering", "GPA"
        ]
        input_df = pd.DataFrame([inputs], columns=feature_names)
        input_scaled = (input_df - train_mean)/train_std
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
    
