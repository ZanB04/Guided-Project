import dash
from dash import dcc, html
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model("artifacts/student_performance.h5")
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("BrightPath Student Grade Predictor"),
    dcc.Input(id = "study-time", type="number", placeholder="StudyTimeWeekly"),
    dcc.Input(id = "absences", type="number", placeholder="Absences"),
    dcc.Input(id = "tutoring", type= "number", placeholder="Tutoring (0/1)"),
    html.Button("Predict", id ="submit"),
    html.Div(id = "output")
])
@app.callback(
    dash.Output("output", "children"),
    dash.Input("submit", "n_clicks"),
    dash.State("study-time", "value"),
    dash.State("absences", "value"),
    dash.State("tutoring", "value")
)
def predict_grade(n, study, absn, tutor):
    if n:
        input_data = np.array([[study, absn, tutor]])
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
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
    port = os.getenv('PORT', 8050)
    app.run_server(debug = True, host = '0.0.0.0', port = port)
    server = app.server
