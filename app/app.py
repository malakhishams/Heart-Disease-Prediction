from nicegui import ui
import pandas as pd
from datetime import datetime
import pickle
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# =========================
# Custom classes required by the pickle model
# =========================

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)

        self.q1_ = X_df.quantile(0.25)
        self.q3_ = X_df.quantile(0.75)
        self.iqr_ = self.q3_ - self.q1_

        self.lower_ = self.q1_ - 1.5 * self.iqr_
        self.upper_ = self.q3_ + 1.5 * self.iqr_

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return X_df


class LabelEncoderr(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(np.asarray(X).ravel())
        return self

    def transform(self, X):
        encoded = self.encoder.transform(np.asarray(X).ravel())
        return encoded.reshape(-1, 1)


# =========================
# Load model
# =========================

with open("model.pkl", "rb") as f:
    pipe = pickle.load(f)


# =========================
# DataFrame for saved patients
# =========================

columns = [
    'timestamp', 'age', 'gender', 'cp', 'bp', 'cholesterol', 'fbs',
    'ekg', 'max_hr', 'ex_angina', 'st_depression', 'slope',
    'vessels', 'thallium', 'work_type', 'smoking_status',
    'prediction'
]

df_patients = pd.DataFrame(columns=columns)


# =========================
# Styling
# =========================

ui.add_head_html("""
<style>
    body {
        background: linear-gradient(135deg, #9ed0e4 0%, #7ba5db 100%);
        min-height: 100vh;
    }
    .main-card {
        background: white;
        border-radius: 28px;
        padding: 32px;
        margin: 40px auto;
        max-width: 1000px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        margin: 16px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 3px solid #3b82f6;
        display: inline-block;
    }
    .input-label {
        font-weight: 600;
        color: #334155;
        margin-bottom: 4px;
        font-size: 0.9rem;
    }
</style>
""")


# =========================
# GUI
# =========================

with ui.card().classes('main-card'):

    ui.label('Heart Disease Prediction').classes('text-h3 font-bold text-center text-primary')
    ui.label('Enter Patient Medical Data').classes('text-subtitle1 text-center text-grey-7 mb-4')
    ui.separator()

    ui.label('Basic Information').classes('section-title')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('Age').classes('input-label')
            age = ui.number(min=10, max=100).classes('w-full').props('outlined dense')

        with ui.column().classes('flex-1'):
            ui.label('Gender').classes('input-label')
            gender = ui.radio(['Female', 'Male']).props('inline dense')

    ui.label('Heart Symptoms').classes('section-title')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('Chest Pain Type').classes('input-label')
            cp = ui.select([
                'Typical Angina',
                'Atypical Angina',
                'Non-anginal Pain',
                'Asymptomatic'
            ]).classes('w-full').props('outlined dense')

        with ui.column().classes('flex-1'):
            ui.label('Resting Blood Pressure').classes('input-label')
            bp = ui.number(min=80, max=250, suffix=' mmHg').classes('w-full').props('outlined dense')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('Cholesterol').classes('input-label')
            cholesterol = ui.number(min=100, max=600, suffix=' mg/dl').classes('w-full').props('outlined dense')

        with ui.column().classes('flex-1'):
            ui.label('Fasting Blood Sugar > 120').classes('input-label')
            fbs = ui.radio(['No', 'Yes']).props('inline dense')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('EKG Results').classes('input-label')
            ekg = ui.select([
                'Normal',
                'ST-T Abnormality',
                'LVH'
            ]).classes('w-full').props('outlined dense')

        with ui.column().classes('flex-1'):
            ui.label('Maximum Heart Rate').classes('input-label')
            max_hr = ui.number(min=60, max=220, suffix=' bpm').classes('w-full').props('outlined dense')

    ui.label('Advanced Tests').classes('section-title')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('Exercise Induced Angina').classes('input-label')
            ex_angina = ui.radio(['No', 'Yes']).props('inline dense')

        with ui.column().classes('flex-1'):
            ui.label('ST Depression').classes('input-label')
            st_depression = ui.number(min=0.0, max=10.0, step=0.1).classes('w-full').props('outlined dense')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('Slope of ST').classes('input-label')
            slope = ui.select([
                'Upsloping',
                'Flat',
                'Downsloping'
            ]).classes('w-full').props('outlined dense')

        with ui.column().classes('flex-1'):
            ui.label('Number of Vessels (Fluoro)').classes('input-label')
            vessels = ui.select([0, 1, 2, 3]).classes('w-full').props('outlined dense')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('Thallium Stress Test').classes('input-label')
            thallium = ui.select([
                'Normal',
                'Fixed Defect',
                'Reversable Defect'
            ]).classes('w-full').props('outlined dense')

    ui.label('Additional Information').classes('section-title')

    with ui.row().classes('w-full gap-6 mb-4'):
        with ui.column().classes('flex-1'):
            ui.label('Work Type').classes('input-label')
            work_type = ui.select([
                'Private',
                'Self-employed',
                'Govt_job',
                'Children'
            ]).classes('w-full').props('outlined dense')

        with ui.column().classes('flex-1'):
            ui.label('Smoking Status').classes('input-label')
            smoking = ui.select([
                'never smoked',
                'formerly smoked',
                'smokes',
                'Unknown'
            ]).classes('w-full').props('outlined dense')

    ui.separator().classes('my-4')

    result_area = ui.column().classes('w-full mt-4 p-4 bg-grey-1 rounded-lg')
    button_container = ui.column().classes('w-full mt-2')

    def save_to_dataframe():
        global df_patients

        required_inputs = [
            age.value,
            gender.value,
            cp.value,
            bp.value,
            cholesterol.value,
            fbs.value,
            ekg.value,
            max_hr.value,
            ex_angina.value,
            st_depression.value,
            slope.value,
            vessels.value,
            thallium.value,
            work_type.value,
            smoking.value
        ]

        if any(value is None for value in required_inputs):
            ui.notify('Please fill all fields before prediction.', type='warning', position='top')
            return

        cp_map = {
            'Typical Angina': 1,
            'Atypical Angina': 2,
            'Non-anginal Pain': 3,
            'Asymptomatic': 4
        }

        ekg_map = {
            'Normal': 0,
            'ST-T Abnormality': 1,
            'LVH': 2
        }

        slope_map = {
            'Upsloping': 1,
            'Flat': 2,
            'Downsloping': 3
        }

        thallium_map = {
            'Normal': 3,
            'Fixed Defect': 6,
            'Reversable Defect': 7
        }

        fbs_val = 1 if fbs.value == 'Yes' else 0
        ex_angina_val = 1 if ex_angina.value == 'Yes' else 0

        # This must match the exact column names used during model training
        model_input = pd.DataFrame([{
            'Age': age.value,
            'Gender': gender.value,
            'Chest pain type': cp_map[cp.value],
            'BP': bp.value,
            'Cholesterol': cholesterol.value,
            'FBS over 120': fbs_val,
            'EKG results': ekg_map[ekg.value],
            'Max HR': max_hr.value,
            'Exercise angina': ex_angina_val,
            'ST depression': st_depression.value,
            'Slope of ST': slope_map[slope.value],
            'Number of vessels fluro': vessels.value,
            'Thallium': thallium_map[thallium.value],
            'work_type': work_type.value,
            'smoking_status': smoking.value
        }])

        try:
            prediction = pipe.predict(model_input)[0]

            probability_text = None

            if hasattr(pipe, 'predict_proba'):
                probabilities = pipe.predict_proba(model_input)[0]

                if hasattr(pipe, 'classes_'):
                    classes = list(pipe.classes_)

                    if 'Yes' in classes:
                        yes_index = classes.index('Yes')
                        probability_text = f'{probabilities[yes_index] * 100:.2f}%'
                    elif 1 in classes:
                        yes_index = classes.index(1)
                        probability_text = f'{probabilities[yes_index] * 100:.2f}%'

        except Exception as e:
            ui.notify(f'Prediction error: {e}', type='negative', position='top')
            result_area.clear()
            with result_area:
                ui.label('Prediction Error').classes('font-bold text-red-600 text-h5')
                ui.label(str(e)).classes('text-red-600')
            return

        new_row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'age': age.value,
            'gender': gender.value,
            'cp': cp_map[cp.value],
            'bp': bp.value,
            'cholesterol': cholesterol.value,
            'fbs': fbs_val,
            'ekg': ekg_map[ekg.value],
            'max_hr': max_hr.value,
            'ex_angina': ex_angina_val,
            'st_depression': st_depression.value,
            'slope': slope_map[slope.value],
            'vessels': vessels.value,
            'thallium': thallium_map[thallium.value],
            'work_type': work_type.value,
            'smoking_status': smoking.value,
            'prediction': prediction
        }

        df_patients = pd.concat(
            [df_patients, pd.DataFrame([new_row])],
            ignore_index=True
        )

        result_area.clear()

        with result_area:
            ui.label('Prediction Result').classes('font-bold text-primary text-h5 mb-2')

            if prediction == 'Yes' or prediction == 1:
                ui.label('High Risk of Heart Disease').classes('text-red-600 text-h5 font-bold')
            else:
                ui.label('Low Risk of Heart Disease').classes('text-green-600 text-h5 font-bold')

            ui.label(f'Model Output: {prediction}').classes('text-sm text-grey-7')

            if probability_text is not None:
                ui.label(f'Heart Disease Probability: {probability_text}').classes('text-lg text-grey-8')

            ui.separator().classes('my-2')

            ui.label(f'Data saved! Total records: {len(df_patients)}').classes('text-sm text-grey-7')
            ui.label(
                f'Age: {age.value} | Gender: {gender.value} | BP: {bp.value}'
            ).classes('text-sm')

        ui.notify('Prediction completed successfully.', type='positive', position='top')

        button_container.clear()
        with button_container:
            ui.button(
                'New Patient',
                on_click=reset_form,
                color='secondary'
            ).classes('w-full').props('size=large')

    def reset_form():
        result_area.clear()

        age.value = None
        gender.value = None
        cp.value = None
        bp.value = None
        cholesterol.value = None
        fbs.value = None
        ekg.value = None
        max_hr.value = None
        ex_angina.value = None
        st_depression.value = None
        slope.value = None
        vessels.value = None
        thallium.value = None
        work_type.value = None
        smoking.value = None

        button_container.clear()

        with button_container:
            ui.button(
                'Predict Heart Disease',
                on_click=save_to_dataframe,
                color='primary'
            ).classes('w-full mt-2').props('size=large')

        ui.notify('Form reset.', type='positive', position='top')

    with button_container:
        ui.button(
            'Predict Heart Disease',
            on_click=save_to_dataframe,
            color='primary'
        ).classes('w-full mt-2').props('size=large')

    def show_inputs():
        result_area.clear()

        with result_area:
            ui.label('Entered Data:').classes('font-bold text-primary mb-2')

            data = {
                'Age': age.value,
                'Gender': gender.value,
                'Chest Pain Type': cp.value,
                'Blood Pressure': f"{bp.value} mmHg",
                'Cholesterol': f"{cholesterol.value} mg/dl",
                'FBS > 120': fbs.value,
                'EKG Results': ekg.value,
                'Max Heart Rate': f"{max_hr.value} bpm",
                'Exercise Angina': ex_angina.value,
                'ST Depression': st_depression.value,
                'ST Slope': slope.value,
                'Number of Vessels': vessels.value,
                'Thallium': thallium.value,
                'Work Type': work_type.value,
                'Smoking Status': smoking.value
            }

            for key, val in data.items():
                ui.label(f'• {key}: {val}').classes('text-sm text-grey-8')

    ui.button(
        'Show Entered Data',
        on_click=show_inputs,
        color='primary'
    ).classes('w-full mt-2').props('size=large')


# =========================
# Run app
# =========================

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='Heart Disease Input Form',
        dark=False,
        port=8080,
        reload=False
    )