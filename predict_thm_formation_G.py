import pandas as pd
import joblib
from stacking_model import StackingModel


def load_models():
    stacking_model = joblib.load('saved_models/stacking_model.pkl')
    xgb_model = joblib.load('saved_models/xgb_model.pkl')
    lgb_model = joblib.load('saved_models/lgb_model.pkl')
    return stacking_model, xgb_model, lgb_model


def batch_predict_thm(input_excel_path, output_excel_path):

    stacking_model, _, _ = load_models()


    required_columns = [
        'pH', 'UV254', 'DOC', 'TN', 'SUVA', 'Br',
        'chlorine dose', 'chlorine comsumption',
        'EEM_I', 'EEM_V'
    ]


    try:
        input_data = pd.read_excel(input_excel_path)
        print(f"load {len(input_data)} data")


        missing_cols = set(required_columns) - set(input_data.columns)
        if missing_cols:
            raise ValueError(f"lack: {missing_cols}")


        input_df = input_data[required_columns].copy()


        input_df = input_df.astype(float)


        print("\ninput data:")
        print(input_df.head())
        print("\nfeature:")
        print(input_df.describe())


        predictions = stacking_model.predict(input_df)
        input_data['THM_prediction'] = predictions


        input_data.to_excel(output_excel_path, index=False)
        print(f"\nsave results: {output_excel_path}")

        return input_data

    except Exception as e:
        print(f"error: {str(e)}")
        return None


if __name__ == "__main__":

    input_file = "THM formation prediction at 3.xlsx"
    output_file = "thm_formation_at3_prediction_results_.xlsx"


    results = batch_predict_thm(input_file, output_file)


    if results is not None:
        print("\nresults:")
        print(results[['UV254', 'chlorine dose', 'THM_prediction']].head())