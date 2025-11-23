import pandas as pd
from catboost import CatBoostRegressor
import numpy as np


def load_model():

    model = CatBoostRegressor()
    model.load_model('saved_models/catboost_model.cbm')


    if hasattr(model, 'feature_names_'):
        print("Training:", model.feature_names_)
    else:
        print("error")

    return model


def batch_predict(input_excel_path, output_excel_path):


    model = load_model()


    expected_columns = ['pH', 'UV254', 'DOC', 'TN', 'SUVA', 'Br',
                        'chlorine dose', 'EEM_I', 'EEM_V']


    try:
        input_data = pd.read_excel(input_excel_path)
        print(f"load {len(input_data)} data")


        missing_cols = set(expected_columns) - set(input_data.columns)
        if missing_cols:
            raise ValueError(f"lack: {missing_cols}")


        input_df = input_data[expected_columns].copy()


        input_df = input_df.astype(np.float32)
        print("\ninput data :")
        print(input_df.head())


        predictions = model.predict(input_df)
        input_data['Cl2_consumption_pred'] = predictions


        input_data.to_excel(output_excel_path, index=False)
        print(f"\n save data: {output_excel_path}")

        return input_data

    except Exception as e:
        print(f"error: {str(e)}")
        return None


if __name__ == "__main__":

    input_file = "Chlorine consumption prediction at 3.xlsx"
    output_file = "Predicted chlorine consumption at 3.xlsx"


    results = batch_predict(input_file, output_file)


    if results is not None:
        print("\nresults:")
        print(results[['UV254', 'chlorine dose', 'Cl2_consumption_pred']].head())