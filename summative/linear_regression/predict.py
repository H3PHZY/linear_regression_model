import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FEATURE_NAMES = [
    "Women's Empowerment Group - 2022",
    "Global Gender Parity Index (GGPI) - 2022",
    "Gender Parity Group - 2022",
    "Human Development Group - 2021",
    "Sustainable Development Goal regions"
]

def load_models(model_path='best_model.pkl', scaler_path='scaler.pkl'):
    """Load the trained model and scaler."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error: Could not find model or scaler file. ({e})")
        print("Please ensure 'best_model.pkl' and 'scaler.pkl' exist in the same directory.")
        raise

def make_prediction(model, scaler, input_data):
    """
    Make a prediction using the provided model and scaler.
    
    Parameters:
    model: Trained sci-kit learn model
    scaler: Trained StandardScaler
    input_data: A Dictionary, Pandas DataFrame, List, or Numpy Array 
                containing the 5 features in the correct order.
                
    Returns:
    Numpy array of predictions
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, (list, np.ndarray)):
        if np.ndim(input_data) == 1:
            input_data = [input_data]
        df = pd.DataFrame(input_data, columns=FEATURE_NAMES)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("Unsupported input format. Please provide a dict, list, numpy array, or pandas DataFrame.")
        
    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input_data: {missing_cols}")
        
    df = df[FEATURE_NAMES]
    scaled_features = scaler.transform(df)
    prediction = model.predict(scaled_features)
    return prediction

if __name__ == "__main__":
    print("Loading model and scaler...")
    model, scaler = load_models()
    
    sample_input = {
        "Women's Empowerment Group - 2022": 1,
        "Global Gender Parity Index (GGPI) - 2022": 0.816,
        "Gender Parity Group - 2022": 1,
        "Human Development Group - 2021": 3,
        "Sustainable Development Goal regions": 2
    }
    
    print("\nSample input provided to the script:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
        
    prediction = make_prediction(model, scaler, sample_input)
    print("\n-------------------------------------------")
    print(f"Predicted Women's Empowerment Index (WEI): {round(prediction[0], 4)}")
    print("-------------------------------------------")
