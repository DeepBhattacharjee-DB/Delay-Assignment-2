import joblib
import pandas as pd

# Load the trained Decision Tree Classifier model
model = joblib.load('decision_tree_classifier.joblib')

# Define a function to make predictions (example)
def predict_logistics_delay(features):
    # Ensure features are in the correct order and format (e.g., DataFrame)
    # The 'features' input should match the x_train_processed format
    # For demonstration, let's assume 'features' is a list/array of numerical inputs
    
    # Example: Create a DataFrame from the input features
    # This part needs to be adjusted based on how you receive input in your app
    feature_names = ['Latitude', 'Longitude', 'Inventory_Level', 'Temperature', 'Humidity', 'Waiting_Time', 'User_Transaction_Amount', 'User_Purchase_Frequency', 'Asset_Utilization', 'Demand_Forecast']
    input_df = pd.DataFrame([features], columns=feature_names)
    
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    return prediction[0], prediction_proba[0]

if __name__ == '__main__':
    print("App started. Enter features to get a prediction.")
    print("Example: 20,30,40,1,2,50,45,88,90,81")
    
    # This is a simplified example for command-line input
    # In a real app, you'd get input from a web form or API request
    try:
        input_str = input("Enter comma-separated numerical features: ")
        input_features = [float(x.strip()) for x in input_str.split(',')]
        
        # Ensure the number of input features matches the model's expected features
        if len(input_features) == len(model.feature_names_in_):
            predicted_class, probabilities = predict_logistics_delay(input_features)
            print(f"Predicted Logistics Delay: {predicted_class}")
            print(f"Probabilities (No Delay, Delay): {probabilities}")
        else:
            print(f"Error: Expected {len(model.feature_names_in_)} features, but got {len(input_features)}.")
    except ValueError:
        print("Invalid input. Please enter a comma-separated list of numbers.")
    except Exception as e:
        print(f"An error occurred: {e}")
