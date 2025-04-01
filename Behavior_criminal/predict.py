import pandas as pd
import random
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
rf_model = joblib.load('RandomForest_best_model_with_class_weights.pkl')

# Define possible values for each attribute
crime_types = ["Murder", "Theft", "Cybercrime", "Fraud", "Kidnapping", "Assault", "Drug Offense", "Human Trafficking"]
locations = ["City", "Rural", "Metro"]
time_of_crime = ["Morning", "Evening", "Night"]
weapons = ["Gun", "Knife", "Digital Tools", "None"]
modus_operandi = ["Breaking & Entering", "Social Engineering", "Online Fraud", "Brute Force", "Extortion"]
psychological_profile = ["Aggressive", "Manipulative", "Opportunistic", "Calculated"]
escape_methods = ["Vehicle", "Running", "Fake Identity", "None"]
conviction_status = ["Convicted", "Under Trial", "Acquitted"]
sentence_duration = ["Short-term", "Long-term", "Life", "Death"]

# Assuming the model was trained with these columns
features = [
    "Age", "Gender", "Crime Type", "Crime Location", "Time of Crime", "Weapon Used", "Prior Offenses",
    "Modus Operandi", "Associated Gang", "Psychological Profile", "Footprint/Fingerprint Match",
    "Escape Method", "Conviction Status", "Sentence Duration", "Education Level", "Geographical Factor", "Job Status"
]

# Label encoder for categorical columns used during training
le = LabelEncoder()


# Randomly generate an individual's data
def generate_random_individual():
    individual = {
        "Age": random.choice(["18-25", "26-35", "36-50", "51+"]),
        "Gender": random.choice(["Male", "Female", "Other"]),
        "Crime Type": random.choice(crime_types),
        "Crime Location": random.choice(locations),
        "Time of Crime": random.choice(time_of_crime),
        "Weapon Used": random.choice(weapons),
        "Prior Offenses": random.choice(["Yes", "No"]),
        "Modus Operandi": random.choice(modus_operandi),
        "Associated Gang": random.choice(["Yes", "No"]),
        "Psychological Profile": random.choice(psychological_profile),
        "Footprint/Fingerprint Match": random.choice(["Yes", "No"]),
        "Escape Method": random.choice(escape_methods),
        "Conviction Status": random.choice(conviction_status),
        "Sentence Duration": random.choice(sentence_duration),
        "Education Level": random.choice(["High School", "Bachelor", "Master", "PhD"]),
        "Geographical Factor": random.choice(["Urban", "Suburban", "Rural"]),
        "Job Status": random.choice(["Employed", "Unemployed", "Student", "Retired"])
    }

    # Encode categorical features
    for col in individual.keys():
        individual[col] = le.fit_transform([individual[col]])[0]

    # Ensure all features are present in the same order as the trained model
    return pd.DataFrame([individual])[features]


# Function to predict if the individual is a criminal
def predict_criminal_status():
    # Generate random data
    random_individual = generate_random_individual()

    # Predict using the trained model
    prediction = rf_model.predict(random_individual)

    # Output the result
    print(f"Random Criminal Prediction:")
    print(random_individual)
    print(f"\nPredicted Crime Type: {prediction[0]}")  # Output predicted crime type


# Run the prediction
predict_criminal_status()
