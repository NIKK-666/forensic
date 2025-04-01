import pandas as pd
import random

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
education_levels = ["High School", "Undergraduate", "Graduate", "Postgraduate"]
job_status = ["Employed", "Unemployed", "Self-Employed", "Student"]
geographical_factors = ["Urban", "Suburban", "Rural"]

# Generate synthetic data
num_records = 10000  # Increase number of records

# Generate synthetic data
data = []

for i in range(1, num_records + 1):
    conviction = random.choice(conviction_status)
    sentence = random.choice(sentence_duration) if conviction == "Convicted" else "N/A"

    record = {
        "Criminal ID": i,
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
        "Conviction Status": conviction,
        "Sentence Duration": sentence,
        "Education Level": random.choice(education_levels),
        "Job Status": random.choice(job_status),
        "Geographical Factor": random.choice(geographical_factors)
    }
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Export dataset to CSV with "Criminal ID"
df.to_csv("enhanced_criminal_behavior_dataset.csv", index=False)

# Display the first few rows
print(df.head())
