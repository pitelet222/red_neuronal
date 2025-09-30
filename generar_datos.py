import numpy as np
import csv
import random
import pandas as pd


# i create a funciton that generates a single sample to train the network

def generate_sample():

    people_num = random.randint(1,7)
    meters = random.randint(50,201)
    have_gas = random.randint(0,2)
    have_induc = random.randint(0,2)
    have_vitro = random.randint(0,2)
    have_AC = random.randint(0,2)
    have_calef = random.randint(0,2)
    month = random.randint(1,12)
    day_of_week = random.randint(0,7)

    # i define  regular consume and conditions  inside consume

    consumption = 2.0 * people_num

    consumption += meters * 0.05

    if have_induc == 1:
        consumption += 3.5
    elif have_vitro ==  1:
        consumption += 2.5

    if have_gas == 1:
        consumption += 1.0
    else:
        consumption += 4.0

    temperature = {
        1: 9, 2: 10, 3: 12, 4: 14, 5: 17, 6: 21,
        7: 24, 8: 25, 9: 22, 10: 18, 11: 13, 12: 10
    }[month]
    # define conditions also depending on the month and AC or Cale

    if have_AC == 1:
        if temperature >= 24:  # Julio-Agosto
            consumption += 8.0 + (temperature - 24) * 1.0
        elif temperature >= 21:  # Junio-Septiembre
            consumption += 4.0

    if have_calef == 1:
        if temperature <= 10:  # Diciembre-Febrero
            consumption += 12.0 + (10 - temperature) * 1.5
        elif temperature <= 14:  # Noviembre, Marzo
            consumption += 6.0

    if month in [6,7,8,9] and have_AC == 1:
        consumption += 8.0
    
    if month in [1,2,11,12] and have_calef == 1:
        consumption += 12.0

    # also add some noise 

    noise = np.random.normal(0,1.5)
    consumption += noise

    # establish a realistic min to avoid negative numbers 

    consumption = max(consumption, 2.0)
    
    features = [people_num, meters, have_gas, have_induc, have_vitro, have_AC, have_calef, month, day_of_week]

    return features, consumption

# create a function that generates the dataset from the samples generated

def generate_dataset(n_sample=5000, saved=True, file="data/sintetic_dataset.csv"):
    data = []


    for i in range(n_sample):

        features, consumption = generate_sample()
        row = features + [consumption]
        data.append(row)

        if (i+1) % 1000 == 0:
            print(f"{i+1}/{n_sample} samples generated")

    columns = [
        "People_num",
        "Meters",
        "Have_gas",
        "Have_induction",
        "Have_vitro",
        "Have_AC",
        "Have_calef",
        "Monthm",
        "Day_of_the_week",
        "Kwh_daily_consumption"
    ]

    df = pd.DataFrame(data, columns=columns)

    if saved:
        df.to_csv(file, index=False)
        print(f"\nDataset saved in: {file}")
        print(f"Consumption statistics:")
        print(df["Kwh_daily_consumption"].describe())

    return df

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    df = generate_dataset(n_sample=5000)
    





