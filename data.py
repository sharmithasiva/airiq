import pandas as pd
import os

data_dir = "airq/"  # Folder where the Excel files are saved
output_file = "aqi_8_cities_2021_2023.csv"

city_to_state = {
    "Delhi": "Delhi.xlsx",
    "Mumbai": "Maharasthra.xlsx",
    "Pune": "Maharashtra.xlsx",
    "Chennai": "TamilNadu.xlsx",
    "Kolkata": "WestBengal.xlsx",
    "Bengaluru": "Karnataka.xlsx",
    "Ahmedabad": "Gujarat.xlsx",
    "Hyderabad": "Telangana.xlsx"
}

merged_data = []

for city, state_file in city_to_state.items():
    file_path = os.path.join(data_dir, state_file)
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df['City'] = city
        merged_data.append(df)
    else:
        print(f"⚠️ File not found: {file_path}")

if merged_data:
    final_df = pd.concat(merged_data, ignore_index=True)

    # Detect date column dynamically
    date_col = None
    for col in final_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_col = col
            break
    
    if date_col:
        final_df['Datetime'] = pd.to_datetime(final_df[date_col],dayfirst=True)
        final_df = final_df.sort_values(['City', 'Datetime'])
    
    final_df.to_csv(output_file, index=False)
    print(f"✅ Combined CSV saved to {output_file}")
else:
    print("⚠️ No data to merge.")
