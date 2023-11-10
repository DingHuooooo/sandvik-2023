import csv
import os

# Define the directory where the files are located
directory = "save/Tool/val"  # Replace with the actual directory path

# Get a list of file names in the directory
file_names = os.listdir(directory)

# Define the name of the CSV file to save
csv_file = "Tool.csv"

# Write the file names to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File Name'])  # Write the header row
    for file_name in file_names:
        writer.writerow([file_name])

print(f"File list has been saved to {csv_file}")
