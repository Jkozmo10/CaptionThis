import pandas as pd

def sort_csv_by_image_filename(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Extract the numeric part of the 'Image Filename' and convert to integer for sorting
    df['Sort Key'] = df['Image Filename'].str.extract('(\d+)').astype(int)

    # Sorting the DataFrame by the numeric key in ascending order
    sorted_df = df.sort_values(by='Sort Key')

    # Dropping the temporary 'Sort Key' column
    sorted_df = sorted_df.drop(columns=['Sort Key'])

    # Save the sorted DataFrame to a new CSV file
    sorted_df.to_csv(output_csv, index=False)

# Replace 'input.csv' and 'output.csv' with your actual file names
input_csv = 'downloaded_images.csv'  # Replace with your input CSV file name
output_csv = 'sorted_output.csv'  # Replace with your desired output CSV file name

# Call the function
sort_csv_by_image_filename(input_csv, output_csv)

print(f"Sorted CSV saved to {output_csv}")
