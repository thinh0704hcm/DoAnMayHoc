import csv

input_file = 'efficientnet_b7_predictions.csv'
output_file = 'small_predictions.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    count = 0

    for row in reader:
        # Check if the filename (first column) contains '-'
        if len(row) >= 2 and '-' in row[0]:
            writer.writerow(row)
            count += 1

print(f"Extracted {count} lines with dashes to {output_file}")
