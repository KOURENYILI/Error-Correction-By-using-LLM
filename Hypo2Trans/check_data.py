import json
import math
import sys

def check_data(file_path):
    print("Starting to check the data...") 
    with open(file_path, 'r') as f:
        data = json.load(f)

   
    for i, item in enumerate(data):
        input1 = item.get('input1')
        input2 = item.get('input2', None)  
        output = item.get('output')

        
        if input1 is None or (input2 is not None and input2 is None) or output is None:
            print(f"Found None value at index {i}")

        
        if isinstance(input1, float) and (math.isnan(input1) or math.isinf(input1)):
            print(f"Found NaN or Inf in input1 at index {i}")
        if input2 is not None and isinstance(input2, float) and (math.isnan(input2) or math.isinf(input2)):
            print(f"Found NaN or Inf in input2 at index {i}")
        if isinstance(output, float) and (math.isnan(output) or math.isinf(output)):
            print(f"Found NaN or Inf in output at index {i}")

if __name__ == "__main__":
    print("Running check_data.py script...")
    if len(sys.argv) < 2:
        print("Usage: python check_data.py <file_path>")
    else:
        file_path = sys.argv[1]
        check_data(file_path)
        print("Data check completed. No issues found.") 
