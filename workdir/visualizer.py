import json

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Convert each line into a dictionary
            data.append(json.loads(line))
    return data

def main():
    filename = 'implementation_v1/20240306_201825/vis_data/scalars.json'  # Change this to your actual file path
    data = load_data(filename)
    
    # Example of working with the data: Print the 'loss' value of each entry
    evaluations = [step for step in data if 'mIoU' in step]

    for step in evaluations:
        print("step: {} mIoU: {}".format(step['step'], step['mIoU']))

if __name__ == "__main__":
    main()
