import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    y_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into x and y values
            try:
                _, y = map(int, line.split())
                y_values.append(y)
            except ValueError:
                print(f"Skipping line: {line.strip()} - unable to parse.")
        x_values = np.linspace(1, len(y_values),len(y_values))
    return x_values, y_values

def plot_data(x1, y1, y2, label1, label2):
    bar_width = 0.5  # Width of the bars
    index = np.arange(len(x1))  # The label locations
    plt.figure(figsize=(10, 6))
    # Create bars for the first dataset
    bars1 = plt.bar(index, y1, bar_width, label=label1, color='b', alpha=0.7)
    # Create bars for the second dataset, shifted by bar_width
    bars2 = plt.bar(index + bar_width, y2, bar_width, label=label2, color='r', alpha=0.7)
    
    for i in range(len(x1) + 1):
            plt.axvline(x=i * bar_width * 2 + 0.75, color='black', linewidth=0.5)

    plt.title('AI agent vs deterministic agent')
    plt.xlabel('Эпизод')
    plt.ylabel('Шаги')
    #plt.xticks(index + bar_width / 2, x1)  # Set x-ticks to be in the center of the grouped bars
    plt.legend()  # Add a legend to differentiate the datasets
    plt.grid(axis='y')  # Add grid lines for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.show()


file_path1 = 'Neuro.txt'  # Change this to your first file path
file_path2 = 'Determ.txt'  # Change this to your second file path

x1, y1 = read_data(file_path1)
x2, y2 = read_data(file_path2)

counter = 0

for i in range(len(y2)):
    if y1[i] <= y2[i] and y1[i] != 0:
        counter += 1

print(f"AI did better {counter} times out of {len(y2)} ({(float(counter) / len(y2)) * 100:.0f}%)")

plot_data(x1, y1, y2, label1='Модель', label2='Алгоритм')