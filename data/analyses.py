import pandas as pd
from matplotlib import pyplot as plt

# Load the data
df = pd.read_csv('./hand_gesture_data_2.csv')
label_counts = df['label'].value_counts()

# Create a pie chart to visualize the distribution
plt.figure(figsize=(8, 8))

# Define a custom autopct function to show percentage and count
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({count})'
    return my_format

# Plot the pie chart
plt.pie(label_counts, labels=label_counts.index, autopct=autopct_format(label_counts), 
        startangle=140, colors=['#ff9999', '#66b3ff'])
plt.title(f'Distribution of Hand Gestures (Total: {df.shape[0]})', fontsize=16)
plt.ylabel('')
plt.show()