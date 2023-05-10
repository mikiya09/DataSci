# time limit, I didn't write functions to collection all the output for plotting, instead I collect result manually and recording them for ploting
import matplotlib.pyplot as plt
import numpy as np



# =================================== 
# Different slack variable for regularization in Linear SVM (4:6) and (8:2)
slack_variable = ['baseline', 'w^2', 'sign(w)-L1', 'w-L2', 'w-L2(¬PCA)', 'w-L2(OvO)']
accuracy_slack_1 = [0.6, 0.1116, 0.5516, 0.60, 0.645, 0.621]
accuracy_slack_2 = [0.6, 0.1, 0.545, 0.62, 0.66, 0.675]

# Set the bar width
bar_width = 0.35

# Set the positions of the bars
ind = np.arange(len(slack_variable))

# Define the colors for the two groups of bars
colors_group1 = ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']
colors_group2 = ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']

# Create the side-by-side bar chart
fig, ax = plt.subplots()
bars1 = ax.bar(ind - bar_width / 2, accuracy_slack_1, bar_width, color=colors_group1, label='Split (4:6)')
bars2 = ax.bar(ind + bar_width / 2, accuracy_slack_2, bar_width, color=colors_group2, label='Split (8:2)')

# Add the labels and title
ax.set_xlabel('Slack Variable')
ax.set_ylabel('Accuracy')
ax.set_title('Hinge loss with different Slack Variables for different dataset splits')

# Set the x-axis tick labels and legend
ax.set_xticks(ind)
ax.set_xticklabels(slack_variable)
ax.legend()

# Remove horizontal gridlines
ax.yaxis.grid(False)

# Display the accuracy value on top of each bar
for bars, accuracy_slack in zip([bars1, bars2], [accuracy_slack_1, accuracy_slack_2]):
    for i, (bar, value) in enumerate(zip(bars, accuracy_slack)):
        # Set the text color to red for the last paired bars
        text_color = 'red' if i == len(slack_variable) - 1 else 'black'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                horizontalalignment='center', verticalalignment='bottom', fontsize=10, color=text_color)

plt.show()



# Different Kernels for dataset splits (4:6) and (8:2) 
# =================================== 
kernel_name = ['Linear_Kernel', 'RBF_Kernel', 'Laplace_RBF', 'Polynomial_Kernel']
kernel_accuracy_1 = [0.635, 0.66, 0.60, 0.2183]
kernel_accuracy_2 = [0.63, 0.725, 0.67, 0.235]


# Plot the lines with custom colors (blue and orange)
plt.plot(kernel_name, kernel_accuracy_1, label='Split (8:2)', marker='o', color='lightgreen')
plt.plot(kernel_name, kernel_accuracy_2, label='Split (4:6)', marker='o', color='lightblue')

plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('Accuracy with different Kernel choices for different dataset splits')
plt.grid(True)

plt.legend()

# Set the y-axis limits
min_accuracy = min(min(kernel_accuracy_1), min(kernel_accuracy_2))
max_accuracy = max(max(kernel_accuracy_1), max(kernel_accuracy_2))
plt.ylim(min_accuracy - 0.01, max_accuracy + 0.01)

# Find the maximum accuracy and its corresponding kernel index for each line
max_index_1 = kernel_accuracy_1.index(max(kernel_accuracy_1))
max_index_2 = kernel_accuracy_2.index(max(kernel_accuracy_2))

# Plot the dotted lines and display the highest values
for max_index, kernel_accuracy, line_color in zip([max_index_1, max_index_2], [kernel_accuracy_1, kernel_accuracy_2], ['red', 'red']):
    max_value = max(kernel_accuracy)
    plt.plot(kernel_name[max_index], max_value, 'o', color=line_color)
    plt.axhline(max_value, color=line_color, linestyle='--', xmax=max_index / (len(kernel_name) - 1))
    plt.axvline(max_index, color=line_color, linestyle='--', ymax=(max_value - min_accuracy + 0.01) / (max_accuracy - min_accuracy + 0.02))
    plt.text(max_index + 0.1, max_value, f"{max_value:.5f}", color=line_color, fontsize=12, verticalalignment='center')

plt.show()




# ===================================
# regularization with different C value with the kernel that generate highest accuracy with C = 10 (rbf kernel)
C = [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
accuracy_with_different_c = [0.39875, 0.718, 0.7275, 0.73, 0.72625, 0.72625, 0.72625, 0.725, 0.725, 0.7275, 0.725, 0.72625]

plt.plot(C, accuracy_with_different_c)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracy with different regularization parameter C (rbf)')
plt.grid(True)

# Set the y-axis limits
min_accuracy = min(accuracy_with_different_c)
max_accuracy = max(accuracy_with_different_c)
plt.ylim(min_accuracy - 0.01, max_accuracy + 0.01)

# Find the maximum accuracy and its corresponding C value
max_index = accuracy_with_different_c.index(max_accuracy)
max_C = C[max_index]

# Plot a red dot at the highest accuracy and draw a red line connecting it to the x-axis
plt.plot(max_C, max_accuracy, 'ro')
plt.axhline(max_accuracy, color='red', linestyle='--', xmax=(max_C - min(C)) / (max(C) - min(C)))

# Connect the highest y-value to its corresponding C value on the x-axis
plt.axvline(max_C, color='red', linestyle='--', ymax=(max_accuracy - min_accuracy + 0.01) / (max_accuracy - min_accuracy + 0.02))

# Display the highest y-value beside the red dotted line
plt.text(max_C + 1, max_accuracy, f"{max_accuracy:.5f}", color='red', fontsize=12, verticalalignment='center')

plt.show()





