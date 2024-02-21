import matplotlib.pyplot as plt

def plot_data (class_counts, directory):

    plt.style.use('ggplot')  # Set the plot style
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed


    class_counts.plot(kind='bar', color='royalblue', width=0.8)
    plt.xlabel('Classes', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1 * class_counts.max())  # Set the y-axis limits
    plt.xticks([], [])  # Remove x-axis ticks and labels
    plt.grid(1)  # Add gridlines for y-axis
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f'{directory}/class_counts.png')  # Save the plot to a file


def scatter_embeddings (embeddings_flat, name, directory):

    plt.scatter(embeddings_flat[:, 0], embeddings_flat[:, 1])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'{name} Embeddings')
    plt.savefig(f'directory/{name}_embeddings.png')