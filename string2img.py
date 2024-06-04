from PIL import Image
import pickle

with open('/home/jpuentes/CLIP/complete_balanced_data_350.pkl', 'rb') as f:
    # Load the pickled object
    data = pickle.load(f)


with open('/home/jpuentes/jpuentes2/CLIP/prost5_sequences_350.pkl', 'rb') as f:
    # Load the pickled object
    data_fold = pickle.load(f)


sequenceID = [sublist[0] for sublist in data]
from PIL import Image, ImageDraw
import random
def generate_image(text, color_map, image_width, image_height):
    # Define character size and padding
    char_size = 20  # Adjust as needed
    padding = 5  # Adjust as needed

    # Calculate the number of characters per row based on image width and character size
    chars_per_row = (image_width + padding) // (char_size + padding)

    # Calculate the number of rows needed based on text length and characters per row
    rows = (len(text) + chars_per_row - 1) // chars_per_row

    # Create a blank image
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    # Draw each character with the assigned color
    for i, char in enumerate(text):
        color = color_map[char]
        x = (i % chars_per_row) * (char_size + padding)
        y = (i // chars_per_row) * (char_size + padding)
        x += (image_width - (chars_per_row * (char_size + padding))) // 2  # Center horizontally
        y += (image_height - (rows * (char_size + padding))) // 2  # Center vertically
        draw.rectangle([x, y, x + char_size, y + char_size], fill=color)

    return image

# Generate and save images
color_map = {}
image_width = 500  # Specify your desired width
image_height = 500  # Specify your desired height
for i, string in enumerate(data_fold):
    
    name = sequenceID[i]
    if name == "NC_015157.1_CDS_0101":
        breakpoint()
    for char in string:
        if char not in color_map:
            color_map[char] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    image = generate_image(string, color_map, image_width, image_height)
    image.save(f"CLIP/foldseek_img/{name}.png")