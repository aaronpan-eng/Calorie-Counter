import os
import pyheif
from PIL import Image

def heic_to_jpg(heic_file, jpg_file):
    # Open the HEIC file
    heif_file = pyheif.read(heic_file)

    # Convert to PIL Image
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Save as JPG
    image.save(jpg_file, format="JPEG")

def convert_heic_folder_to_jpg(heic_folder, jpg_folder):
    # Create JPG folder if it doesn't exist
    if not os.path.exists(jpg_folder):
        os.makedirs(jpg_folder)

    # Iterate through HEIC files in the directory
    for filename in os.listdir(heic_folder):
        if filename.endswith(".heic"):
            heic_file = os.path.join(heic_folder, filename)
            jpg_file = os.path.join(jpg_folder, os.path.splitext(filename)[0] + ".jpg")
            heic_to_jpg(heic_file, jpg_file)
            print(f"Converted: {filename}")

if __name__ == "__main__":
    heic_folder_path = "heic_folder"  # Change this to your HEIC folder path
    jpg_folder_path = "jpg_folder"    # Change this to desired JPG folder path

    convert_heic_folder_to_jpg(heic_folder_path, jpg_folder_path)
    print("Conversion completed.")