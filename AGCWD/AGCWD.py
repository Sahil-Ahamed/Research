import numpy as np
import cv2
import os

def agcwd(image, w=0.5):
    is_colorful = len(image.shape) >= 3
    img = extract_value_channel(image) if is_colorful else image
    img_pdf = get_pdf(img)
    max_intensity = np.max(img_pdf)
    min_intensity = np.min(img_pdf)

    # Apply weighting distribution
    w_img_pdf = max_intensity * (((img_pdf - min_intensity) / (max_intensity - min_intensity)) ** w)
    w_img_cdf = np.cumsum(w_img_pdf) / np.sum(w_img_pdf)

    # Intensity transformation
    l_intensity = np.arange(0, 256)
    l_intensity = np.array([255 * (e / 255) ** (1 - w_img_cdf[e]) for e in l_intensity], dtype=np.uint8)

    # Enhance the image
    enhanced_image = np.copy(img)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            intensity = enhanced_image[i, j]
            enhanced_image[i, j] = l_intensity[intensity]

    # Clip pixel intensities to [0.1, 0.9] and rescale to [0, 255]
    enhanced_image = np.clip(enhanced_image / 255.0, 0.1, 0.9)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    # Reapply value channel if the image is colorful
    enhanced_image = set_value_channel(image, enhanced_image) if is_colorful else enhanced_image
    return enhanced_image

def extract_value_channel(color_image):
    color_image = color_image.astype(np.float32) / 255.
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return np.uint8(v * 255)

def get_pdf(gray_image):
    height, width = gray_image.shape
    pixel_count = height * width
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist / pixel_count

def set_value_channel(color_image, value_channel):
    value_channel = value_channel.astype(np.float32) / 255
    color_image = color_image.astype(np.float32) / 255.
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_image[:, :, 2] = value_channel
    color_image = np.array(cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR) * 255, dtype=np.uint8)
    return color_image

def process_images(input_folder, output_folder, w=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(input_path):
            continue

        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping invalid image: {input_path}")
            continue

        enhanced_image = agcwd(image, w)

        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
        cv2.imwrite(output_path, enhanced_image)
        print(f"Processed and saved: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', help='Path to the input image folder')
    parser.add_argument('output_folder', help='Path to the output image folder')
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder, w=0.5)

if __name__ == '__main__':
    main()
