import numpy as np
import cv2
import os
class GaussNoiseGenerator:

    def __init__(self, dataset=None) -> None:
        pass

    def generate_gaussian_noise_image(self,mean_vector, covariance_matrix, target_size=(224, 224,3)):
        """
        Generate a Gaussian noise image using the provided mean vector and covariance matrix.
        """
        # Generate random samples from a multivariate Gaussian distribution
        noise_vector = np.random.multivariate_normal(mean_vector, covariance_matrix)

        # Reshape the noise vector back to an image
        noise_image = noise_vector.reshape(target_size)

        # Normalize the pixel values to [0, 255]
        noise_image = cv2.normalize(noise_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return noise_image

    def preprocess_image(self,image_path, target_size=(224,224,3)):
        """
        Preprocess an image by resizing and converting it to grayscale.
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)
        return image.flatten()  # Flatten the image into a 1D vector

    def compute_prior_gaussian(self,image_folder):
        """
        Compute the prior Gaussian distribution from a set of images.
        """
        image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]
        num_images = len(image_paths)

        # Initialize an array to store flattened image vectors
        image_vectors = np.zeros((num_images, 224*224*3))

        # Preprocess each image and stack them
        for i, image_path in enumerate(image_paths):
            image_vectors[i] = self.preprocess_image(image_path)

        # Compute the mean vector and covariance matrix
        mean_vector = np.mean(image_vectors, axis=0)
        covariance_matrix = np.cov(image_vectors, rowvar=False)

        return mean_vector, covariance_matrix

    # # Example usage:
    # image_folder = "path/to/your/image/folder"  # Update with your actual image folder
    # mean_vector, covariance_matrix = compute_prior_gaussian(image_folder)

    # print("Prior Gaussian Mean Vector:")
    # print(mean_vector)
    # print("\nPrior Gaussian Covariance Matrix:")
    # print(covariance_matrix)