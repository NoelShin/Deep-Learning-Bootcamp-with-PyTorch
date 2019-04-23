import random


class ImageBuffer(object):
    def __init__(self, n_images=50):
        """
        ImageBuffer is for updating the discriminator with historical images of the generator. By doing this, one can
        prevent severe artifacts which appears in synthetic images from the generator in an original setting without
        buffer. To this end, we store previous n_images and randomly picked one from the buffer or return the current
        synthesized image by 50% chance every iteration.
        For more information, you can check "Learning from Simulated and Unsupervised Images through Adversarial
        Training"
        """

        self.buffer = list()
        self.n_images = n_images

    def __call__(self, image):
        if len(self.buffer) < self.n_images:
            self.buffer.append(image)
        else:
            if random.random() > 0.5:
                new_image = image
                index = random.randint(0, self.n_images - 1)  # Get a random index in [0, n_images - 1]
                image = self.buffer[index]  # Draw image from the buffer.
                self.buffer[index] = new_image  # Replace the drawn image with current image.

            else:
                pass
        return image
