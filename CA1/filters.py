import numpy as np

def apply_filter(image, filter_type, **kwargs):

    """
    Apply various filters to an image without using high-level library functions.
    
    Parameters:
    image: numpy array representing the image (grayscale or RGB)
    filter_type: string specifying the filter type ('gaussian', 'mean', 'median', 'canny', 'sobel', 'laplacian')
    **kwargs: additional arguments for specific filters
    
    Returns:
    Filtered image as numpy array
    """

    def gaussian_kernel(size, sigma=1):
        """Create a Gaussian kernel"""
        kernel = np.zeros((size, size))
        center = size // 2

        # Calculate the Gaussian values
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        return kernel

    def convolve2d(image, kernel):
        """Perform 2D convolution operation"""
        # Pad the image to handle borders
        pad_size = kernel.shape[0] // 2
        padded_image = np.pad(image, pad_size, mode='edge')

        output = np.zeros_like(image, dtype=np.float64)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                roi = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                # Perform convolution
                output[i, j] = np.sum(roi * kernel)
        return output

    def mean_filter(image, size=3):
        """Apply mean filter"""
        kernel = np.ones((size, size)) / (size * size)
        if len(image.shape) == 3:  # Color image
            filtered_image = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                filtered_image[:, :, c] = convolve2d(image[:, :, c], kernel)
            return np.clip(filtered_image, 0, 255).astype(np.uint8)
        else:  # Grayscale image
            filtered_image = convolve2d(image, kernel)
            return np.clip(filtered_image, 0, 255).astype(np.uint8)

    def median_filter(image, size=3):
        """Apply median filter"""
        pad_size = size // 2
        padded_image = np.pad(image, pad_size, mode='edge')

        output = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if len(image.shape) == 3:  # Color image
                    for c in range(image.shape[2]):
                        roi = padded_image[i:i+size, j:j+size, c]
                        output[i, j, c] = np.median(roi)
                else:  # Grayscale
                    roi = padded_image[i:i+size, j:j+size]
                    output[i, j] = np.median(roi)

        return output

    def sobel_filter(image):
        """Apply Sobel edge detection filter"""
        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float64)
        
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float64)
        
        if len(image.shape) == 3:  # Color image
            # Convert to grayscale first
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        else:  # Grayscale image
            gray = image.astype(np.float64)

        # Apply the Sobel kernels
        grad_x = convolve2d(gray, sobel_x)
        grad_y = convolve2d(gray, sobel_y)

        # Calculate the gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255 range
        magnitude = np.clip(magnitude / np.max(magnitude) * 255, 0, 255)

        return magnitude.astype(np.uint8)

    def laplacian_filter(image):
        """Apply Laplacian edge detection filter"""
        # Define Laplacian kernel
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float64)

        if len(image.shape) == 3:  # Color image
            # Convert to grayscale first
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        else:  # Grayscale image
            gray = image.astype(np.float64)

        # Apply the Laplacian kernel
        result = convolve2d(gray, laplacian_kernel)

        # Take absolute value and normalize
        result = np.abs(result)
        result = np.clip(result / np.max(result) * 255, 0, 255)

        return result.astype(np.uint8)

    def canny_filter(image, low_threshold=50, high_threshold=150):
        """Simple Canny edge detection implementation"""
        if len(image.shape) == 3:  # Color image
            # Convert to grayscale first
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        else:  # Grayscale image
            gray = image.astype(np.float64)

        # Apply Gaussian blur (approximate)
        # Using a 5x5 Gaussian kernel with sigma=1
        gaussian_kernel_5x5 = gaussian_kernel(5, 1)
        blurred = convolve2d(gray, gaussian_kernel_5x5)

        # Compute gradients using Sobel
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float64)
        
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float64)

        grad_x = convolve2d(blurred, sobel_x)
        grad_y = convolve2d(blurred, sobel_y)

        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Non-maximum suppression (simplified)
        suppressed = np.zeros_like(magnitude)
        for i in range(1, magnitude.shape[0]-1):
            for j in range(1, magnitude.shape[1]-1):
                angle = direction[i, j]

                # Quantize angle to 0, 45, 90, or 135 degrees
                if (0 <= angle < np.pi/4) or (-np.pi <= angle < -3*np.pi/4):
                    q1, q2 = magnitude[i, j+1], magnitude[i, j-1]
                elif (np.pi/4 <= angle < np.pi/2) or (-3*np.pi/4 <= angle < -np.pi/2):
                    q1, q2 = magnitude[i+1, j-1], magnitude[i-1, j+1]
                elif (np.pi/2 <= angle < 3*np.pi/4) or (-np.pi/2 <= angle < -np.pi/4):
                    q1, q2 = magnitude[i+1, j], magnitude[i-1, j]
                else:  # (3*np.pi/4 <= angle <= np.pi) or (-np.pi/4 <= angle < 0)
                    q1, q2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

                if magnitude[i, j] >= q1 and magnitude[i, j] >= q2:
                    suppressed[i, j] = magnitude[i, j]

        # Double threshold
        weak = 50
        strong = 255
        thresholded = np.zeros_like(suppressed)
        thresholded[(suppressed >= low_threshold) & (suppressed <= high_threshold)] = weak
        thresholded[suppressed > high_threshold] = strong

        # Edge tracking by hysteresis (simplified)
        result = np.zeros_like(thresholded)
        strong_i, strong_j = np.where(thresholded == strong)
        result[strong_i, strong_j] = strong

        # Weak pixels connected to strong pixels become strong
        for i in range(1, result.shape[0]-1):
            for j in range(1, result.shape[1]-1):
                if thresholded[i, j] == weak:
                    if (thresholded[i-1:i+2, j-1:j+2] == strong).any():
                        result[i, j] = strong

        return result.astype(np.uint8)

    def gaussian_filter(image, size=5, sigma=1):
        """Apply Gaussian blur filter"""
        kernel = gaussian_kernel(size, sigma)
        if len(image.shape) == 3:  # Color image
            filtered_image = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                filtered_image[:, :, c] = convolve2d(image[:, :, c], kernel)
            return np.clip(filtered_image, 0, 255).astype(np.uint8)
        else:  # Grayscale image
            filtered_image = convolve2d(image, kernel)
            return np.clip(filtered_image, 0, 255).astype(np.uint8)

    # Apply the appropriate filter
    if filter_type.lower() == 'gaussian':
        size = kwargs.get('size', 5)
        sigma = kwargs.get('sigma', 1)
        return gaussian_filter(image, size, sigma)

    elif filter_type.lower() == 'mean':
        size = kwargs.get('size', 3)
        return mean_filter(image, size)

    elif filter_type.lower() == 'median':
        size = kwargs.get('size', 3)
        return median_filter(image, size)

    elif filter_type.lower() == 'sobel':
        return sobel_filter(image)

    elif filter_type.lower() == 'laplacian':
        return laplacian_filter(image)

    elif filter_type.lower() == 'canny':
        low_threshold = kwargs.get('low_threshold', 50)
        high_threshold = kwargs.get('high_threshold', 150)
        return canny_filter(image, low_threshold, high_threshold)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Supported types: 'gaussian', 'mean', 'median', 'canny', 'sobel', 'laplacian'")
