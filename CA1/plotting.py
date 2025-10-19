import cv2
import numpy as np
import matplotlib.pyplot as plt

def plotImageComparisons(images, titles, cols=3, figsize=(15, 5)):
    """
    Plot multiple images in a grid for comparison.
    """
    rows = int(np.ceil(len(images) / cols))
    _, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle case where there's only one row or column
    if rows == 1:
        axes = axes if cols > 1 else [axes]
    elif cols == 1:
        axes = [axes] if rows == 1 else axes
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            # Ensure img is in correct format for imshow
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            # For RGB images, convert to float [0.0, 1.0] for safe rendering GRAY SCALE BUG
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_float = img.astype(np.float32) / 255.0
                axes[i].imshow(img_float, cmap=None)  # Explicit RGB
            else:
                axes[i].imshow(img, cmap='gray')

            axes[i].axis('off')
            axes[i].set_title(title)

    # Hide any unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plotColorChannels(imageRGB, imageHSV=None):

    """
    Plot RGB and optionally HSV channels of an image.
    
    Args:
        imageRGB: RGB image
        imageHSV: Optional HSV image to show HSV channels
    """

    if imageHSV is not None:
        # Plot RGB and HSV channels together
        rChannel = imageRGB[:, :, 0]
        gChannel = imageRGB[:, :, 1]
        bChannel = imageRGB[:, :, 2]

        hChannel = imageHSV[:, :, 0]
        sChannel = imageHSV[:, :, 1]
        vChannel = imageHSV[:, :, 2]

        _, axes = plt.subplots(2, 4, figsize=(15, 10))

        # RGB channels
        axes[0, 0].imshow(rChannel, cmap='Reds')
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Red Channel')

        axes[0, 1].imshow(gChannel, cmap='Greens')
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Green Channel')

        axes[0, 2].imshow(bChannel, cmap='Blues')
        axes[0, 2].axis('off')
        axes[0, 2].set_title('Blue Channel')

        axes[0, 3].imshow(imageRGB)
        axes[0, 3].axis('off')
        axes[0, 3].set_title('Original RGB')

        # HSV channels
        axes[1, 0].imshow(hChannel, cmap='hsv')
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Hue Channel')

        axes[1, 1].imshow(sChannel, cmap='gray')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Saturation Channel')

        axes[1, 2].imshow(vChannel, cmap='gray')
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Value Channel')

        hsvAsRgb = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2RGB)
        axes[1, 3].imshow(hsvAsRgb)
        axes[1, 3].axis('off')
        axes[1, 3].set_title('HSV to RGB')

    else:
        # Plot just RGB channels
        rChannel = imageRGB[:, :, 0]
        gChannel = imageRGB[:, :, 1]
        bChannel = imageRGB[:, :, 2]

        _, axes = plt.subplots(1, 4, figsize=(15, 5))

        axes[0].imshow(rChannel, cmap='Reds')
        axes[0].axis('off')
        axes[0].set_title('Red Channel')

        axes[1].imshow(gChannel, cmap='Greens')
        axes[1].axis('off')
        axes[1].set_title('Green Channel')

        axes[2].imshow(bChannel, cmap='Blues')
        axes[2].axis('off')
        axes[2].set_title('Blue Channel')

        axes[3].imshow(imageRGB)
        axes[3].axis('off')
        axes[3].set_title('Original RGB')

    plt.tight_layout()
    plt.show()


def plotHistograms(imageRGB, grayImage=None):

    """
    Plot RGB and grayscale histograms.
    
    Args:
        imageRGB: RGB image for RGB histograms
        grayImage: Grayscale image for grayscale histogram
    """

    _, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original RGB image
    axes[0, 0].imshow(imageRGB)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Original RGB Image')

    # RGB histograms
    axes[0, 1].set_title('RGB Histograms')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')

    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist = cv2.calcHist([imageRGB], [i], None, [256], [0, 256])
        axes[0, 1].plot(hist, color=color, label=f'{color.capitalize()} channel')
    axes[0, 1].legend()

    if grayImage is not None:
        # Grayscale histogram using OpenCV
        grayHist = cv2.calcHist([grayImage], [0], None, [256], [0, 256])
        axes[1, 0].plot(grayHist, color='gray')
        axes[1, 0].set_title('Grayscale Histogram')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')

        # Grayscale histogram using matplotlib
        axes[1, 1].hist(grayImage.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        axes[1, 1].set_title('Grayscale Histogram (Matplotlib)')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
    else:
        # Just disable the unused subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def plotHistogramComparison(original, stretched, equalized, imageRGB):

    """
    Plot histogram comparison between original, stretched, and equalized images.
    
    Args:
        original: Original grayscale image
        stretched: Stretched grayscale image
        equalized: Equalized grayscale image
        imageRGB: Original RGB image for comparison
    """

    histOriginal = cv2.calcHist([original], [0], None, [256], [0, 256])
    histStretched = cv2.calcHist([stretched], [0], None, [256], [0, 256])
    histEqualized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    _, axes = plt.subplots(2, 4, figsize=(18, 12))

    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Original Grayscale')

    # Stretched image
    axes[0, 1].imshow(stretched, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Histogram Stretched')

    # Equalized image
    axes[0, 2].imshow(equalized, cmap='gray')
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Histogram Equalized')

    # Original RGB for comparison
    axes[0, 3].imshow(imageRGB)
    axes[0, 3].axis('off')
    axes[0, 3].set_title('Original RGB')

    # Original histogram
    axes[1, 0].plot(histOriginal, color='gray')
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')

    # Stretched histogram
    axes[1, 1].plot(histStretched, color='gray')
    axes[1, 1].set_title('Stretched Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')

    # Equalized histogram
    axes[1, 2].plot(histEqualized, color='gray')
    axes[1, 2].set_title('Equalized Histogram')
    axes[1, 2].set_xlabel('Pixel Intensity')
    axes[1, 2].set_ylabel('Frequency')

    # Combined histogram comparison
    axes[1, 3].plot(histOriginal, color='red', label='Original', alpha=0.7)
    axes[1, 3].plot(histStretched, color='blue', label='Stretched', alpha=0.7)
    axes[1, 3].plot(histEqualized, color='green', label='Equalized', alpha=0.7)
    axes[1, 3].set_title('Histogram Comparison')
    axes[1, 3].set_xlabel('Pixel Intensity')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].legend()

    plt.tight_layout()
    plt.show()


def calculateContrast(img):

    """
    Calculate the contrast of an image as the standard deviation of pixel intensities.
    
    Args:
        img: Input image
    
    Returns:
        Contrast value (standard deviation)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

    return np.std(gray)


def adjustContrast(image, factor):

    """
    Adjust the contrast of an image.
    
    Args:
        image: Input image
        factor: Contrast adjustment factor (>1 increases contrast, <1 decreases)
    
    Returns:
        Contrast-adjusted image
    """

    # Convert to float to avoid overflow
    imgFloat = image.astype(np.float64)
    mean = np.mean(imgFloat)
    # Apply contrast adjustment
    adjusted = mean + factor * (imgFloat - mean)
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)
