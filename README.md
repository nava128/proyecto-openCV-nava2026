# Computer vision course projects

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Coursework from the Computer Vision course, done with Kasra Noorbakhsh. The assignments move from classical image processing implemented by hand to pretrained CNN pipelines and end with a from-scratch conditional GAN, covering thresholding and color spaces, OCR without machine learning, pretrained classification/detection/segmentation models, an MLP baseline, and pix2pix-style image-to-image translation.

## Repository layout

| Folder | Topics |
|---|---|
| [`CA1`](CA1) | thresholding, RGB/grayscale/HSV, contrast adjustment, histograms, Gaussian/salt-and-pepper noise, hand-written convolution kernels, video denoising |
| [`CA2`](CA2) | CAPTCHA OCR pipeline: synthetic sample generation, denoising and sharpening, character segmentation, template matching with normalized cross-correlation |
| [`CA3`](CA3) | pretrained-model inference: EfficientNet-B4 classification, Faster R-CNN (ResNet50-FPN-v2) detection, DeepLabV3+ segmentation |
| [`CA4`](CA4) | MLP classifier on Fashion-MNIST, plus an improved variant with dropout and batch normalization |
| [`FinalProject`](FinalProject) | pix2pix (U-Net generator + PatchGAN discriminator) for paired image-to-image translation, plus a restoration-task variant |

## Project summaries

**CA1 — image processing fundamentals.** Grayscale and binary conversion, HSV color space, and contrast adjustment via mean-centered intensity scaling, worked through mathematically and in code. Part 2 adds Gaussian and salt-and-pepper noise and studies histograms and convolution kernels; the kernels (Gaussian, mean, median, Sobel, Laplacian, Canny) are implemented from scratch in [`filters.py`](CA1/filters.py) rather than called from OpenCV's high-level API. Part 3 applies the same pipeline to a synthetically noised video and evaluates how well each filter recovers the original frames.

**CA2 — CAPTCHA OCR without machine learning.** Generates synthetic three-character CAPTCHA samples, corrupts them with salt-and-pepper noise and Gaussian blur, then reverses the damage with denoising and unsharp masking. Characters are segmented out of the cleaned images, and recognition is done by comparing each segmented character against template images in [`Mapset`](CA2/Mapset) using normalized cross-correlation, with results (filename, predicted label, correlation score) written to a CSV.

**CA3 — pretrained CNNs for classification, detection, and segmentation.** Runs the same set of real-world images through three pretrained torchvision models: EfficientNet-B4 (ImageNet-1K) for classification behind a 12-step preprocessing pipeline (CLAHE, unsharp masking, aspect-preserving padding, Lanczos resizing), Faster R-CNN with a ResNet50-FPN-v2 backbone (COCO) for detection at a 0.5 confidence threshold, and DeepLabV3+ with a ResNet50 backbone (COCO, VOC label mapping) for semantic segmentation. Each section includes a per-image breakdown of what the model got right, where it struggled (small objects, overlapping instances), and how detection and segmentation outputs differ in practice.

**CA4 — MLP baseline on Fashion-MNIST.** A three-layer fully connected network (784 → 256 → 128 → 10, 235,146 parameters) trained with cross-entropy loss, with a confusion-matrix analysis of which classes it confuses (Shirt vs. Pullover vs. Coat) and why a flattened-vector model struggles there. An improved version (784 → 512 → 256 → 128 → 10 with dropout, batch normalization, and a lower learning rate) pushes test accuracy from roughly 87–88% to roughly 88–89%.

**Final project — pix2pix image-to-image translation.** A U-Net generator and 70×70 PatchGAN discriminator, implemented from scratch and trained on the edges2shoes split of the Kaggle pix2pix dataset, with adversarial loss combined with L1 reconstruction loss as in the original paper. The trained model reaches 16.91 dB PSNR and 0.728 SSIM on both validation and test sets, against an 8.04 dB / 0.640 input-as-output baseline, and ships with an interactive widget for browsing predictions by split and sample index. A second variant reconfigures the same architecture for image restoration (corrupted/masked inputs reconstructed back to the clean target) to match a later metrics requirement from the course staff.

## Stack

PyTorch and torchvision for the CNN and GAN work, OpenCV and NumPy for the classical image-processing assignments, run on Kaggle and Colab GPUs. Individual notebooks list their exact dependencies and datasets.
