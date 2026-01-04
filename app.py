"""
Digital Image Edge Detection Application
=========================================
A Streamlit-based web application for detecting edges in digital images
using various edge detection algorithms including Sobel, Roberts, Prewitt,
Laplacian, Frei-Chen, and Canny operators.

Author: Senior Python Developer & Computer Vision Expert
Version: 1.0.0
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def inject_custom_css(theme='light'):
    """
    Inject custom CSS to style the Streamlit application.
    Handles both Light and Dark themes.
    """
    # Define colors based on theme
    if theme == 'dark':
        bg_color = "#101622"
        sidebar_bg = "#0d121c"
        card_bg = "#1c1f27"
        text_color = "#ffffff"
        subtext_color = "#9da6b9"
        border_color = "#282e39"
        primary_color = "#135bec"
        primary_hover = "#1d4ed8"
        success_bg = "rgba(5, 150, 105, 0.1)"
        success_text = "#34d399"
        success_border = "rgba(5, 150, 105, 0.2)"
    else:  # light
        bg_color = "#ffffff"
        sidebar_bg = "#f8fafc"
        card_bg = "#ffffff"
        text_color = "#1f2937"
        subtext_color = "#6b7280"
        border_color = "#e5e7eb"
        primary_color = "#059669"
        primary_hover = "#047857"
        success_bg = "#ecfdf5"
        success_text = "#059669"
        success_border = "#a7f3d0"

    st.markdown(f"""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global settings */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {text_color};
        }}

        /* Hide Streamlit default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Main container styling */
        .stApp {{
            background-color: {bg_color};
        }}
        
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid {border_color};
        }}
        
        [data-testid="stSidebar"] .block-container {{
            padding-top: 2rem;
        }}
        
        /* Title styling */
        .main-title {{
            color: {text_color};
            font-size: 3rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }}
        
        .main-title span {{
            color: {primary_color};
        }}
        
        .subtitle {{
            color: {subtext_color};
            text-align: center;
            font-size: 1.125rem;
            margin-bottom: 3rem;
            font-weight: 400;
        }}
        
        /* Card styling for image containers */
        .image-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }}
        
        .image-card-title {{
            color: {text_color};
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        /* Button styling */
        .stButton > button {{
            width: 100%;
            background-color: {primary_color} !important;
            color: #ffffff !important;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px {primary_color}33, 0 2px 4px -1px {primary_color}1a;
        }}
        
        .stButton > button:hover {{
            background-color: {primary_hover} !important;
            color: #ffffff !important;
            box-shadow: 0 10px 15px -3px {primary_color}66, 0 4px 6px -2px {primary_color}33;
            transform: translateY(-1px);
        }}
        
        .stButton > button p {{
            color: #ffffff !important;
        }}
        
        /* Download button styling */
        .stDownloadButton > button {{
            width: 100%;
            background-color: {card_bg};
            color: {primary_color};
            border: 1px solid {primary_color};
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.2s ease;
        }}
        
        .stDownloadButton > button:hover {{
            background-color: {bg_color};
            color: {primary_hover};
        }}
        
        /* Radio button styling */
        .stRadio > div {{
            background: transparent;
            padding: 0;
        }}
        
        .stRadio > div > label {{
            color: {text_color} !important;
        }}
        
        /* File uploader styling */
        [data-testid="stFileUploader"] {{
            background: {card_bg};
            border: 1px dashed {border_color};
            border-radius: 12px;
            padding: 2rem;
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {primary_color};
            background: {card_bg};
        }}
        
        /* Slider styling */
        /* Removed custom slider styling to prevent conflicts */
        
        /* Info box styling */
        .stAlert {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            color: {subtext_color};
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background: {card_bg};
            border-radius: 8px;
            border: 1px solid {border_color};
            color: {text_color};
        }}
        
        /* Sidebar header */
        .sidebar-header {{
            color: {text_color};
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid {border_color};
        }}
        
        /* Algorithm info card */
        .algo-info {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }}
        
        .algo-info h4 {{
            color: {primary_color};
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        
        .algo-info p {{
            color: {subtext_color};
            font-size: 0.875rem;
            line-height: 1.5;
        }}
        
        /* Success message */
        .success-msg {{
            background: {success_bg};
            color: {success_text};
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 1px solid {success_border};
            text-align: center;
            margin: 1rem 0;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        
        /* Metric styling */
        [data-testid="stMetric"] {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 1rem;
        }}
        
        /* Custom headers */
        h1, h2, h3 {{
            color: {text_color} !important;
            font-weight: 700 !important;
        }}
        
        p, label {{
            color: {subtext_color} !important;
        }}
        
        /* Helper classes for dynamic styling */
        .dynamic-card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1rem;
        }}
        
        .dynamic-text {{
            color: {text_color};
        }}
        
        .dynamic-subtext {{
            color: {subtext_color};
        }}
        
        .upload-instruction {{
            background: {card_bg};
            border: 1px dashed {border_color};
            border-radius: 16px;
            padding: 3rem;
            text-align: center;
            margin: 2rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def apply_sobel_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection operator.
    
    The Sobel operator uses two 3x3 kernels for detecting horizontal and vertical edges:
    
    Kernel X (Horizontal):        Kernel Y (Vertical):
    [-1  0  1]                    [-1 -2 -1]
    [-2  0  2]                    [ 0  0  0]
    [-1  0  1]                    [ 1  2  1]
    
    The gradient magnitude is computed as: G = |Gx| + |Gy| (approximation)
    
    Args:
        gray_image: Grayscale input image as numpy array
        
    Returns:
        Edge-detected image as uint8 numpy array
    """
    # Apply Sobel in X direction (detects vertical edges)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel in Y direction (detects horizontal edges)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Convert to absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combine both gradients with equal weights
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return output


def apply_roberts_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Roberts Cross edge detection operator.
    
    The Roberts Cross operator uses two 2x2 kernels for diagonal edge detection:
    
    Kernel X (45°):    Kernel Y (135°):
    [1   0]            [0   1]
    [0  -1]            [-1  0]
    
    This operator is particularly sensitive to diagonal edges and is one of the
    earliest edge detection operators. It computes the gradient at a 45° angle.
    
    Args:
        gray_image: Grayscale input image as numpy array
        
    Returns:
        Edge-detected image as uint8 numpy array
    """
    # Define Roberts Cross kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    
    # Apply convolution with each kernel
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    
    # Convert to absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combine both gradients
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return output


def apply_prewitt_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Prewitt edge detection operator.
    
    The Prewitt operator uses two 3x3 kernels similar to Sobel but with uniform weights:
    
    Kernel X (Horizontal):        Kernel Y (Vertical):
    [-1  0  1]                    [ 1  1  1]
    [-1  0  1]                    [ 0  0  0]
    [-1  0  1]                    [-1 -1 -1]
    
    Unlike Sobel, Prewitt doesn't emphasize the center pixel, making it 
    less sensitive to noise but also slightly less accurate for edge localization.
    
    Args:
        gray_image: Grayscale input image as numpy array
        
    Returns:
        Edge-detected image as uint8 numpy array
    """
    # Define Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
    
    # Apply convolution with each kernel
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    
    # Convert to absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combine both gradients
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return output


def apply_laplacian_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian edge detection operator.
    
    The Laplacian operator is a second-order derivative operator that detects
    edges by finding zero-crossings. The commonly used kernel is:
    
    [0   1  0]
    [1  -4  1]
    [0   1  0]
    
    Or the diagonal-inclusive version:
    [1   1  1]
    [1  -8  1]
    [1   1  1]
    
    The Laplacian detects edges in all directions simultaneously and is 
    sensitive to noise, so it's often used with Gaussian smoothing (LoG).
    
    Args:
        gray_image: Grayscale input image as numpy array
        
    Returns:
        Edge-detected image as uint8 numpy array
    """
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
    
    # Convert to absolute values and scale to uint8
    output = cv2.convertScaleAbs(laplacian)
    
    return output


def apply_frei_chen_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Frei-Chen edge detection operator.
    
    The Frei-Chen operator is similar to Sobel but uses √2 instead of 2
    in the center positions for better isotropy:
    
    Kernel X:                     Kernel Y:
    [-1   0    1]                 [-1  -√2  -1]
    [-√2  0   √2]                 [ 0   0    0]
    [-1   0    1]                 [ 1   √2   1]
    
    The √2 weighting provides a more uniform response to edges at different
    orientations compared to the Sobel operator, offering better rotational
    symmetry in edge detection.
    
    Args:
        gray_image: Grayscale input image as numpy array
        
    Returns:
        Edge-detected image as uint8 numpy array
    """
    sqrt2 = np.sqrt(2)
    
    # Define Frei-Chen kernels with √2 weighting
    kernel_x = np.array([
        [-1, 0, 1],
        [-sqrt2, 0, sqrt2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    kernel_y = np.array([
        [-1, -sqrt2, -1],
        [0, 0, 0],
        [1, sqrt2, 1]
    ], dtype=np.float64)
    
    # Apply convolution with each kernel
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    
    # Convert to absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combine both gradients
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return output


def apply_canny_edge_detection(gray_image: np.ndarray, 
                                low_threshold: int = 50, 
                                high_threshold: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection algorithm.
    
    The Canny edge detector is a multi-stage algorithm:
    1. Noise Reduction: Apply Gaussian filter to smooth the image
    2. Gradient Calculation: Find intensity gradients using Sobel
    3. Non-maximum Suppression: Thin edges to 1-pixel width
    4. Double Thresholding: Classify edges as strong, weak, or non-edges
    5. Edge Tracking by Hysteresis: Connect weak edges to strong edges
    
    Args:
        gray_image: Grayscale input image as numpy array
        low_threshold: Lower threshold for hysteresis (weak edges below this are discarded)
        high_threshold: Upper threshold for hysteresis (edges above this are strong)
        
    Returns:
        Binary edge-detected image as uint8 numpy array
    """
    # Apply Canny edge detection
    output = cv2.Canny(gray_image, low_threshold, high_threshold)
    
    return output


def apply_dilation(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Dilation morphological operation."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(gray_image, kernel, iterations=1)


def apply_erosion(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Erosion morphological operation."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(gray_image, kernel, iterations=1)


def apply_opening(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Opening morphological operation (Erosion followed by Dilation)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)


def apply_closing(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Closing morphological operation (Dilation followed by Erosion)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)


def apply_morphological_gradient(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Morphological Gradient (Difference between Dilation and Erosion)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)


def apply_region_filling(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Region Filling (Hole Filling).
    Fills holes in the image. Works best on binary images.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Threshold to binary using Otsu's method
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if background is white (assuming top-left corner is background)
    # If background is white (255), invert the image so background becomes black (0)
    if binary[0, 0] == 255:
        binary = cv2.bitwise_not(binary)
    
    # Copy the binary image
    im_floodfill = binary.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels larger than the image.
    h, w = binary.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground
    im_out = binary | im_floodfill_inv
    
    return im_out


def process_image(image: Image.Image, algorithm: str, 
                  canny_low: int = 50, canny_high: int = 150,
                  kernel_size: int = 3) -> np.ndarray:
    """
    Main image processing function that applies the selected edge detection algorithm.
    
    This function handles the complete pipeline:
    1. Convert PIL Image to OpenCV format (numpy array)
    2. Convert to grayscale
    3. Apply optional noise reduction (Gaussian blur)
    4. Apply selected edge detection algorithm
    5. Normalize and convert to uint8 for display
    
    Args:
        image: PIL Image object (RGB)
        algorithm: Name of the edge detection algorithm to apply
        canny_low: Low threshold for Canny (only used if algorithm is 'Canny')
        canny_high: High threshold for Canny (only used if algorithm is 'Canny')
        kernel_size: Kernel size for morphological operations
        
    Returns:
        Edge-detected image as uint8 numpy array (0-255)
        
    Raises:
        ValueError: If an unknown algorithm is specified
    """
    # Convert PIL image to OpenCV format (numpy array)
    img_array = np.array(image.convert('RGB'))
    
    # Convert from RGB to Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur for noise reduction (improves edge detection quality)
    # Only apply blur for edge detection algorithms, not necessarily for morphology if we want raw effect
    # But for consistency and noise reduction, we'll keep it, or maybe make it optional.
    # For now, let's keep it as it helps with most operations.
    if algorithm not in ['Region Filling']: # Region filling needs binary, blur might affect thresholding slightly but usually ok.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Dictionary mapping algorithm names to their functions
    algorithm_map = {
        'Sobel': apply_sobel_edge_detection,
        'Roberts': apply_roberts_edge_detection,
        'Prewitt': apply_prewitt_edge_detection,
        'Laplacian': apply_laplacian_edge_detection,
        'Frei-Chen': apply_frei_chen_edge_detection,
        'Region Filling': apply_region_filling
    }
    
    morphology_map = {
        'Dilation': apply_dilation,
        'Erosion': apply_erosion,
        'Opening': apply_opening,
        'Closing': apply_closing,
        'Morphological Gradient': apply_morphological_gradient
    }
    
    # Apply the selected algorithm
    if algorithm == 'Canny':
        output = apply_canny_edge_detection(gray, canny_low, canny_high)
    elif algorithm in morphology_map:
        output = morphology_map[algorithm](gray, kernel_size)
    elif algorithm in algorithm_map:
        output = algorithm_map[algorithm](gray)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Ensure output is normalized to 0-255 and converted to uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output


def convert_image_for_download(image_array: np.ndarray) -> bytes:
    """
    Convert numpy array image to bytes for download.
    
    Args:
        image_array: Image as numpy array (grayscale)
        
    Returns:
        PNG image as bytes
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return buffer.getvalue()


# ============================================================================
# ALGORITHM INFORMATION
# ============================================================================

ALGORITHM_INFO = {
    'Sobel': {
        'name': 'Sobel Operator',
        'description': 'Operator Sobel menggunakan dua kernel 3x3 untuk menghitung pendekatan gradien turunan intensitas gambar. Operator ini memberikan bobot lebih pada piksel di tengah (bobot 2), sehingga sedikit lebih tahan terhadap noise dibandingkan operator sederhana lainnya. Sangat baik untuk mendeteksi tepi vertikal dan horizontal.',
        'kernel_x': '[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]',
        'kernel_y': '[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]'
    },
    'Roberts': {
        'name': 'Roberts Cross Operator',
        'description': 'Operator Roberts Cross adalah salah satu algoritma deteksi tepi paling awal. Menggunakan kernel 2x2 kecil yang diputar 45 derajat. Sangat cepat dikomputasi tetapi sangat sensitif terhadap noise karena ukuran kernelnya yang kecil. Cocok untuk gambar dengan noise rendah dan tepi yang tajam.',
        'kernel_x': '[[1, 0], [0, -1]]',
        'kernel_y': '[[0, 1], [-1, 0]]'
    },
    'Prewitt': {
        'name': 'Prewitt Operator',
        'description': 'Operator Prewitt mirip dengan Sobel tetapi menggunakan bobot seragam (1) pada kernelnya. Ini membuatnya sedikit kurang sensitif terhadap noise dibandingkan Roberts, tetapi deteksi tepinya mungkin tidak setajam Sobel. Sering digunakan untuk mendeteksi tepi vertikal dan horizontal.',
        'kernel_x': '[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]',
        'kernel_y': '[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]'
    },
    'Laplacian': {
        'name': 'Laplacian Operator',
        'description': 'Operator Laplacian menggunakan turunan kedua dari intensitas gambar. Berbeda dengan operator lain yang mencari nilai maksimum gradien, Laplacian mencari "zero-crossings" (titik di mana turunan kedua bernilai nol). Ini mendeteksi tepi ke segala arah sekaligus, tetapi sangat sensitif terhadap noise.',
        'kernel': '[[0, 1, 0], [1, -4, 1], [0, 1, 0]]'
    },
    'Frei-Chen': {
        'name': 'Frei-Chen Operator',
        'description': 'Operator Frei-Chen mirip dengan Sobel tetapi menggunakan akar 2 (√2) sebagai bobot di tengah, bukan 2. Bobot ini dipilih untuk memberikan respons yang lebih seragam (isotropik) terhadap tepi di segala arah sudut, sehingga deteksi tepi lebih halus dan akurat secara geometris.',
        'kernel_x': '[[-1, 0, 1], [-√2, 0, √2], [-1, 0, 1]]',
        'kernel_y': '[[-1, -√2, -1], [0, 0, 0], [1, √2, 1]]'
    },
    'Canny': {
        'name': 'Canny Edge Detector',
        'description': 'Algoritma Canny dianggap sebagai standar emas dalam deteksi tepi. Ini adalah metode multi-tahap: (1) Reduksi noise dengan Gaussian, (2) Perhitungan gradien, (3) Non-maximum suppression untuk menipiskan tepi, dan (4) Hysteresis thresholding untuk menyambungkan tepi yang lemah ke tepi yang kuat. Menghasilkan garis tepi yang tipis dan bersih.',
        'parameters': 'Low Threshold, High Threshold'
    },
    'Dilation': {
        'name': 'Dilation (Dilasi)',
        'description': 'Operasi morfologi yang menambah piksel pada batas objek dalam gambar. Efeknya adalah memperbesar area objek yang terang dan mengecilkan area gelap (lubang). Berguna untuk menyambungkan bagian objek yang terputus.',
        'parameters': 'Kernel Size'
    },
    'Erosion': {
        'name': 'Erosion (Erosi)',
        'description': 'Kebalikan dari dilasi, erosi mengikis piksel pada batas objek. Efeknya adalah memperkecil area objek yang terang dan memperbesar lubang. Berguna untuk menghilangkan noise kecil (bintik putih) di latar belakang gelap.',
        'parameters': 'Kernel Size'
    },
    'Opening': {
        'name': 'Opening',
        'description': 'Kombinasi erosi diikuti oleh dilasi. Berguna untuk menghilangkan noise kecil (bintik terang) dari latar belakang gelap sambil tetap mempertahankan bentuk dan ukuran objek utama.',
        'parameters': 'Kernel Size'
    },
    'Closing': {
        'name': 'Closing',
        'description': 'Kombinasi dilasi diikuti oleh erosi. Berguna untuk menutup lubang kecil di dalam objek terang atau menyambungkan komponen yang berdekatan.',
        'parameters': 'Kernel Size'
    },
    'Morphological Gradient': {
        'name': 'Morphological Gradient',
        'description': 'Perbedaan antara hasil dilasi dan erosi dari sebuah gambar. Hasilnya tampak seperti outline atau tepi dari objek. Sangat berguna untuk deteksi tepi morfologis.',
        'parameters': 'Kernel Size'
    },
    'Region Filling': {
        'name': 'Region Filling (Hole Filling)',
        'description': 'Algoritma untuk mengisi lubang tertutup di dalam objek. Bekerja dengan mengidentifikasi area gelap yang dikelilingi sepenuhnya oleh area terang dan mengisinya menjadi terang.',
        'parameters': 'None'
    }
}


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def render_controls(key_suffix=""):
    """
    Render the control panel widgets in the main area.
    Returns a dictionary of user inputs.
    """
    # File uploader
    st.markdown("### 1. Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'png', 'jpeg'],
        help="Supported formats: JPG, PNG, JPEG",
        key=f"uploader_{key_suffix}"
    )
    
    st.markdown("---")
    
    # Algorithm selection
    st.markdown("### 2. Select Algorithm")
    
    col_algo, col_params = st.columns([1, 1])
    
    with col_algo:
        algorithm = st.radio(
            "Choose Operator:",
            options=['Sobel', 'Roberts', 'Prewitt', 'Laplacian', 'Frei-Chen', 'Canny', 
                     'Dilation', 'Erosion', 'Opening', 'Closing', 'Morphological Gradient', 'Region Filling'],
            help="Select the edge detection or morphological algorithm to apply",
            key=f"algo_{key_suffix}",
            horizontal=True
        )
    
    # Parameters
    canny_low, canny_high = 50, 150
    kernel_size = 3
    
    with col_params:
        if algorithm == 'Canny':
            st.markdown("**Canny Parameters**")
            c1, c2 = st.columns(2)
            with c1:
                canny_low = st.slider(
                    "Low Threshold",
                    min_value=0,
                    max_value=255,
                    value=50,
                    key=f"low_{key_suffix}"
                )
            with c2:
                canny_high = st.slider(
                    "High Threshold",
                    min_value=0,
                    max_value=255,
                    value=150,
                    key=f"high_{key_suffix}"
                )
        elif algorithm in ['Dilation', 'Erosion', 'Opening', 'Closing', 'Morphological Gradient']:
            st.markdown("**Morphology Parameters**")
            
            # Show info about selected algorithm
            info = ALGORITHM_INFO[algorithm]
            st.markdown(f"""
            <div style="background-color: rgba(5, 150, 105, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(5, 150, 105, 0.2);">
                <p style="margin: 0; font-size: 0.9rem; color: inherit;"><strong>{info['name']}</strong>: {info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            kernel_size = st.slider(
                "Kernel Size",
                min_value=3,
                max_value=21,
                value=3,
                step=2,
                help="Size of the structuring element (odd number)",
                key=f"kernel_{key_suffix}"
            )
        else:
            # Show info about selected algorithm
            info = ALGORITHM_INFO[algorithm]
            st.markdown(f"""
            <div style="background-color: rgba(5, 150, 105, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid rgba(5, 150, 105, 0.2);">
                <p style="margin: 0; font-size: 0.9rem; color: inherit;"><strong>{info['name']}</strong>: {info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Process button
    detect_btn = st.button("Detect Edges / Apply Filter", key=f"btn_{key_suffix}")
    
    return {
        "uploaded_file": uploaded_file,
        "algorithm": algorithm,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "kernel_size": kernel_size,
        "detect_btn": detect_btn
    }

def main():
    """Main function to run the Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Edge Detection",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state for theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # Inject custom CSS based on current theme
    inject_custom_css(st.session_state.theme)
    
    # ========================================================================
    # HEADER & THEME TOGGLE
    # ========================================================================
    
    col_title, col_toggle = st.columns([6, 1])
    with col_title:
        st.markdown('<h1 class="main-title" style="margin-top:0;">Digital Image <span>Edge Detection</span></h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Turn your images into edge maps in seconds.</p>', unsafe_allow_html=True)
    
    with col_toggle:
        # Theme Toggle
        if st.button(f"{'☀️' if st.session_state.theme == 'dark' else '🌙'} Theme", key="theme_toggle"):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
            
    st.markdown("---")

    # ========================================================================
    # CONTROLS (PRIMARY SECTION)
    # ========================================================================
    
    controls = render_controls("main")
    
    uploaded_file = controls['uploaded_file']
    algorithm = controls['algorithm']
    canny_low = controls['canny_low']
    canny_high = controls['canny_high']
    kernel_size = controls['kernel_size']
    detect_btn = controls['detect_btn']

    # ========================================================================
    # RESULTS AREA
    # ========================================================================

    # Check if image is uploaded
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Create two columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="image-card">
                <div class="image-card-title">Original Image</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            
            # Display image info
            img_array = np.array(image)
            st.caption(f"Size: {img_array.shape[1]} × {img_array.shape[0]} pixels")
        
        # Process and display result
        if detect_btn or 'result_image' in st.session_state:
            if detect_btn:
                with st.spinner("Processing..."):
                    result_image = process_image(image, algorithm, canny_low, canny_high, kernel_size)
                    st.session_state['result_image'] = result_image
                    st.session_state['algorithm_used'] = algorithm
            
            with col2:
                st.markdown(f"""
                <div class="image-card">
                    <div class="image-card-title">Result ({st.session_state.get('algorithm_used', algorithm)})</div>
                </div>
                """, unsafe_allow_html=True)
                
                if 'result_image' in st.session_state:
                    st.image(st.session_state['result_image'], use_column_width=True)
                    
                    # Download button
                    result_bytes = convert_image_for_download(st.session_state['result_image'])
                    st.download_button(
                        label="Download Result",
                        data=result_bytes,
                        file_name=f"processed_{st.session_state.get('algorithm_used', algorithm).lower()}.png",
                        mime="image/png"
                    )
                    
                    st.markdown('<div class="success-msg">Processing completed successfully.</div>', 
                               unsafe_allow_html=True)
        else:
            with col2:
                st.markdown("""
                <div class="image-card">
                    <div class="image-card-title">Result</div>
                </div>
                """, unsafe_allow_html=True)
                st.info("Click 'Detect Edges / Apply Filter' button above to process the image.")
    
    else:
        # Show instruction when no image is uploaded
        st.info("Please upload an image to start.")
        
        # Show available algorithms info
        with st.expander("Learn about the algorithms"):
            for algo, info in ALGORITHM_INFO.items():
                st.markdown(f"**{info['name']}**")
                st.markdown(info['description'])
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.85rem; padding: 1rem 0;" class="dynamic-subtext">
        <p>Digital Image Processing - Edge Detection Application</p>
        <p>Built with Streamlit • OpenCV • NumPy</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
