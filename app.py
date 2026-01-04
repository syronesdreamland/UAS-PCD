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

def inject_custom_css():
    """
    Inject custom CSS to style the Streamlit application.
    Removes default header/footer and applies modern dark theme styling.
    """
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global settings */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }

        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container styling */
        .stApp {
            background-color: #101622;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0d121c;
            border-right: 1px solid #282e39;
        }
        
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        
        /* Title styling */
        .main-title {
            color: #ffffff;
            font-size: 3rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }
        
        .main-title span {
            color: #135bec;
        }
        
        .subtitle {
            color: #9da6b9;
            text-align: center;
            font-size: 1.125rem;
            margin-bottom: 3rem;
            font-weight: 400;
        }
        
        /* Card styling for image containers */
        .image-card {
            background: #1c1f27;
            border: 1px solid #282e39;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .image-card-title {
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            background-color: #135bec;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(19, 91, 236, 0.1), 0 2px 4px -1px rgba(19, 91, 236, 0.06);
        }
        
        .stButton > button:hover {
            background-color: #1d4ed8;
            box-shadow: 0 10px 15px -3px rgba(19, 91, 236, 0.2), 0 4px 6px -2px rgba(19, 91, 236, 0.1);
            transform: translateY(-1px);
        }
        
        /* Download button styling */
        .stDownloadButton > button {
            width: 100%;
            background-color: #1c1f27;
            color: #ffffff;
            border: 1px solid #282e39;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.2s ease;
        }
        
        .stDownloadButton > button:hover {
            background-color: #282e39;
            border-color: #3b4354;
            color: #ffffff;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background: transparent;
            padding: 0;
        }
        
        .stRadio > div > label {
            color: #ffffff !important;
        }
        
        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background: #1c1f27;
            border: 1px dashed #282e39;
            border-radius: 12px;
            padding: 2rem;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #135bec;
            background: #1c1f27;
        }
        
        /* Slider styling */
        .stSlider > div > div {
            background: #135bec;
        }
        
        /* Info box styling */
        .stAlert {
            background: #1c1f27;
            border: 1px solid #282e39;
            border-radius: 8px;
            color: #9da6b9;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: #1c1f27;
            border-radius: 8px;
            border: 1px solid #282e39;
            color: #ffffff;
        }
        
        /* Sidebar header */
        .sidebar-header {
            color: #ffffff;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #282e39;
        }
        
        /* Algorithm info card */
        .algo-info {
            background: #1c1f27;
            border: 1px solid #282e39;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        .algo-info h4 {
            color: #ffffff;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .algo-info p {
            color: #9da6b9;
            font-size: 0.875rem;
            line-height: 1.5;
        }
        
        /* Success message */
        .success-msg {
            background: rgba(5, 150, 105, 0.1);
            color: #34d399;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 1px solid rgba(5, 150, 105, 0.2);
            text-align: center;
            margin: 1rem 0;
            font-size: 0.875rem;
        }
        
        /* Metric styling */
        [data-testid="stMetric"] {
            background: #1c1f27;
            border: 1px solid #282e39;
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Custom headers */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        p, label {
            color: #9da6b9 !important;
        }
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


def process_image(image: Image.Image, algorithm: str, 
                  canny_low: int = 50, canny_high: int = 150) -> np.ndarray:
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
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Dictionary mapping algorithm names to their functions
    algorithm_map = {
        'Sobel': apply_sobel_edge_detection,
        'Roberts': apply_roberts_edge_detection,
        'Prewitt': apply_prewitt_edge_detection,
        'Laplacian': apply_laplacian_edge_detection,
        'Frei-Chen': apply_frei_chen_edge_detection,
    }
    
    # Apply the selected algorithm
    if algorithm == 'Canny':
        output = apply_canny_edge_detection(gray, canny_low, canny_high)
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
        'description': 'Uses 3×3 kernels to compute gradient approximation. Emphasizes center pixels with weight 2. Good for detecting horizontal and vertical edges.',
        'kernel_x': '[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]',
        'kernel_y': '[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]'
    },
    'Roberts': {
        'name': 'Roberts Cross Operator',
        'description': 'Uses 2×2 kernels for diagonal edge detection. One of the earliest edge detectors. Simple but sensitive to noise.',
        'kernel_x': '[[1, 0], [0, -1]]',
        'kernel_y': '[[0, 1], [-1, 0]]'
    },
    'Prewitt': {
        'name': 'Prewitt Operator',
        'description': 'Uses 3×3 kernels with uniform weights. Less sensitive to noise than Roberts but also less accurate for edge localization.',
        'kernel_x': '[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]',
        'kernel_y': '[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]'
    },
    'Laplacian': {
        'name': 'Laplacian Operator',
        'description': 'Second-order derivative operator. Detects edges in all directions by finding zero-crossings. Sensitive to noise.',
        'kernel': '[[0, 1, 0], [1, -4, 1], [0, 1, 0]]'
    },
    'Frei-Chen': {
        'name': 'Frei-Chen Operator',
        'description': 'Similar to Sobel but uses √2 for better isotropy. Provides more uniform response to edges at different orientations.',
        'kernel_x': '[[-1, 0, 1], [-√2, 0, √2], [-1, 0, 1]]',
        'kernel_y': '[[-1, -√2, -1], [0, 0, 0], [1, √2, 1]]'
    },
    'Canny': {
        'name': 'Canny Edge Detector',
        'description': 'Multi-stage algorithm: Gaussian smoothing → Gradient calculation → Non-max suppression → Double thresholding → Edge tracking by hysteresis.',
        'parameters': 'Low Threshold, High Threshold'
    }
}


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main function to run the Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Edge Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # ========================================================================
    # SIDEBAR - Control Panel
    # ========================================================================
    
    with st.sidebar:
        st.markdown('<p class="sidebar-header">Control Panel</p>', unsafe_allow_html=True)
        
        # File uploader
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'png', 'jpeg'],
            help="Supported formats: JPG, PNG, JPEG"
        )
        
        st.markdown("---")
        
        # Algorithm selection
        st.markdown("### Edge Detection Algorithm")
        algorithm = st.radio(
            "Choose Operator:",
            options=['Sobel', 'Roberts', 'Prewitt', 'Laplacian', 'Frei-Chen', 'Canny'],
            help="Select the edge detection algorithm to apply"
        )
        
        # Canny-specific parameters
        if algorithm == 'Canny':
            st.markdown("---")
            st.markdown("### Canny Parameters")
            canny_low = st.slider(
                "Low Threshold",
                min_value=0,
                max_value=255,
                value=50,
                help="Edges with gradient below this are discarded"
            )
            canny_high = st.slider(
                "High Threshold",
                min_value=0,
                max_value=255,
                value=150,
                help="Edges with gradient above this are strong edges"
            )
        else:
            canny_low, canny_high = 50, 150
        
        st.markdown("---")
        
        # Process button
        detect_btn = st.button("Detect Edges", use_container_width=True)
        
        # Algorithm info expander
        st.markdown("---")
        with st.expander("Algorithm Information"):
            info = ALGORITHM_INFO[algorithm]
            st.markdown(f"**{info['name']}**")
            st.markdown(info['description'])
            if 'kernel_x' in info:
                st.code(f"Kernel X: {info['kernel_x']}")
                st.code(f"Kernel Y: {info['kernel_y']}")
            elif 'kernel' in info:
                st.code(f"Kernel: {info['kernel']}")
            elif 'parameters' in info:
                st.info(f"Parameters: {info['parameters']}")
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Title
    st.markdown('<h1 class="main-title">Digital Image <span>Edge Detection</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Turn your images into edge maps in seconds.</p>', unsafe_allow_html=True)
    
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
            st.image(image, use_container_width=True)
            
            # Display image info
            img_array = np.array(image)
            st.caption(f"Size: {img_array.shape[1]} × {img_array.shape[0]} pixels")
        
        # Process and display result
        if detect_btn or 'result_image' in st.session_state:
            if detect_btn:
                with st.spinner("Processing..."):
                    result_image = process_image(image, algorithm, canny_low, canny_high)
                    st.session_state['result_image'] = result_image
                    st.session_state['algorithm_used'] = algorithm
            
            with col2:
                st.markdown(f"""
                <div class="image-card">
                    <div class="image-card-title">Result ({st.session_state.get('algorithm_used', algorithm)})</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(st.session_state['result_image'], use_container_width=True)
                
                # Download button
                result_bytes = convert_image_for_download(st.session_state['result_image'])
                st.download_button(
                    label="Download Result",
                    data=result_bytes,
                    file_name=f"edge_detection_{st.session_state.get('algorithm_used', algorithm).lower()}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                st.markdown('<div class="success-msg">Edge detection completed successfully.</div>', 
                           unsafe_allow_html=True)
        else:
            with col2:
                st.markdown("""
                <div class="image-card">
                    <div class="image-card-title">Result</div>
                </div>
                """, unsafe_allow_html=True)
                st.info("Click 'Detect Edges' button in the sidebar to process the image.")
    
    else:
        # Show instruction when no image is uploaded
        st.markdown("""
        <div style="
            background: #1c1f27;
            border: 1px dashed #282e39;
            border-radius: 16px;
            padding: 3rem;
            text-align: center;
            margin: 2rem 0;
        ">
            <h3 style="color: #ffffff; margin-bottom: 0.5rem;">Upload an Image to Begin</h3>
            <p style="color: #9da6b9;">
                Use the sidebar to upload a JPG, PNG, or JPEG image
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show available algorithms
        st.markdown("### Available Edge Detection Algorithms")
        
        algo_cols = st.columns(3)
        algorithms_list = list(ALGORITHM_INFO.keys())
        
        for idx, algo in enumerate(algorithms_list):
            with algo_cols[idx % 3]:
                info = ALGORITHM_INFO[algo]
                st.markdown(f"""
                <div style="
                    background: #1c1f27;
                    border: 1px solid #282e39;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    height: 180px;
                ">
                    <h4 style="color: #ffffff; margin-bottom: 0.5rem;">{info['name']}</h4>
                    <p style="color: #9da6b9; font-size: 0.85rem; line-height: 1.4;">
                        {info['description'][:150]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9da6b9; font-size: 0.85rem; padding: 1rem 0;">
        <p>Digital Image Processing - Edge Detection Application</p>
        <p>Built with Streamlit • OpenCV • NumPy</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
