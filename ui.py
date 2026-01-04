import streamlit as st
from utils import ALGORITHM_INFO

def render_controls(container=st):
    """
    Render controls and return selected parameters.
    """
    container.header("⚙️ Pengaturan")
    
    # Algorithm selection
    algorithm = container.selectbox(
        "Pilih Algoritma",
        list(ALGORITHM_INFO.keys()),
        index=0
    )
    
    # Display algorithm description
    container.info(f"**{ALGORITHM_INFO[algorithm]['name']}**\n\n{ALGORITHM_INFO[algorithm]['description']}")
    
    # Parameters dictionary to return
    params = {'algorithm': algorithm}
    
    # Additional parameters for specific algorithms
    if algorithm == 'Canny':
        container.subheader("Parameter Canny")
        params['canny_low'] = container.slider(
            "Low Threshold", 
            min_value=0, 
            max_value=255, 
            value=50,
            help="Batas bawah untuk hysteresis thresholding. Tepi dengan gradien di bawah ini akan dibuang."
        )
        params['canny_high'] = container.slider(
            "High Threshold", 
            min_value=0, 
            max_value=255, 
            value=150,
            help="Batas atas untuk hysteresis thresholding. Tepi dengan gradien di atas ini akan dianggap sebagai tepi kuat."
        )
    elif algorithm in ['Dilation', 'Erosion', 'Opening', 'Closing', 'Morphological Gradient']:
        container.subheader(f"Parameter {algorithm}")
        params['kernel_size'] = container.slider(
            "Kernel Size",
            min_value=3,
            max_value=21,
            value=3,
            step=2,
            help="Ukuran kernel (ganjil). Semakin besar nilai, semakin kuat efeknya."
        )
        
    return params
