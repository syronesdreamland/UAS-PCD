import streamlit as st
from utils import ALGORITHM_INFO

def render_controls():
    """
    Render sidebar controls and return selected parameters.
    """
    st.sidebar.header("⚙️ Pengaturan")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Pilih Algoritma",
        list(ALGORITHM_INFO.keys()),
        index=0
    )
    
    # Display algorithm description
    st.sidebar.info(f"**{ALGORITHM_INFO[algorithm]['name']}**\n\n{ALGORITHM_INFO[algorithm]['description']}")
    
    # Parameters dictionary to return
    params = {'algorithm': algorithm}
    
    # Additional parameters for specific algorithms
    if algorithm == 'Canny':
        st.sidebar.subheader("Parameter Canny")
        params['canny_low'] = st.sidebar.slider(
            "Low Threshold", 
            min_value=0, 
            max_value=255, 
            value=50,
            help="Batas bawah untuk hysteresis thresholding. Tepi dengan gradien di bawah ini akan dibuang."
        )
        params['canny_high'] = st.sidebar.slider(
            "High Threshold", 
            min_value=0, 
            max_value=255, 
            value=150,
            help="Batas atas untuk hysteresis thresholding. Tepi dengan gradien di atas ini akan dianggap sebagai tepi kuat."
        )
    elif algorithm in ['Dilation', 'Erosion', 'Opening', 'Closing', 'Morphological Gradient']:
        st.sidebar.subheader(f"Parameter {algorithm}")
        params['kernel_size'] = st.sidebar.slider(
            "Kernel Size",
            min_value=3,
            max_value=21,
            value=3,
            step=2,
            help="Ukuran kernel (ganjil). Semakin besar nilai, semakin kuat efeknya."
        )
        
    return params
