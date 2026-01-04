import streamlit as st
from utils import ALGORITHM_INFO

def render_controls():
    st.sidebar.header("Pengaturan Algoritma")
    
    # Group algorithms by category
    edge_detection_algos = ['Sobel', 'Roberts', 'Prewitt', 'Laplacian', 'Frei-Chen', 'Canny']
    morphology_algos = ['Dilation', 'Erosion', 'Opening', 'Closing', 'Morphological Gradient', 'Region Filling']
    
    category = st.sidebar.radio("Pilih Kategori:", ["Deteksi Tepi", "Morfologi"])
    
    if category == "Deteksi Tepi":
        algorithm = st.sidebar.selectbox("Pilih Algoritma:", edge_detection_algos)
    else:
        algorithm = st.sidebar.selectbox("Pilih Operasi:", morphology_algos)
    
    # Display algorithm description
    if algorithm in ALGORITHM_INFO:
        st.sidebar.info(ALGORITHM_INFO[algorithm]['description'])
        if 'kernel_x' in ALGORITHM_INFO[algorithm]:
             with st.sidebar.expander("Lihat Kernel"):
                st.code(f"Kernel X:\n{ALGORITHM_INFO[algorithm]['kernel_x']}\n\nKernel Y:\n{ALGORITHM_INFO[algorithm]['kernel_y']}")
        elif 'kernel' in ALGORITHM_INFO[algorithm]:
             with st.sidebar.expander("Lihat Kernel"):
                st.code(f"Kernel:\n{ALGORITHM_INFO[algorithm]['kernel']}")

    # Dynamic parameters based on selection
    canny_low = 50
    canny_high = 150
    kernel_size = 3
    
    if algorithm == 'Canny':
        st.sidebar.subheader("Parameter Canny")
        canny_low = st.sidebar.slider("Low Threshold", 0, 255, 50)
        canny_high = st.sidebar.slider("High Threshold", 0, 255, 150)
    elif algorithm in morphology_algos and algorithm != 'Region Filling':
        st.sidebar.subheader("Parameter Morfologi")
        kernel_size = st.sidebar.slider("Ukuran Kernel (Ganjil)", 3, 21, 3, step=2)
        
    return algorithm, canny_low, canny_high, kernel_size
