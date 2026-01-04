import streamlit as st
from PIL import Image
import numpy as np
from styles import inject_custom_css
from ui import render_controls
from utils import process_image, convert_image_for_download

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Aplikasi Deteksi Tepi & Morfologi",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Main header
    st.title("🖼️ Analisis Citra Digital: Deteksi Tepi & Morfologi")
    st.markdown("""
    Aplikasi ini mendemonstrasikan berbagai algoritma **Deteksi Tepi (Edge Detection)** dan **Operasi Morfologi** 
    yang umum digunakan dalam Pengolahan Citra Digital. Unggah gambar Anda dan eksperimen dengan berbagai operator!
    """)
    
    # Render sidebar controls
    params = render_controls()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    # File uploader
    with col1:
        st.subheader("1. Unggah Gambar")
        uploaded_file = st.file_uploader("Pilih gambar (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Open and display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Asli", use_column_width=True)
            
            # Image info
            st.info(f"Dimensi: {image.size[0]}x{image.size[1]} piksel | Mode: {image.mode}")
    
    # Process image if uploaded
    with col2:
        st.subheader("2. Hasil Pemrosesan")
        
        if uploaded_file is not None:
            if st.button("Proses Gambar", type="primary"):
                with st.spinner(f"Menerapkan algoritma {params['algorithm']}..."):
                    try:
                        # Extract parameters based on selected algorithm
                        canny_low = params.get('canny_low', 50)
                        canny_high = params.get('canny_high', 150)
                        kernel_size = params.get('kernel_size', 3)
                        
                        # Process the image
                        result_image = process_image(
                            image, 
                            params['algorithm'], 
                            canny_low, 
                            canny_high,
                            kernel_size
                        )
                        
                        # Display result
                        st.image(result_image, caption=f"Hasil: {params['algorithm']}", use_column_width=True)
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Unduh Hasil",
                            data=convert_image_for_download(result_image),
                            file_name=f"hasil_{params['algorithm'].lower().replace(' ', '_')}.png",
                            mime="image/png"
                        )
                        
                        st.success("Pemrosesan selesai!")
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")
            else:
                st.info("Klik tombol 'Proses Gambar' untuk melihat hasil.")
        else:
            st.warning("Silakan unggah gambar terlebih dahulu di panel sebelah kiri.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Dibuat untuk Praktikum Pengolahan Citra Digital</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
