import streamlit as st

def inject_custom_css():
    """
    Inject custom CSS to style the Streamlit application.
    """
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global settings */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* File uploader styling */
        [data-testid="stFileUploader"] {
            border-radius: 12px;
            padding: 1rem;
        }
        
        /* Slider styling */
        .stSlider > div > div {
            /* Default streamlit slider */
        }
        
        /* Info box styling */
        .stAlert {
            border-radius: 8px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            border-radius: 8px;
        }
        
        /* Custom headers */
        h1, h2, h3 {
            font-weight: 700 !important;
        }
    </style>
    """, unsafe_allow_html=True)
