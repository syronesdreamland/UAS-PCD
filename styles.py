import streamlit as st

def inject_custom_css():
    """
    Inject custom CSS to improve the UI appearance.
    """
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .main {
                background-color: #0e1117;
            }
        }
        
        /* Header styling */
        h1 {
            color: #1f77b4;
            text-align: center;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
            margin-bottom: 30px;
        }
        
        /* Card-like container for images */
        .css-1r6slb0 {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            font-weight: bold;
        }
        
        /* Info box styling */
        .stAlert {
            border-radius: 8px;
        }
        
        /* Fix text readability in dark/light mode */
        .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6, span, div {
            color: inherit !important; 
        }
        
        /* Ensure text is readable on light background */
        @media (prefers-color-scheme: light) {
            .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6, span, div {
                color: #31333F !important;
            }
            h1 {
                color: #1f77b4 !important;
            }
        }
        
        /* Ensure text is readable on dark background */
        @media (prefers-color-scheme: dark) {
            .stMarkdown, .stText, p, label, h1, h2, h3, h4, h5, h6, span, div {
                color: #FAFAFA !important;
            }
            h1 {
                color: #4da6ff !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
