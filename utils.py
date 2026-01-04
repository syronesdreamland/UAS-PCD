import cv2
import numpy as np
from PIL import Image
import io

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
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def apply_sobel_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return output

def apply_roberts_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return output

def apply_prewitt_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return output

def apply_laplacian_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
    output = cv2.convertScaleAbs(laplacian)
    return output

def apply_frei_chen_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    sqrt2 = np.sqrt(2)
    kernel_x = np.array([[-1, 0, 1], [-sqrt2, 0, sqrt2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -sqrt2, -1], [0, 0, 0], [1, sqrt2, 1]], dtype=np.float64)
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    output = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return output

def apply_canny_edge_detection(gray_image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    output = cv2.Canny(gray_image, low_threshold, high_threshold)
    return output

def apply_dilation(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(gray_image, kernel, iterations=1)

def apply_erosion(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(gray_image, kernel, iterations=1)

def apply_opening(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

def apply_closing(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

def apply_morphological_gradient(gray_image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)

def apply_region_filling(gray_image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary[0, 0] == 255:
        binary = cv2.bitwise_not(binary)
    im_floodfill = binary.copy()
    h, w = binary.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = binary | im_floodfill_inv
    return im_out

def process_image(image: Image.Image, algorithm: str, canny_low: int = 50, canny_high: int = 150, kernel_size: int = 3) -> np.ndarray:
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    if algorithm not in ['Region Filling']:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
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
    
    if algorithm == 'Canny':
        output = apply_canny_edge_detection(gray, canny_low, canny_high)
    elif algorithm in morphology_map:
        output = morphology_map[algorithm](gray, kernel_size)
    elif algorithm in algorithm_map:
        output = algorithm_map[algorithm](gray)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def convert_image_for_download(image_array: np.ndarray) -> bytes:
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()
