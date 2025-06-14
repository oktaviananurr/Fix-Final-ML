import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import cv2
import tempfile
import os
import json
from glob import glob
from collections import Counter
from detect import run  # Assuming your detect.py and best.pt are in the correct path

import pathlib
import platform

# Patch agar WindowsPath dari model tidak error di Linux
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath


# --- Page Configuration ---
st.set_page_config(page_title="Deteksi Penyakit Kulit", page_icon="🔬", layout="centered")

# --- Environment Variable (Optional, good practice) ---
# os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false" # Keep if needed

# --- Particles.js HTML with Updated Text ---


# --- CSS Loading Function ---
def load_css(file_name):
    """Loads a CSS file and applies it using st.markdown."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        # st.write(f"DEBUG: Loaded CSS from {file_name}") # For debugging
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Default styles might be applied.")
    except Exception as e:
        st.error(f"Error loading CSS {file_name}: {e}")


# --- Load Treatment Information ---
@st.cache_data
def load_penanganan(path="treatment.json"):
    try:
        with open(path, "r", encoding="utf-8") as f: # Added encoding
            data = json.load(f)
        return {item["label"]: item["instructions"] for item in data}
    except FileNotFoundError:
        st.error(f"Error: '{path}' not found. Please ensure the treatment JSON file exists.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{path}'. Please check its format.")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading '{path}': {e}")
        return {}


penanganan_dict = load_penanganan()
class_names = ['biang_keringat', 'bisul', 'herpes', 'jerawat', 'kanker_melanoma', 'kurap', 'kutil', 'psoriasis', 'seborrheic_keratoses', 'urtikaria']

# --- Detection Helper Functions ---
def load_image_opencv(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def get_latest_exp_folder(base_path="runs/detect"):
    exp_folders = sorted(glob(os.path.join(base_path, "exp*")), key=os.path.getmtime)
    return exp_folders[-1] if exp_folders else None

def read_detection_labels(txt_path):
    labels = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            labels = [int(line.split()[0]) for line in f.readlines()]
    return labels

def run_detection_logic(file):
    image = load_image_opencv(file)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    result = {"filename": file.name, "labels": [], "image_path": None, "error": None}

    try:
        # Ensure the 'detect' script and 'best.pt' are correctly set up.
        # The 'project' argument defines where 'runs/detect/exp*' folders are created.
        # The 'name' argument defines the 'exp*' folder name. Using a generic name.
        run(weights='./best.pt', source=tmp_path, conf_thres=0.3, imgsz=(640, 640),
            save_txt=True, save_conf=True, save_crop=False, project='runs/detect', name='exp', exist_ok=True) # Added project, name, exist_ok

        latest_exp = get_latest_exp_folder() # This should now point to 'runs/detect/exp' or 'runs/detect/exp2' etc.

        if latest_exp:
            img_path = os.path.join(latest_exp, os.path.basename(tmp_path))
            txt_path = os.path.join(latest_exp, "labels", os.path.splitext(os.path.basename(tmp_path))[0] + ".txt")

            if not os.path.exists(img_path):
                # Fallback if image is directly in latest_exp (e.g. if source was a dir)
                possible_image_files = glob(os.path.join(latest_exp, os.path.splitext(os.path.basename(tmp_path))[0] + ".*"))
                if possible_image_files:
                    img_path = possible_image_files[0]
                else:
                    result["error"] = f"Detected image not found in {latest_exp} for {os.path.basename(tmp_path)}"
                    img_path = None # ensure it's None

            labels = read_detection_labels(txt_path)
            label_names = [class_names[c] for c in labels if c < len(class_names)] # Added safety check

            result.update({"labels": label_names, "image_path": img_path if os.path.exists(img_path) else None})
            if img_path and not os.path.exists(img_path):
                 result["error"] = result.get("error", "") + f" Image path {img_path} does not exist."

        else:
            result["error"] = "❌ Folder hasil deteksi (exp*) tidak ditemukan di 'runs/detect'."
    except Exception as e:
        st.error(f"⚠️ Deteksi gagal untuk {file.name}: {e}")
        result["error"] = str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result

# --- Main Application UI ---
st.header("Deteksi Penyakit Kulit")
gambar_header =  "https://images.app.goo.gl/1idLX4FqVgfAsAoY8"
st.image(gambar_header, use_column_width=True, width=50)

st.markdown("Berikut adalah penjelasan singkat mengenai beberapa jenis penyakit kulit yang dapat dideteksi:")

daftar_penyakit = [
    'biang_keringat', 'bisul', 'herpes', 'jerawat', 'kanker_melanoma',
    'kurap', 'kutil', 'psoriasis', 'seborrheic_keratoses', 'urtikaria'
]

deskripsi_penyakit = {
    'biang_keringat': "Biang keringat (miliaria) adalah ruam kecil kemerahan dan menonjol yang terasa gatal dan terkadang perih. Kondisi ini disebabkan oleh penyumbatan kelenjar keringat, sehingga keringat tidak bisa keluar ke permukaan kulit.",
    'bisul': "Bisul (furunkel) adalah benjolan merah pada kulit yang berisi nanah dan terasa nyeri. Kondisi ini disebabkan oleh infeksi bakteri Staphylococcus aureus pada folikel rambut.",
    'herpes': "Herpes kulit bisa merujuk pada beberapa kondisi, umumnya herpes zoster (cacar ular) atau herpes simpleks. Herpes zoster disebabkan oleh reaktivasi virus Varicella-zoster (penyebab cacar air) dan ditandai ruam lepuh yang nyeri pada satu sisi tubuh. Herpes simpleks menyebabkan luka lepuh di sekitar mulut (HSV-1) atau alat kelamin (HSV-2).",
    'jerawat': "Jerawat (acne vulgaris) adalah masalah kulit yang terjadi ketika folikel rambut tersumbat oleh minyak dan sel kulit mati. Kondisi ini sering ditandai dengan munculnya komedo, bintik merah meradang, hingga benjolan berisi nanah.",
    'kanker_melanoma': "Melanoma adalah jenis kanker kulit yang paling berbahaya. Kanker ini berkembang pada melanosit, sel penghasil melanin (pigmen pemberi warna kulit). Tanda umum melanoma adalah munculnya tahi lalat baru atau perubahan pada tahi lalat yang sudah ada, seperti bentuk, ukuran, atau warna yang tidak biasa (asimetris, batas tidak rata, warna beragam, diameter lebih dari 6mm, dan evolusi/perubahan).",
    'kurap': "Kurap (tinea corporis) adalah infeksi jamur pada kulit yang menyebabkan ruam melingkar kemerahan, bersisik, dan terasa gatal. Bagian tengah ruam seringkali terlihat lebih jernih.",
    'kutil': "Kutil (verruca) adalah pertumbuhan kecil jinak pada kulit yang disebabkan oleh infeksi Human Papillomavirus (HPV). Bentuknya bisa beragam, dari datar hingga menonjol seperti kembang kol, dan bisa muncul di berbagai bagian tubuh.",
    'psoriasis': "Psoriasis adalah penyakit autoimun kronis yang menyebabkan peradangan pada kulit. Gejalanya berupa bercak merah tebal bersisik keperakan, gatal, dan terkadang nyeri. Psoriasis dapat muncul di bagian tubuh mana pun, termasuk kulit kepala, siku, dan lutut.",
    'seborrheic_keratoses': "Keratosis seboroik adalah pertumbuhan kulit non-kanker (jinak) yang umum terjadi seiring bertambahnya usia. Lesinya bisa tampak seperti kutil, berwarna cokelat, hitam, atau terang, dengan permukaan kasar atau licin, dan seolah 'menempel' di kulit.",
    'urtikaria': "Urtikaria (biduran atau kaligata) adalah reaksi kulit yang ditandai dengan munculnya bentol-bentol kemerahan atau pucat yang menonjol (bidur) dan terasa sangat gatal. Kondisi ini bisa disebabkan oleh alergi, infeksi, stres, atau faktor lainnya."
}

for penyakit in daftar_penyakit:
    nama_penyakit_tampil = penyakit.replace("_", " ").title()
    with st.expander(f"**{nama_penyakit_tampil}**"):
        st.write(deskripsi_penyakit.get(penyakit, "Deskripsi belum tersedia."))

st.markdown("---")
st.info("⚠️ **Penting**: Informasi di atas hanya bersifat umum dan tidak menggantikan diagnosis atau saran medis profesional. Jika Anda memiliki keluhan kulit, segera konsultasikan dengan dokter atau ahli dermatologi.")

st.markdown("<div style='margin-top: 12rem;'></div>", unsafe_allow_html=True)


# Initialize session state for detection results if not already present
if 'detection_results' not in st.session_state:
    st.session_state['detection_results'] = []
if 'uploaded_file_key' not in st.session_state: # Use a different key for the file_uploader if needed
    st.session_state['uploaded_file_key'] = 0


# File uploader
uploaded_file = st.file_uploader("Upload Gambar Penyakit Kulit", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state['uploaded_file_key']}")

# Buttons in columns
col1, col2 = st.columns(2)
with col1:
    detect_btn = st.button("Mulai Deteksi", use_container_width=True)
with col2:
    clear_btn = st.button("Hapus Ini", use_container_width=True)

if detect_btn:
    if uploaded_file is None:
        st.warning("⚠️ Harap upload gambar terlebih dahulu.")
    else:
        st.session_state['detection_results'] = []  # Reset previous results for a new detection
        with st.spinner("🔬 Menganalisis gambar... Mohon tunggu sebentar..."):
            result = run_detection_logic(uploaded_file)
            st.session_state['detection_results'].append(result)
            # No rerun needed here, results will display below

if clear_btn:
    st.session_state['detection_results'] = []
    st.session_state['uploaded_file_key'] += 1 # Increment key to reset file uploader
    # st.experimental_rerun() # Use st.rerun() for newer Streamlit versions
    st.rerun()


# --- Display Detection Results ---
if st.session_state['detection_results']:
    for result in st.session_state['detection_results']:
        st.markdown(f"--- \n ### Hasil Deteksi untuk: {result['filename']}")

        if result.get("error"):
            st.error(f"Terjadi kesalahan: {result['error']}")

        if result['image_path'] and os.path.exists(result["image_path"]):
            st.image(result["image_path"], caption="Gambar dengan Deteksi", use_container_width=True)
        elif uploaded_file and not result.get("error"): # If no processed image but no error, show original
            st.image(uploaded_file, caption="Gambar Asli (Tidak ada output deteksi visual)", use_container_width=True)


        if result['labels']:
            counts = Counter(result["labels"])
            st.markdown("--- \n #### ✅ Penyakit Kulit yang Terdeteksi:")
            for label, count in counts.items():
                display_label = label.replace("_", " ").title()
                st.success(f"**{display_label}**") # Removed count if only one instance is primary

            st.markdown("--- \n #### 📝 Saran Penanganan:")
            displayed_treatments = set()
            for label in result["labels"]: # Iterate in order of detection if relevant
                if label not in displayed_treatments:
                    treatment_steps = penanganan_dict.get(label)
                    display_label = label.replace("_", " ").title()
                    if treatment_steps:
                        with st.expander(f"**Penanganan untuk {display_label}**", expanded=False):
                            for i, step in enumerate(treatment_steps, 1):
                                st.markdown(f"{i}. {step}")
                    else:
                        st.info(f"Tidak ada informasi penanganan spesifik untuk {display_label} dalam data kami.")
                    displayed_treatments.add(label)
        elif not result.get("error"): # No labels and no error
            st.info("ℹ️ Tidak ada jenis penyakit kulit yang terdeteksi pada gambar ini atau deteksi di bawah ambang batas.")

elif not uploaded_file: # Initial state or after clearing
    st.info("Selamat datang! Silakan upload gambar kulit yang ingin diperiksa dan klik 'Jalankan Deteksi'.")