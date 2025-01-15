import streamlit as st
import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk memuat model yang telah dilatih
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Konversi gambar ke Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Memuat model Gradient Boosting
gb_model = load_model("gradient_boosting_model.pkl")  # Model Gradient Boosting

# Pengaturan halaman
st.set_page_config(page_title="Prediksi Biaya Asuransi", page_icon="💼", layout="wide")

# Header aplikasi
header_image_base64 = get_base64_image("header_image.jpg")
st.markdown(
    f"""
    <div style='background-color: #f4f4f8; padding: 20px; text-align: center;'>
        <img src='data:image/jpeg;base64,{header_image_base64}' style='width: 40%; border-radius: 10px;'>
        <h1 style='font-family: Arial, sans-serif; color: #4CAF50;'>Prediksi Biaya Asuransi</h1>
        <p style='font-size: 18px; color: #555;'>Gunakan aplikasi ini untuk memprediksi biaya asuransi Anda dengan mudah dan cepat</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Fungsi untuk memproses input data
def preprocess_input(age, bmi, children, sex, smoker, region):
    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }])
    return input_data

# Kontainer form
st.markdown(
    """
    <style>
    .form-container {
        margin: 0 auto;
        max-width: 400px;
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        font-family: Arial, sans-serif;
        text-align: center;
    }
    .form-header {
        text-align: center;
        color: #333;
        margin-bottom: 20px;
    }
    </style>
    
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 class='form-header'>Masukkan Data Anda</h2>", unsafe_allow_html=True)

# Form input
form = st.form(key="insurance_form")
age = form.number_input("🧑 Umur:", min_value=18, max_value=100, value=30, step=1)
bmi = form.number_input("📊 BMI:", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = form.number_input("👶 Jumlah Anak:", min_value=0, max_value=10, value=2, step=1)
sex = form.selectbox("👫 Jenis Kelamin:", options=["male", "female"])
smoker = form.selectbox("🚬 Perokok:", options=["yes", "no"])
region = form.selectbox("📍 Wilayah:", options=["northeast", "northwest", "southeast", "southwest"])
submit = form.form_submit_button(" Prediksi Sekarang")

# Load dataset
data_path = "Regression.csv"  # Ganti dengan path dataset Anda
data = pd.read_csv(data_path)

# Proses prediksi jika tombol submit ditekan
if submit:
    input_data = preprocess_input(age, bmi, children, sex, smoker, region)
    gb_pred = gb_model.predict(input_data)[0]
    st.markdown(
        f"""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: #4CAF50;'>Hasil Prediksi</h3>
            <p style='font-size: 20px; color: #333;'>💰 Biaya Asuransi Anda diperkirakan sebesar:</p>
            <p style='font-size: 28px; color: #4CAF50;'><b>$ {gb_pred:,.2f}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tambahkan hasil prediksi ke dataset untuk visualisasi
    result_row = input_data.copy()
    result_row['charges'] = gb_pred
    result_row['source'] = 'Prediction'  # Kolom sumber data
    data['source'] = 'Dataset'
    data_with_prediction = pd.concat([data, result_row], ignore_index=True)

    # Tampilkan boxplot untuk semua kolom numerik
    st.markdown("### Analisis Data: Boxplot Berdasarkan Data Anda")
    for column in ['age', 'bmi', 'children', 'charges']:
        st.markdown(f"#### Distribusi {column.capitalize()}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=data_with_prediction, x='source', y=column, ax=ax)
        ax.set_title(f"Distribusi {column.capitalize()} berdasarkan Sumber Data")
        ax.set_xlabel("Sumber Data")
        ax.set_ylabel(column.capitalize())
        st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 10px;'>
        <p style='font-size: 14px; color: #777;'>
            Dikembangkan oleh <a href='#' style='color: #4CAF50;'>Tim Anda</a> • Hak Cipta &copy; 2025
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
