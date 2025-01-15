import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk memuat model yang telah dilatih
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Memuat model Gradient Boosting
gb_model = load_model("gradient_boosting_model.pkl")  # Model Gradient Boosting

# Streamlit UI
st.title("Aplikasi Prediksi Biaya Asuransi")
st.write("Gunakan aplikasi ini untuk memprediksi biaya asuransi berdasarkan data input Anda.")

# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Data")
age = st.sidebar.number_input("Umur:", min_value=18, max_value=100, value=30, step=1)
bmi = st.sidebar.number_input("BMI:", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.sidebar.number_input("Jumlah Anak:", min_value=0, max_value=10, value=2, step=1)
sex = st.sidebar.selectbox("Jenis Kelamin:", options=["male", "female"])
smoker = st.sidebar.selectbox("Perokok:", options=["yes", "no"])
region = st.sidebar.selectbox("Wilayah:", options=["northeast", "northwest", "southeast", "southwest"])

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

# Tombol prediksi
if st.sidebar.button("Prediksi"):
    # Preprocess input data
    input_data = preprocess_input(age, bmi, children, sex, smoker, region)

    # Prediksi
    gb_pred = gb_model.predict(input_data)[0]

    # Tampilkan hasil prediksi
    st.write("### Hasil Prediksi Biaya Asuransi")
    st.write(f"**Gradient Boosting Prediction:** Rp {gb_pred:,.2f}")

# Informasi tentang model
st.write("### Tentang Model")
st.write("Model Gradient Boosting digunakan untuk memprediksi biaya asuransi berdasarkan faktor-faktor seperti usia, BMI, jumlah anak, kebiasaan merokok, jenis kelamin, dan wilayah.")
