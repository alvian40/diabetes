import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Muat model dan preprocessor
# Pastikan path file sudah benar
model_dt = joblib.load('decision_tree_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Asumsi Anda juga perlu mapping kelas seperti sebelumnya
class_description_mapping = {
    'N': 'No Diabetes',
    'P': 'Prediabetes',
    'Y': 'Diabetes'
}

st.title("Aplikasi Prediksi Diabetes")

st.write("""
Aplikasi ini memprediksi kemungkinan seseorang menderita diabetes berdasarkan fitur-fitur kesehatan yang dimasukkan.
""")

st.sidebar.header('Input Fitur Pasien')

# Fungsi untuk mendapatkan input dari pengguna
def user_input_features():
    # Sesuaikan dengan nama kolom yang ada di dataset Anda
    # Ingat bahwa model dilatih dengan kolom ID dan No_Pation
    # Jadi, meskipun mungkin tidak relevan secara medis untuk input, kita perlu memasukkannya
    # atau melatih ulang model tanpa kolom tersebut jika memang tidak digunakan.
    # Untuk contoh ini, kita akan memasukkan semua kolom yang digunakan saat training.

    # ID dan No_Pation mungkin tidak perlu diinput manual oleh pengguna,
    # bisa di-generate atau diabaikan jika model dilatih tanpa itu.
    # Jika model dilatih DENGAN ID dan No_Pation, kita harus menyediakannya.
    # Mari kita asumsikan ID dan No_Pation diisi acak atau disesuaikan jika ada logikanya.
    # Untuk contoh ini, kita akan buat input teks sederhana atau di-generate.
    # Jika Anda melatih ulang model TANPA ID dan No_Pation, hapus input ini dan sesuaikan kolom di bawah.

    # Opsi 1: Input ID dan No_Pation manual (kurang umum untuk aplikasi user)
    # id_pasien = st.sidebar.text_input('ID Pasien', 'Otomatis') # Bisa di-generate
    # no_pation = st.sidebar.text_input('Nomor Pasien', 'Otomatis') # Bisa di-generate

    # Opsi 2: Generate ID dan No_Pation (sesuai dengan cara kita membuat random_sample sebelumnya)
    # Ini lebih masuk akal jika ID/No_Pation hanya penanda dan tidak mempengaruhi prediksi model
    id_pasien = np.random.randint(1, 1000)
    no_pation = np.random.randint(10000, 99999)


    gender = st.sidebar.selectbox('Jenis Kelamin', ('M', 'F'))
    age = st.sidebar.slider('Usia', 20, 80, 40) # Rentang usia disesuaikan
    urea = st.sidebar.number_input('Urea (mmol/L)', 1.0, 40.0, 5.0, step=0.1) # Rentang dan step disesuaikan
    cr = st.sidebar.number_input('Kreatinin (umol/L)', 10, 1500, 80, step=1) # Rentang dan step disesuaikan
    hba1c = st.sidebar.number_input('HbA1c (%)', 3.0, 15.0, 6.0, step=0.1) # Rentang dan step disesuaikan
    chol = st.sidebar.number_input('Cholesterol (mmol/L)', 1.0, 10.0, 5.0, step=0.1) # Rentang dan step disesuaikan
    tg = st.sidebar.number_input('Trigliserida (mmol/L)', 0.1, 10.0, 1.5, step=0.1) # Rentang dan step disesuaikan
    hdl = st.sidebar.number_input('HDL (mmol/L)', 0.1, 3.0, 1.0, step=0.1) # Rentang dan step disesuaikan
    ldl = st.sidebar.number_input('LDL (mmol/L)', 0.1, 7.0, 3.0, step=0.1) # Rentang dan step disesuaikan
    vldl = st.sidebar.number_input('VLDL (mmol/L)', 0.0, 5.0, 1.0, step=0.1) # Rentang dan step disesuaikan
    bmi = st.sidebar.number_input('BMI (kg/m²)', 15.0, 50.0, 25.0, step=0.1) # Rentang dan step disesuaikan


    data = {'ID': id_pasien,
            'No_Pation': no_pation,
            'Gender': gender,
            'AGE': age,
            'Urea': urea,
            'Cr': cr,
            'HbA1c': hba1c,
            'Chol': chol,
            'TG': tg,
            'HDL': hdl,
            'LDL': ldl,
            'VLDL': vldl,
            'BMI': bmi
           }
    features = pd.DataFrame(data, index=[0])

    # Pastikan urutan kolom sesuai dengan X_train sebelum preprocessing
    # Anda bisa mendapatkan urutan kolom dari X_train jika Anda menyimpannya
    # Untuk saat ini, saya akan menggunakan urutan dari data awal yang kita lihat
    original_columns_order = ['ID', 'No_Pation', 'Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
    features = features[original_columns_order]


    return features

input_df = user_input_features()

st.subheader('Input Fitur Pasien')
st.write(input_df)

# Terapkan preprocessing pada input pengguna
# Penting: Gunakan preprocessor yang SUDAH DILATIH
if 'preprocessor' in locals(): # Cek apakah preprocessor berhasil dimuat
    input_processed = preprocessor.transform(input_df)

    # st.subheader('Input Setelah Preprocessing (Debug)')
    # st.write(input_processed) # Bisa ditampilkan untuk debug


    # Lakukan prediksi menggunakan model yang sudah dilatih
    if 'model_dt' in locals(): # Cek apakah model berhasil dimuat
        st.subheader('Hasil Prediksi')

        # Lakukan prediksi
        prediction = model_dt.predict(input_processed)
        prediction_proba = model_dt.predict_proba(input_processed)

        # Tampilkan prediksi kelas
        predicted_class_label = prediction[0]
        st.write(f"**Prediksi Status Diabetes:** {class_description_mapping.get(predicted_class_label, predicted_class_label)}")

        # Tampilkan probabilitas prediksi
        st.subheader('Probabilitas Prediksi')
        predicted_class_names = model_dt.classes_
        for i, prob in enumerate(prediction_proba[0]):
             st.write(f"{class_description_mapping.get(predicted_class_names[i], predicted_class_names[i])}: {prob:.2%}")

    else:
        st.error("Model belum dimuat dengan benar.")

else:
    st.error("Preprocessor belum dimuat dengan benar.")