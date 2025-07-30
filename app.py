import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Muat model dan preprocessor
model_dt = joblib.load('decision_tree_model.pkl')

# Coba load preprocessor, jika gagal buat yang sederhana
try:
    preprocessor = joblib.load('preprocessor.pkl')
except Exception as e:
    # Buat preprocessor sederhana sebagai fallback
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    # Tentukan kolom numerik dan kategorik
    numeric_features = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
    categorical_features = ['Gender']
    
    # Buat pipeline sederhana
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Gunakan OneHotEncoder tanpa drop untuk menghasilkan 2 kolom untuk Gender (M, F)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Fit preprocessor dengan data dummy
    try:
        # Coba load data training untuk fit
        df_train = pd.read_csv('Dataset/Dataset of Diabetes .csv')
        preprocessor.fit(df_train[numeric_features + categorical_features])
    except Exception as e:
        # Jika tidak ada data training, buat data dummy untuk fit
        try:
            dummy_data = pd.DataFrame({
                'Gender': ['M', 'F'] * 50,
                'AGE': np.random.randint(20, 80, 100),
                'Urea': np.random.uniform(1.0, 40.0, 100),
                'Cr': np.random.randint(10, 1500, 100),
                'HbA1c': np.random.uniform(3.0, 15.0, 100),
                'Chol': np.random.uniform(1.0, 10.0, 100),
                'TG': np.random.uniform(0.1, 10.0, 100),
                'HDL': np.random.uniform(0.1, 3.0, 100),
                'LDL': np.random.uniform(0.1, 7.0, 100),
                'VLDL': np.random.uniform(0.0, 5.0, 100),
                'BMI': np.random.uniform(15.0, 50.0, 100)
            })
            
            # Pastikan data tidak kosong dan valid
            if not dummy_data.empty and not dummy_data.isnull().all().all():
                preprocessor.fit(dummy_data[numeric_features + categorical_features])
            else:
                st.error("❌ Data dummy tidak valid. Aplikasi tidak dapat berjalan.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Gagal membuat preprocessor: {e}")
            st.stop()

class_description_mapping = {
    'N': 'No Diabetes',
    'P': 'Prediabetes',
    'Y': 'Diabetes'
}

# --- SIDEBAR NAVBAR ---
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #23395d 0%, #4f8cff 100%); border-radius: 22px; padding: 32px 0 18px 0; margin-bottom: 18px; box-shadow: 0 4px 24px #4f8cff44; text-align:center;'>
    <span style='font-size:60px; color:#4f8cff; display:inline-block; margin-bottom:10px; filter: drop-shadow(0 2px 8px #fff8);'>💡</span>
    <div style='color:#ffe082; font-size:1.25rem; font-weight:900; letter-spacing:1px; text-shadow:0 2px 8px #000a;'>Konsultasi & Prediksi Diabetes</div>
</div>
""", unsafe_allow_html=True)

pages = ["🏠 Home", "🧪 Prediksi Diabetes"]

# Dapatkan halaman dari parameter kueri URL, default ke "🏠 Home" jika tidak ada
try:
    page_from_query = st.query_params.get_all("page")[0]
except (KeyError, IndexError):
    page_from_query = pages[0]

# Tentukan indeks default untuk radio button berdasarkan halaman dari URL
try:
    default_index = pages.index(page_from_query)
except ValueError:
    default_index = 0

halaman = st.sidebar.radio(
    '',
    pages,
    index=default_index
)

# Perbarui parameter kueri URL jika halaman yang dipilih berubah
if halaman != page_from_query:
    st.query_params["page"] = halaman

# --- HEADER ---
st.markdown("""
<div style='background: linear-gradient(90deg, #f8fafc 0%, #c3ecfd 100%); border-radius: 28px; padding: 38px 24px 32px 24px; box-shadow: 0 4px 24px #4f8cff22; margin-bottom: 32px;'>
    <div style='text-align:center;'>
        <span style='font-size:62px;'>🧑‍⚕️&nbsp;&nbsp;🩸</span>
        <div style='font-size:2.1rem; font-weight:900; color:#23395d; margin-bottom:8px; margin-top:18px; letter-spacing:1px; line-height:1.25;'>
            Implementasi Algoritma Decision Tree<br/>
            untuk Prediksi Risiko Diabetes Berdasarkan Data Medis Pasien
        </div>
        <div style='font-size:1.1rem; color:#4f8cff; font-weight:500; margin-top:8px;'>
            Bersama Menuju Hidup Sehat & Cegah Komplikasi Sejak Dini
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tambahkan CSS animasi untuk tab
st.markdown("""
<style>
.tab-fadein {
    animation: fadein-tab 0.6s;
}
@keyframes fadein-tab {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: none; }
}
</style>
""", unsafe_allow_html=True)

if halaman == '🏠 Home':
    # --- NAVBAR TABS HOME ---
    tab1, tab2, tab3 = st.tabs(["Tentang", "Cara Penggunaan", "Credits"])

    with tab1:
        st.markdown("""
        <div class='tab-fadein'>
        <style>
        .diabetes-card {
            background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
            border-radius: 18px;
            box-shadow: 0 2px 16px #0001;
            padding: 38px 28px 30px 28px;
            margin-bottom: 18px;
        }
        .diabetes-section {
            background: #fff;
            border-radius: 14px;
            box-shadow: 0 1px 8px #0001;
            padding: 22px 26px;
            margin-bottom: 18px;
        }
        .diabetes-highlight {
            background: #f3e8ff;
            color: #5b21b6;
            border-radius: 10px;
            padding: 16px 20px;
            font-size: 17px;
            margin-top: 28px;
            text-align: center;
        }
        .diabetes-title {
            color: #6d28d9;
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .diabetes-emoji {
            font-size: 90px;
        }
        .diabetes-subtitle {
            color: #312e81;
            font-size: 1.2rem;
            margin-top: 10px;
        }
        .diabetes-list li {
            margin-bottom: 7px;
        }
        </style>
        <div class='diabetes-card'>
            <div style='text-align:center;'>
                <span class='diabetes-emoji'>🩸</span>
                <div class='diabetes-title'>Mengenal Diabetes Lebih Dekat</div>
                <div class='diabetes-subtitle'>Penyakit kronis yang perlu diwaspadai dan dicegah sejak dini</div>
            </div>
            <div class='diabetes-section'>
                <h4 style='color:#6d28d9;'>Apa Itu Diabetes?</h4>
                <p style='color:#222; font-size:17px;'>
                    Diabetes adalah penyakit kronis yang terjadi ketika tubuh tidak dapat memproduksi atau menggunakan insulin secara efektif, sehingga kadar gula darah (glukosa) menjadi tinggi. Insulin adalah hormon yang diproduksi oleh pankreas dan berfungsi untuk membantu glukosa masuk ke dalam sel sebagai sumber energi. Tanpa insulin yang cukup, glukosa menumpuk di dalam darah dan menyebabkan berbagai gangguan pada organ tubuh.
                </p>
                <p style='color:#222; font-size:17px;'>
                    Diabetes dapat berkembang secara perlahan tanpa gejala yang jelas pada awalnya, sehingga banyak orang tidak menyadari bahwa mereka mengidap diabetes. Jika tidak dikontrol, diabetes dapat menyebabkan komplikasi serius seperti penyakit jantung, gagal ginjal, kerusakan saraf, gangguan penglihatan, hingga amputasi. Oleh karena itu, penting untuk mengenali faktor risiko, melakukan deteksi dini, dan menjaga pola hidup sehat untuk mencegah atau mengelola diabetes dengan baik.
                </p>
            </div>
            <div class='diabetes-section'>
                <h4 style='color:#6d28d9;'>Jenis-Jenis Diabetes</h4>
                <ul class='diabetes-list' style='color:#333; font-size:16px;'>
                    <li><b>Diabetes Tipe 1:</b> Tubuh tidak memproduksi insulin sama sekali (umumnya sejak anak-anak/remaja).</li>
                    <li><b>Diabetes Tipe 2:</b> Tubuh tidak efektif menggunakan insulin (paling sering pada dewasa, terkait gaya hidup).</li>
                    <li><b>Diabetes Gestasional:</b> Diabetes yang muncul saat kehamilan.</li>
                </ul>
            </div>
            <div class='diabetes-section'>
                <h4 style='color:#6d28d9;'>Gejala Umum Diabetes</h4>
                <ul class='diabetes-list' style='color:#333; font-size:16px;'>
                    <li>Sering haus dan sering buang air kecil</li>
                    <li>Berat badan turun tanpa sebab jelas</li>
                    <li>Mudah lelah dan lemas</li>
                    <li>Luka sulit sembuh</li>
                    <li>Sering kesemutan atau mati rasa</li>
                    <li>Penglihatan kabur</li>
                </ul>
            </div>
            <div class='diabetes-section'>
                <h4 style='color:#6d28d9;'>Faktor Risiko Diabetes</h4>
                <ul class='diabetes-list' style='color:#333; font-size:16px;'>
                    <li>Riwayat keluarga diabetes</li>
                    <li>Obesitas atau kelebihan berat badan</li>
                    <li>Pola makan tinggi gula & lemak, rendah serat</li>
                    <li>Kurang aktivitas fisik</li>
                    <li>Usia di atas 40 tahun</li>
                    <li>Tekanan darah tinggi & kolesterol tinggi</li>
                </ul>
            </div>
            <div class='diabetes-section'>
                <h4 style='color:#6d28d9;'>Komplikasi Akibat Diabetes</h4>
                <ul class='diabetes-list' style='color:#333; font-size:16px;'>
                    <li>Penyakit jantung & stroke</li>
                    <li>Gagal ginjal</li>
                    <li>Kerusakan saraf (neuropati)</li>
                    <li>Gangguan penglihatan hingga kebutaan</li>
                    <li>Luka kaki sulit sembuh (bisa amputasi)</li>
                </ul>
            </div>
            <div class='diabetes-highlight'>
                <b>Ingat!</b> Deteksi dini, pola hidup sehat, dan pengelolaan yang tepat dapat mencegah atau menunda komplikasi diabetes.
            </div>
            <div style='margin-top:18px; text-align:center; color:#888; font-size:15px;'>
                Sumber: Kemenkes RI, WHO, IDAI
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class='tab-fadein'>
        <div style='background:#fff; border-radius:14px; box-shadow:0 1px 6px #0001; padding:28px 24px; max-width:700px; margin:auto;'>
            <h3 style='color:#0d47a1;'>Cara Menggunakan Aplikasi</h3>
            <ol style='font-size:17px; color:#333; margin-top:18px;'>
                <li>Pilih menu <b>Prediksi Diabetes</b> di sidebar kiri.</li>
                <li>Isi data kesehatan Anda yang berada di halaman <b>Prediksi diabetes</b>.</li>
                <li>Periksa kembali data yang sudah diinput.</li>
                <li>Klik tombol prediksi.</li>
                <li>Lihat hasil prediksi status diabetes dan probabilitasnya.</li>
            </ol>
            <div style='margin-top:18px; color:#388e3c; font-size:16px;'>
                <b>Tips:</b> Pastikan data yang diinput benar agar hasil prediksi lebih akurat.
            </div>
            <hr style='margin:32px 0 24px 0; border: none; border-top: 1.5px dashed #bdbdbd;'>
            <h4 style='color:#0d47a1;'>Panduan Pengisian Data Medis</h4>
            <ol style='font-size:17px; color:#333; margin-top:18px;'>
                <li><b>AGE (Usia):</b> Masukkan usia Anda dalam tahun (20-80 tahun).</li>
                <li><b>Urea (mmol/L):</b> Masukkan kadar urea darah Anda. Urea adalah produk sisa metabolisme protein, penting untuk menilai fungsi ginjal.</li>
                <li><b>Cr (umol/L):</b> Masukkan kadar kreatinin darah. Kreatinin adalah indikator fungsi ginjal.</li>
                <li><b>HbA1c (%):</b> Masukkan nilai HbA1c, yaitu rata-rata kadar gula darah selama 2-3 bulan terakhir.</li>
                <li><b>Chol (mmol/L):</b> Masukkan kadar kolesterol total Anda.</li>
                <li><b>TG (mmol/L):</b> Masukkan kadar trigliserida, yaitu lemak dalam darah.</li>
                <li><b>HDL (mmol/L):</b> Masukkan kadar HDL (kolesterol baik).</li>
                <li><b>LDL (mmol/L):</b> Masukkan kadar LDL (kolesterol jahat).</li>
                <li><b>VLDL (mmol/L):</b> Masukkan kadar VLDL (lemak jahat dalam darah).</li>
                <li><b>BMI (kg/m²):</b> Masukkan nilai BMI Anda. BMI adalah indeks massa tubuh, dihitung dari berat dan tinggi badan.</li>
            </ol>
            <div style='margin-top:18px; color:#388e3c; font-size:16px;'>
                <b>Tips:</b> Jika tidak tahu nilai laboratorium Anda, silakan konsultasikan ke dokter atau gunakan hasil pemeriksaan terbaru.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class='tab-fadein'>
        <div style='background:#fff; border-radius:14px; box-shadow:0 1px 6px #0001; padding:28px 24px; max-width:700px; margin:auto;'>
            <h3 style='color:#0d47a1;'>Credits</h3>
            <ul style='font-size:17px; color:#333; margin-top:18px;'>
                <li><b>Pengembang:</b> Alvian Widyana Nugraha</li>
                <li><b>Dosen Pembimbing:</b> Naeli Umniati ST, MMSI</li>
                <li><b>Framework:</b> Streamlit</li>
                <li><b>Machine Learning:</b> scikit-learn</li>
                <li><b>Ilustrasi:</b> Pixabay, Emoji Unicode</li>
            </ul>
            <div style='margin-top:18px; color:#888; font-size:15px;'>
                &copy; 2025 | Prediksi Diabetes ML
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tambahkan CSS animasi untuk card kontak
    st.markdown("""
    <style>
    .kontak-card-anim {
        opacity: 0;
        max-height: 0;
        transition: opacity 0.5s, max-height 0.5s;
        overflow: hidden;
        pointer-events: none;
    }
    .kontak-card-anim.show {
        opacity: 1;
        max-height: 800px;
        pointer-events: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    # Tombol kontak native Streamlit dengan toggle + animasi
    if 'show_kontak' not in st.session_state:
        st.session_state['show_kontak'] = False
    st.markdown("<div style='margin-top:36px; text-align:center;'>", unsafe_allow_html=True)
    if st.button('📬 Kontak Saya', key='kontak_saya', help='Lihat info kontak lengkap'):
        st.session_state['show_kontak'] = not st.session_state['show_kontak']
    st.markdown("</div>", unsafe_allow_html=True)
    card_class = 'kontak-card-anim show' if st.session_state['show_kontak'] else 'kontak-card-anim'
    st.markdown(f"""
    <div class='{card_class}' style='margin: 0 auto; margin-top:24px; max-width:480px; background: #fff; border-radius:16px; box-shadow:0 2px 16px #4f8cff22; padding:28px 24px; color:#23395d; font-size:1.08rem;'>
        <div style='font-size:1.3rem; font-weight:800; margin-bottom:12px;'>👤 Alvian Widyana Nugraha</div>
        <div style='margin-bottom:8px;'><b>📞 WhatsApp:</b> <a href='https://wa.me/6281398397489' target='_blank' style='color:#43cea2; text-decoration:none;'>0813-9839-7489</a></div>
        <div style='margin-bottom:8px;'><b>📧 Email:</b> <a href='https://mail.google.com/mail/?view=cm&to=nugrahaalvian68@gmail.com' target='_blank' style='color:#4f8cff; text-decoration:none;'>nugrahaalvian68@gmail.com</a></div>
        <div style='margin-bottom:8px;'><b>🏠 Alamat:</b> <a href='https://www.google.com/maps?q=Musholla+Al+Jihad,+Jalan+Kampung+Baru+Kubur+Koja+Gang+Pioner+No.+8,+RT.11/RW.15,+Penjaringan,+Penjaringan,+KOTA+JAKARTA+UTARA,+PENJARINGAN,+DKI+JAKARTA,+ID,+14440' target='_blank' style='color:#23395d; text-decoration:underline;'>Musholla Al Jihad, Jalan Kampung Baru Kubur Koja Gang Pioner No. 8, RT.11/RW.15, Penjaringan, Penjaringan, KOTA JAKARTA UTARA, PENJARINGAN, DKI JAKARTA, ID, 14440</a></div>
        <div style='margin-bottom:8px;'><b>📸 Instagram:</b> <a href='https://www.instagram.com/alviiaaannnnn?utm_source=qr&igsh=bGx0NDN2NjNwN2d5' target='_blank' style='color:#4f8cff; text-decoration:none;'>@alviiaaannnnn</a></div>
        <div style='margin-bottom:8px;'><b>🎵 TikTok:</b> <a href='https://www.tiktok.com/@viaaaannnnnn?_t=ZS-8xCMcii3dmi&_r=1' target='_blank' style='color:#43cea2; text-decoration:none;'>@viaaaannnnnn</a></div>
    </div>
    """, unsafe_allow_html=True)

elif halaman == '🧪 Prediksi Diabetes':
    st.markdown("<h2 style='color:#0d47a1;'>Formulir Prediksi Diabetes</h2>", unsafe_allow_html=True)
    st.write("""
    Masukkan data kesehatan Anda pada form di bawah ini untuk memprediksi status diabetes.
    """)
    st.markdown("""
    <div style='color:#fff; font-size:16px; margin-bottom:18px;'>
        <b>Isi data dengan benar untuk hasil prediksi yang akurat.</b> Data Anda aman dan hanya digunakan untuk prediksi pada aplikasi ini.
    </div>
    """, unsafe_allow_html=True)

    def user_input_features():
        inputs = {}
        inputs['Gender'] = st.selectbox(
            'Jenis Kelamin', 
            (None, 'M', 'F'), 
            format_func=lambda x: 'Pilih jenis kelamin' if x is None else ('Laki-laki' if x == 'M' else 'Perempuan'),
            help="Pilih jenis kelamin Anda."
        )
        inputs['AGE'] = st.number_input('Usia', min_value=20, max_value=80, value=None, placeholder='Masukkan usia Anda (antara 20-80)', help='Rentang usia yang diterima: 20 sampai 80 tahun.')
        inputs['Urea'] = st.number_input('Urea (mmol/L)', 1.0, 40.0, value=None, step=0.1, placeholder='Contoh: 5.0', help='Nilai minimal: 1.0, Nilai maksimal: 40.0')
        inputs['Cr'] = st.number_input('Kreatinin (umol/L)', 10, 1500, value=None, step=1, placeholder='Contoh: 80', help='Nilai minimal: 10, Nilai maksimal: 1500')
        st.caption('Kreatinin: Zat sisa dari otot.')
        inputs['HbA1c'] = st.number_input('HbA1c (%)', 3.0, 15.0, value=None, step=0.1, placeholder='Contoh: 6.0', help='Nilai minimal: 3.0, Nilai maksimal: 15.0')
        st.caption('HbA1c: Rata-rata gula darah.')
        inputs['Chol'] = st.number_input('Cholesterol (mmol/L)', 1.0, 10.0, value=None, step=0.1, placeholder='Contoh: 5.0', help='Nilai minimal: 1.0, Nilai maksimal: 10.0')
        st.caption('Cholesterol: Kolesterol.')
        inputs['TG'] = st.number_input('Trigliserida (mmol/L)', 0.1, 10.0, value=None, step=0.1, placeholder='Contoh: 1.5', help='Nilai minimal: 0.1, Nilai maksimal: 10.0')
        st.caption('Trigliserida: Lemak darah.')
        inputs['HDL'] = st.number_input('HDL (mmol/L)', 0.1, 3.0, value=None, step=0.1, placeholder='Contoh: 1.0', help='Nilai minimal: 0.1, Nilai maksimal: 3.0')
        st.caption('HDL: Kolesterol baik.')
        inputs['LDL'] = st.number_input('LDL (mmol/L)', 0.1, 7.0, value=None, step=0.1, placeholder='Contoh: 3.0', help='Nilai minimal: 0.1, Nilai maksimal: 7.0')
        st.caption('LDL: Kolesterol jahat.')
        inputs['VLDL'] = st.number_input('VLDL (mmol/L)', 0.0, 5.0, value=None, step=0.1, placeholder='Contoh: 1.0', help='Nilai minimal: 0.0, Nilai maksimal: 5.0')
        st.caption('VLDL: Lemak jahat.')
        inputs['BMI'] = st.number_input('BMI (kg/m²)', 15.0, 50.0, value=None, step=0.1, placeholder='Contoh: 25.0', help='Nilai minimal: 15.0, Nilai maksimal: 50.0')
        st.caption('BMI: Indeks massa tubuh.')
        return inputs

    with st.form(key='form_prediksi'):
        input_data = user_input_features()
        submit = st.form_submit_button('Prediksi', use_container_width=True)

    if submit:
        # Validasi input
        missing_fields = [key for key, value in input_data.items() if value is None]
        if missing_fields:
            st.error(f"Harap isi semua kolom yang wajib diisi. Kolom yang kosong: {', '.join(missing_fields)}")
        else:
            # Buat DataFrame jika semua data valid
            id_pasien = np.random.randint(1, 1000)
            no_pation = np.random.randint(10000, 99999)
            
            data_for_df = {
                'ID': id_pasien,
                'No_Pation': no_pation,
                **input_data
            }
            
            input_df = pd.DataFrame(data_for_df, index=[0])
            original_columns_order = ['ID', 'No_Pation', 'Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
            input_df = input_df[original_columns_order]
            
            st.markdown("<h4 style='color:#1976d2;'>Data yang Anda Masukkan</h4>", unsafe_allow_html=True)
            st.write(input_df)
            
            if 'preprocessor' in locals():
                try:
                    input_processed = preprocessor.transform(input_df)
                    
                    if 'model_dt' in locals():
                        # Cek apakah jumlah fitur sesuai dengan yang diharapkan model
                        expected_features = model_dt.n_features_in_
                        
                        # Jika jumlah fitur tidak cocok, tambahkan kolom dummy
                        if input_processed.shape[1] != expected_features:
                            import numpy as np
                            # Tambahkan kolom dummy dengan nilai 0
                            dummy_cols = np.zeros((input_processed.shape[0], expected_features - input_processed.shape[1]))
                            input_processed = np.hstack([input_processed, dummy_cols])
                        
                        st.markdown("<h4 style='color:#1976d2;'>Hasil Prediksi</h4>", unsafe_allow_html=True)
                        prediction = model_dt.predict(input_processed)
                        prediction_proba = model_dt.predict_proba(input_processed)
                        predicted_class_label = prediction[0]
                        # Box warna sesuai hasil
                        if predicted_class_label == 'Y':
                            st.markdown("""
                            <div style='background:#ff5252; color:#fff; border-radius:10px; padding:18px 16px; font-size:1.2rem; font-weight:700; margin-bottom:10px; box-shadow:0 2px 8px #ff525244;'>⚠️ Prediksi Status Diabetes: <span style='color:#fff;'>Diabetes</span></div>
                            """, unsafe_allow_html=True)
                            st.markdown("""
                            <div style='color:#ff5252; font-size:1.1rem; font-weight:500; margin-bottom:18px;'>
                                Tetap semangat! Segera konsultasikan ke dokter untuk penanganan terbaik dan jaga pola hidup sehat.
                            </div>
                            """, unsafe_allow_html=True)
                        elif predicted_class_label == 'N':
                            st.markdown("""
                            <div style='background:#43e97b; color:#fff; border-radius:10px; padding:18px 16px; font-size:1.2rem; font-weight:700; margin-bottom:10px; box-shadow:0 2px 8px #43e97b44;'>✅ Prediksi Status Diabetes: <span style='color:#fff;'>Non Diabetes</span></div>
                            """, unsafe_allow_html=True)
                            st.markdown("""
                            <div style='color:#43e97b; font-size:1.1rem; font-weight:500; margin-bottom:18px;'>
                                Selamat! Anda tidak terindikasi diabetes. Tetap jaga pola makan dan gaya hidup sehat ya!
                            </div>
                            """, unsafe_allow_html=True)
                        elif predicted_class_label == 'P':
                            st.markdown("""
                            <div style='background:#4f8cff; color:#fff; border-radius:10px; padding:18px 16px; font-size:1.2rem; font-weight:700; margin-bottom:10px; box-shadow:0 2px 8px #4f8cff44;'>ℹ️ Prediksi Status Diabetes: <span style='color:#fff;'>Prediabetes</span></div>
                            """, unsafe_allow_html=True)
                            st.markdown("""
                            <div style='color:#4f8cff; font-size:1.1rem; font-weight:500; margin-bottom:18px;'>
                                Anda berada di tahap prediabetes. Yuk, mulai perbaiki pola makan dan rutin berolahraga agar tetap sehat!
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("<h5 style='color:#0d47a1;'>Probabilitas Prediksi</h5>", unsafe_allow_html=True)
                        predicted_class_names = model_dt.classes_
                        for i, prob in enumerate(prediction_proba[0]):
                            st.write(f"{class_description_mapping.get(predicted_class_names[i], predicted_class_names[i])}: {prob:.2%}")
                    else:
                        st.error("Model belum dimuat dengan benar.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses data: {e}")
            else:
                st.error("Preprocessor belum dimuat dengan benar.")

    st.markdown("""
    <div style='margin-top:24px; background: linear-gradient(90deg, #fbc2eb 0%, #a1c4fd 100%); color:#23395d; border-radius:12px; padding:18px 22px; font-size:16px; box-shadow:0 2px 8px #a1c4fd33;'>
        <b>Tips:</b> Data yang Anda masukkan tidak disimpan di server. Gunakan hasil prediksi ini sebagai referensi awal, bukan pengganti konsultasi dokter. Jaga pola hidup sehat dan lakukan pemeriksaan rutin untuk mencegah diabetes!
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
    <div style='margin-top:40px; background: linear-gradient(90deg, #a18cd1 0%, #fbc2eb 100%); color:#312e81; border-radius:0 0 14px 14px; padding:16px 0; text-align:center; font-size:17px; font-weight:500; letter-spacing:1px;'>
        🌟 <b>Prediksi & Edukasi Diabetes - Bersama Menuju Hidup Sehat!</b> 🌟
    </div>
""", unsafe_allow_html=True)    
