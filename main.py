import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image, top_n=3):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)[0]
    
    # Get top N predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_classes = [class_name_mapping[idx] for idx in top_indices]
    top_confidences = predictions[top_indices] * 100  # Convert confidence to percentage
    
    return top_classes, top_confidences

# Mapping nama kelas dalam bahasa Indonesia
class_name_mapping = {
    0: 'Apel - Jamur Hawar Daun Apel',
    1: 'Apel - Busuk Hitam',
    2: 'Apel - Karat Cedar Apel',
    3: 'Apel - Sehat',
    4: 'Blueberry - Sehat',
    5: 'Ceri (termasuk asam) - Bulai Berpudar',
    6: 'Ceri (termasuk asam) - Sehat',
    7: 'Jagung (jagung) - Bercak Daun Cercospora Bercak Daun Abu',
    8: 'Jagung (jagung) - Karat Umum',
    9: 'Jagung (jagung) - Busuk Daun Utara',
    10: 'Jagung (jagung) - Sehat',
    11: 'Anggur - Busuk Hitam',
    12: 'Anggur - Esca (Cacar Hitam)',
    13: 'Anggur - Layu Daun (Bercak Daun Isariopsis)',
    14: 'Anggur - Sehat',
    15: 'Jeruk - Haunglongbing (Penyakit Hijau Citrus)',
    16: 'Persik - Bacterial spot',
    17: 'Persik - Sehat',
    18: 'Lada, Bel - Bacterial spot',
    19: 'Lada, Bel - Sehat',
    20: 'Kentang - Busuk Daun Awal',
    21: 'Kentang - Busuk Daun Akhir',
    22: 'Kentang - Sehat',
    23: 'Raspberry - Sehat',
    24: 'Kedelai - Sehat',
    25: 'Squash - Bulai Berpudar',
    26: 'Strawberry - Daun Terbakar',
    27: 'Strawberry - Sehat',
    28: 'Tomat - Bacterial spot',
    29: 'Tomat - Busuk Daun Awal',
    30: 'Tomat - Busuk Daun Akhir',
    31: 'Tomat - Molda Daun',
    32: 'Tomat - Bercak Daun Septoria',
    33: 'Tomat - Kutu Laba-laba Dua Bintik',
    34: 'Tomat - Bercak Target',
    35: 'Tomat - Virus Kuning Keriput Daun Tomat',
    36: 'Tomat - Virus Mozaik Tomat',
    37: 'Tomat - Sehat'
}

# Updated recommendations with links
recommendations = {
    'Apel - Jamur Hawar Daun Apel': {
        'recommendation': 'Periksa kondisi lingkungan tumbuh apel dan pastikan pemeliharaan yang tepat.',
        'link': 'https://extension.umn.edu/plant-diseases/apple-scab'
    },
    'Apel - Busuk Hitam': {
        'recommendation': 'Pastikan sanitasi yang baik di sekitar pohon apel dan pertimbangkan penggunaan fungisida.',
        'link': 'https://extension.umn.edu/plant-diseases/black-rot-apple'
    },
    'Apel - Karat Cedar Apel': {
        'recommendation': 'Gunakan varietas apel yang tahan terhadap karat cedar jika memungkinkan.',
        'link': 'https://extension.umn.edu/plant-diseases/cedar-apple-rust'
    },
    'Blueberry - Sehat': {
        'recommendation': 'Lanjutkan pemeliharaan tanaman kentang secara rutin dan perhatikan tanda-tanda penyakit.',
        'link': 'https://www.southernliving.com/garden/growing-blueberries'
    },
    'Ceri (termasuk asam) - Bulai Berpudar': {
        'recommendation': 'Pertimbangkan aplikasi fungisida dan praktik pengelolaan sanitasi yang baik.',
        'link': 'https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/'
    },
    'Jagung (jagung) - Bercak Daun Cercospora Bercak Daun Abu': {
        'recommendation': 'Pertimbangkan rotasi tanaman dan aplikasi fungisida yang tepat.',
        'link': 'https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn'
    },
    'Jagung (jagung) - Karat Umum': {
        'recommendation': 'Pertimbangkan varietas tahan dan praktik sanitasi yang baik.',
        'link': 'https://extension.umn.edu/corn-pest-management/common-rust-corn'
    },
    'Kentang - Sehat':{
        'recommendation': 'Lanjutkan pemeliharaan tanaman kentang secara rutin dan perhatikan tanda-tanda penyakit.',
        'link': 'https://www.corteva.id/berita/Pemeliharaan-dan-Panen-Kentang.html'
    },
    'Kentang - Busuk Daun Awal': {
        'recommendation': 'Pastikan rotasi tanaman yang tepat dan praktik sanitasi yang baik.',
        'link': 'https://extension.umn.edu/disease-management/early-blight-tomato-and-potato'
    },
    'Kentang - Busuk Daun Akhir': {
        'recommendation': 'Gunakan varietas yang tahan dan praktik rotasi tanaman yang baik.',
        'link': 'https://www.ndsu.edu/agriculture/extension/publications/late-blight-potato'
    },
    'Raspberry - Sehat': {
        'recommendation': 'Pantau kondisi tanaman raspberry secara rutin dan pastikan sanitasi yang baik di sekitar kebun raspberry.',
        'link': 'https://extension.umn.edu/fruit/growing-raspberries-home-garden'
    },
    'Kedelai - Sehat': {
        'recommendation': 'Lanjutkan pemeliharaan tanaman kedelai secara rutin dan pastikan kondisi tanah yang sesuai.',
        'link': 'https://tirto.id/cara-menanam-kedelai-pemeliharaan-tanaman-hingga-teknik-panen-gzWX'
    },
    'Squash - Bulai Berpudar': {
        'recommendation': 'Pastikan sirkulasi udara yang baik di sekitar tanaman squash dan hindari penyiraman langsung ke daun.',
        'link': 'https://savvygardening.com/powdery-mildew-on-squash/'
    },
    'Strawberry - Daun Terbakar': {
        'recommendation': 'Pertimbangkan teknik irigasi yang tidak menyemprotkan air ke daun dan pastikan kondisi tanah yang baik.',
        'link': 'https://content.ces.ncsu.edu/leaf-scorch-of-strawberry'
    },
    'Tomat - Bacterial spot': {
        'recommendation': 'Gunakan varietas tomat yang tahan terhadap spot bakterial dan pertimbangkan aplikasi fungisida jika dibutuhkan.',
        'link': 'https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper'
    },
    'Tomat - Busuk Daun Awal': {
        'recommendation': 'Pastikan sanitasi yang baik dan pertimbangkan aplikasi fungisida yang tepat untuk mengendalikan busuk daun awal.',
        'link': 'https://extension.umn.edu/disease-management/early-blight-tomato-and-potato'
    },
    'Tomat - Busuk Daun Akhir': {
        'recommendation': 'Lakukan sanitasi yang baik di lapangan dan pastikan perlakuan pasca-panen yang sesuai.',
        'link': 'https://extension.umn.edu/disease-management/late-blight'
    },
    'Tomat - Molda Daun': {
        'recommendation': 'Pertimbangkan teknik pengendalian hama yang lebih ketat dan pertimbangkan aplikasi fungisida jika diperlukan.',
        'link': 'https://extension.umn.edu/disease-management/tomato-leaf-mold'
    },
    'Tomat - Bercak Daun Septoria': {
        'recommendation': 'Gunakan varietas tomat yang tahan terhadap septoria dan pertimbangkan teknik sanitasi yang lebih baik.',
        'link': 'https://www.missouribotanicalgarden.org/PlantFinder/PlantFinderDetails.aspx?kempercode=a414'
    },
    'Tomat - Kutu Laba-laba Dua Bintik': {
        'recommendation': 'Pertimbangkan penggunaan insektisida jika populasi kutu laba-laba tinggi dan pastikan kondisi pertanian yang baik.',
        'link': 'https://extension.umn.edu/yard-and-garden-insects/spider-mites'
    },
    'Tomat - Bercak Target': {
        'recommendation': 'Pastikan teknik sanitasi yang baik di lapangan dan pertimbangkan aplikasi fungisida jika diperlukan.',
        'link': 'https://www.apsnet.org/edcenter/disandpath/fungalasco/pdlessons/Pages/TargetSpot.aspx'
    },
    'Tomat - Virus Kuning Keriput Daun Tomat': {
        'recommendation': 'Gunakan varietas tomat yang tahan terhadap virus kuning keriput daun tomat dan pastikan kondisi pertanian yang baik.',
        'link': 'https://www.agric.wa.gov.au/tomatoes/tomato-yellow-leaf-curl-virus-tomato'
    },
    'Tomat - Virus Mozaik Tomat': {
        'recommendation': 'Pertimbangkan penggunaan bibit yang bebas virus dan pastikan sanitasi yang baik di lapangan.',
        'link': 'https://extension.umn.edu/disease-management/tobacco-mosaic-virus'
    }
}

# Streamlit UI
st.title("Sistem Pendeteksi Penyakit Tanaman")

menu = ["Beranda", "Riwayat Prediksi", "Deteksi Penyakit"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

if choice == "Beranda":
    st.subheader("Beranda")
    st.write("Selamat datang di Sistem Pendeteksi Penyakit Tanaman!")

elif choice == "Riwayat Prediksi":
    st.subheader("Riwayat Prediksi")
    prediction_history = st.session_state.get('prediction_history', [])
    if len(prediction_history) == 0:
        st.write("Belum ada riwayat prediksi.")
    else:
        for i, (test_image, predicted_class, confidence) in enumerate(prediction_history):
            st.write(f"{i+1}. Prediksi: {predicted_class} ({confidence:.2f}%)")
            st.image(test_image, caption=predicted_class, use_column_width=True)

elif choice == "Deteksi Penyakit":
    st.subheader("Deteksi Penyakit")
    
    method = st.radio("Pilih metode deteksi gambar:", ("Unggah Gambar", "Scan dengan Kamera"))
    
    if method == "Unggah Gambar":
        image_file = st.file_uploader("Unggah gambar daun", type=['jpeg', 'jpg', 'png'])
        if image_file is not None:
            st.image(image_file, caption="Gambar yang diunggah", use_column_width=True)
            test_image = image_file
            
            if st.button("Deteksi Penyakit dari Gambar yang Diunggah"):
                top_classes, top_confidences = model_prediction(test_image)
                st.write("Prediksi teratas:")
                for i, (pred_class, confidence) in enumerate(zip(top_classes, top_confidences)):
                    st.write(f"{i+1}. {pred_class}: {confidence:.2f}%")
                    recommendation = recommendations[pred_class]['recommendation']
                    link = recommendations[pred_class]['link']
                    st.write(f"Rekomendasi: {recommendation} ([Info lebih lanjut]({link}))")
                
                # Save prediction to history
                prediction_history = st.session_state.get('prediction_history', [])
                prediction_history.append((test_image, top_classes[0], top_confidences[0]))
                st.session_state['prediction_history'] = prediction_history
    
    elif method == "Scan dengan Kamera":
        st.write("Silakan buka kamera untuk melakukan pemindaian gambar.")
        camera_file = st.camera_input("Ambil gambar daun")
        
        if camera_file is not None:
            st.image(camera_file, caption="Gambar yang diambil", use_column_width=True)
            test_image = camera_file
            
            if st.button("Deteksi Penyakit dari Gambar yang Diambil"):
                top_classes, top_confidences = model_prediction(test_image)
                st.write("Prediksi teratas:")
                for i, (pred_class, confidence) in enumerate(zip(top_classes, top_confidences)):
                    st.write(f"{i+1}. {pred_class}: {confidence:.2f}%")
                    recommendation = recommendations[pred_class]['recommendation']
                    link = recommendations[pred_class]['link']
                    st.write(f"Rekomendasi: {recommendation} ([Info lebih lanjut]({link}))")
                
                # Save prediction to history
                prediction_history = st.session_state.get('prediction_history', [])
                prediction_history.append((test_image, top_classes[0], top_confidences[0]))
                st.session_state['prediction_history'] = prediction_history

