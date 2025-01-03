import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Fungsi utama
def main():
    st.title("🌊 Water Potability Prediction")
    st.write("Aplikasi ini membantu Anda menganalisis potabilitas air menggunakan algoritma klasifikasi.")

    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Upload Dataset", "Visualisasi Data", "Analisis dan Evaluasi", "Prediksi Kelayakan Air"])

    # Halaman Upload Dataset
    if page == "Upload Dataset":
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Unggah file dataset (CSV)", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)

            # Tampilkan dataset
            st.write("### Dataset (5 Baris Pertama)")
            st.write(dataset.head())

            st.write("### Informasi Dataset")
            st.write(dataset.info())

            st.write("### Nilai Missing")
            st.write(dataset.isnull().sum())

    # Halaman Visualisasi Data
    elif page == "Visualisasi Data":
        st.header("Visualisasi Data")
        uploaded_file = st.sidebar.file_uploader("Unggah file dataset (CSV)", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)

            # Visualisasi distribusi Potability
            st.subheader("Distribusi Potability")
            plt.figure(figsize=(6, 4))
            sns.countplot(data=dataset, x='Potability')
            plt.title("Distribusi Data Kualitas Air")
            plt.xlabel("Potability")
            plt.ylabel("Jumlah")
            st.pyplot(plt)

            # Visualisasi korelasi
            st.subheader("Korelasi Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Korelasi Heatmap")
            st.pyplot(plt)

    # Halaman Analisis dan Evaluasi
    elif page == "Analisis dan Evaluasi":
        st.header("Analisis dan Evaluasi")
        uploaded_file = st.sidebar.file_uploader("Unggah file dataset (CSV)", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)

            # Preprocessing
            st.subheader("Preprocessing Data")
            st.write("Mengganti nilai yang hilang dengan mean...")
            dataset = dataset.fillna(dataset.mean())

            # Fitur dan target
            fitur = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity']
            target = 'Potability'

            X = dataset[fitur]
            y = dataset[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalisasi
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Pemilihan algoritma
            st.subheader("Pilih Algoritma")
            algorithms = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(),
                'SVM': SVC(),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier()
            }
            selected_algo = st.selectbox("Pilih Algoritma", list(algorithms.keys()))
            model = algorithms[selected_algo]

            # Evaluasi
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.write(f"**Akurasi: {accuracy:.4f}**")
            st.write("**Confusion Matrix:**")
            st.write(cm)

            # Plot confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Confusion Matrix - {selected_algo}")
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

    # Halaman Prediksi Kelayakan Air
    elif page == "Prediksi Kelayakan Air":
        st.header("Prediksi Kelayakan Air")
        st.write("Masukkan nilai dari 9 fitur untuk memprediksi apakah air layak diminum atau tidak.")

        # Form input
        ph = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0, step=0.01)
        hardness = st.number_input("Hardness", min_value=0.0, step=0.01)
        solids = st.number_input("Solids", min_value=0.0, step=0.01)
        chloramines = st.number_input("Chloramines", min_value=0.0, step=0.01)
        sulfate = st.number_input("Sulfate", min_value=0.0, step=0.01)
        conductivity = st.number_input("Conductivity", min_value=0.0, step=0.01)
        organic_carbon = st.number_input("Organic Carbon", min_value=0.0, step=0.01)
        trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, step=0.01)
        turbidity = st.number_input("Turbidity", min_value=0.0, step=0.01)

        # Button prediksi
        if st.button("Prediksi"):
            input_data = [[ph, hardness, solids, chloramines, sulfate, conductivity,organic_carbon, trihalomethanes, turbidity]]
            input_data_scaled = scaler.transform(input_data)  # Pastikan scaler diinisialisasi sebelumnya
            prediction = model.predict(input_data_scaled)
            result = "Layak Minum" if prediction[0] == 1 else "Tidak Layak Minum"
            st.success(f"Hasil Prediksi: **{result}**")

if __name__ == "__main__":
    main()
