import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
import numpy as np
import matplotlib.pyplot as plt
import shap

# Cache loading of the model and data
@st.cache_resource
def load_model():
    return joblib.load('models/catboost-airnology-2.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('datasets/df_fe_test.csv')
    if 'id' in df.columns:
        df_no_id = df.drop(columns=['id'])
    else:
        df_no_id = df
    shap_values = np.load('models/shap_values.npy')  # SHAP values
    base_values = np.load('models/base_values.npy')  # Base values
    return df, df_no_id, shap_values, base_values

# Load model and dataset
model = load_model()
df, df_no_id, shap_values, base_values = load_data()

# Class mapping
class_mapping = {
    0: 'Background',
    1: 'Benign',
    2: 'Bruteforce',
    3: 'Bruteforce-XML',
    4: 'Probing',
    5: 'XMRIGCC CryptoMiner'
}

# Reverse class mapping
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

# Initialize session states
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'selected_class_index' not in st.session_state:
    st.session_state.selected_class_index = None
if 'prediction_feedback' not in st.session_state:
    st.session_state.prediction_feedback = pd.DataFrame(columns=['instance_index', 'prediction', 'feedback'])
if 'explanation_feedback' not in st.session_state:
    st.session_state.explanation_feedback = pd.DataFrame(columns=['instance_index', 'class', 'feedback'])

# Title and description
st.title("Prediksi Model dan SHAP Explainability")
st.write("**Presented by:** mesin cuci lari pagi 🔥🔥🔥🔥🔥")

# Dataset viewer placed under title
st.subheader("Tampilan Dataset")
st.dataframe(df)

# Option to choose between Local or Global SHAP explanations
explanation_type = st.radio(
    "Pilih Tipe Penjelasan:",
    ('Prediksi dan Penjelasan Lokal', 'Penjelasan Global')
)

# Local Explanation (Prediction and SHAP)
if explanation_type == 'Prediksi dan Penjelasan Lokal':
    # Select an instance
    selected_instance = st.selectbox("Pilih instance untuk prediksi:", df.index)

    # Display the selected instance
    st.subheader(f"Instance yang Dipilih: {selected_instance}")
    st.write(df.loc[selected_instance])

    # Make prediction
    if st.button("Prediksi"):
        instance = df_no_id.loc[[selected_instance]]  # Get selected row without 'id'
        prediction = model.predict(instance)[0][0]  # Model prediction (string)
        
        # Map the prediction (string) back to its integer class index
        selected_class_index = reverse_class_mapping[prediction]

        # Save prediction and selected_class_index to session state
        st.session_state.prediction = prediction
        st.session_state.selected_class_index = selected_class_index

        # Display prediction
        st.subheader("Prediksi")
        st.write(f"Kelas yang Diprediksi: {prediction}")

        # Allow user to flag if this prediction is wrong
        if st.button("Tandai Prediksi Salah"):
            st.write("Anda menandai prediksi ini sebagai salah.")
            # Log the feedback into the prediction_feedback pool (DataFrame)
            new_prediction_feedback = pd.DataFrame({
                'instance_index': [selected_instance],
                'prediction': [prediction],
                'feedback': ['Prediksi Salah']
            })
            st.session_state.prediction_feedback = pd.concat([st.session_state.prediction_feedback, new_prediction_feedback], ignore_index=True)

    # If prediction exists in session state
    if st.session_state.prediction:
        # Allow user to select another class for SHAP explanation (default is predicted class)
        selected_class = st.selectbox(
            "Pilih kelas untuk melihat SHAP values:",
            options=list(class_mapping.values()),
            index=list(class_mapping.values()).index(st.session_state.prediction)  # Default to predicted class
        )

        # Get the class index for the selected class (reverse mapping)
        selected_class_index = reverse_class_mapping[selected_class]

        # Add SHAP explanation button
        if st.button("Tampilkan Penjelasan SHAP"):
            # SHAP explanation for the selected instance
            instance_shap_values = shap_values[selected_class_index, selected_instance, :-1]

            # Prepare base value for the specific class and instance
            base_value_class = base_values[selected_class_index]

            # Extract the input data (features) for the specific instance
            instance_data = df_no_id.loc[selected_instance].values

            # Create SHAP explanation object for the specific class and instance
            shap_exp = shap.Explanation(
                values=instance_shap_values,  # SHAP values for the selected class
                base_values=base_value_class,  # Base value for the selected class
                data=instance_data,  # Data for the instance
                feature_names=df_no_id.columns.tolist()  # Feature names
            )

            # Create the Matplotlib figure
            fig, ax = plt.subplots()

            # Explanation guide on how to read the plot
            st.write("""
                **Cara Membaca Grafik SHAP Waterfall**:
                - **Sumbu X**: Kontribusi setiap fitur terhadap prediksi.
                - **Sumbu Y**: Fitur diurutkan berdasarkan kepentingannya.
                - **SHAP Positif**: Mendorong prediksi ke arah kelas positif.
                - **SHAP Negatif**: Mendorong prediksi ke arah kelas negatif.
            """)

            # Display SHAP waterfall plot for the selected instance
            st.write(f"Grafik SHAP Waterfall untuk Kelas: {selected_class}")
            shap.initjs()
            shap.waterfall_plot(shap_exp, max_display=10)
            st.pyplot(fig)

            # Allow the user to provide feedback on the explanation
            explanation_feedback = st.selectbox("Beri feedback untuk penjelasan ini:", [
                "Penjelasan Bagus", "Terlalu Rumit", "Fitur Tidak Relevan", "Penjelasan Tidak Jelas", "Saya pikir ini salah."
            ])

            if st.button("Kirim Feedback Penjelasan"):
                new_explanation_feedback = pd.DataFrame({
                    'instance_index': [selected_instance],
                    'class': [selected_class],
                    'feedback': [explanation_feedback]
                })
                st.session_state.explanation_feedback = pd.concat([st.session_state.explanation_feedback, new_explanation_feedback], ignore_index=True)
                st.write("Feedback Anda telah dikirim.")

# Global Explanation
elif explanation_type == 'Penjelasan Global':
    global_option = st.radio("Pilih Tipe Penjelasan Global:", ("Summary Plot", "Hubungan Nilai Fitur dengan Target"))

    # Global SHAP Summary Plot
    if global_option == "Summary Plot":
        st.subheader("Plot Ringkasan SHAP Global")

        # Select class to explain SHAP values for
        selected_class = st.selectbox(
            "Pilih kelas untuk melihat SHAP values:",
            options=list(class_mapping.values())  # Display class names
        )

        # Get the class index for the selected class (reverse mapping)
        selected_class_index = reverse_class_mapping[selected_class]

        # Add button to generate global SHAP explanation
        if st.button("Tampilkan Plot Ringkasan SHAP Global"):
            st.write(f"Plot Ringkasan SHAP untuk kelas {selected_class}")

            # SHAP Summary Plot for the selected class
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[selected_class_index], df_no_id)
            st.pyplot(fig)

            st.write("""
                **Cara Membaca Plot Ringkasan SHAP**:
                - **Sumbu X**: Nilai SHAP untuk setiap fitur.
                - **Sumbu Y**: Fitur yang paling berkontribusi pada prediksi.
                - Warna menunjukkan nilai fitur: merah = nilai tinggi, biru = nilai rendah.
            """)

            # Allow the user to provide feedback
            explanation_feedback = st.selectbox("Beri feedback untuk plot ini:", [
                "Penjelasan Bagus", "Terlalu Rumit", "Fitur Tidak Relevan", "Penjelasan Tidak Jelas", "Saya pikir ini salah."
            ])

            if st.button("Kirim Feedback Penjelasan"):
                new_explanation_feedback = pd.DataFrame({
                    'instance_index': ['Global Summary'],
                    'class': [selected_class],
                    'feedback': [explanation_feedback]
                })
                st.session_state.explanation_feedback = pd.concat([st.session_state.explanation_feedback, new_explanation_feedback], ignore_index=True)
                st.write("Feedback Anda telah dikirim.")

    # Relationship of Feature Values with Target
    elif global_option == "Hubungan Nilai Fitur dengan Target":
        st.subheader("Hubungan Nilai Fitur dengan Target")

        # Allow the user to select a feature
        feature_name = st.selectbox("Pilih fitur untuk analisis:", df.columns.tolist())
        # Provide summary statistics for the selected feature
        st.write("Ringkasan Fitur:")
        st.write(df[feature_name].describe(percentiles=np.arange(0, 1.1, 0.1)))

        # Check if the feature is continuous or discrete
        def is_discrete_column(col):
            return pd.api.types.is_integer_dtype(col) or \
                   (pd.api.types.is_numeric_dtype(col) and np.all(np.equal(np.mod(col.dropna(), 1), 0)))

        is_continuous = not is_discrete_column(df[feature_name])

        if is_continuous:
            # Allow the user to define bins
            user_bins = st.text_input("Tentukan interval (bin) untuk fitur (pisahkan dengan koma):", value="")
            if user_bins:
                user_bins = list(map(float, user_bins.split(',')))

                # Bin the feature using user-defined bins
                df_binned = pd.cut(df[feature_name], bins=user_bins, include_lowest=True, right=False)

                # Drop NaN values
                valid_indices = ~df_binned.isna()
                df_binned = df_binned[valid_indices]
                df_filtered = df.loc[valid_indices].reset_index(drop=True)

                # Update SHAP values to match filtered data
                shap_values_filtered = [shap_values[i][valid_indices.values] for i in range(len(class_mapping))]

                # Get midpoints for bins and format x-axis labels
                bin_midpoints = [interval.mid for interval in df_binned.cat.categories]
                x_axis_labels = [f"{interval.left} ≤ x < {interval.right}" for interval in df_binned.cat.categories]
                categories = df_binned.cat.categories

        else:
            # For discrete features, allow the user to define the range of values to show
            unique_values = sorted(df[feature_name].dropna().unique())
            st.write(f"Nilai unik yang tersedia: {unique_values}")

            # Allow the user to input a range of discrete values to include
            user_bins = st.text_input("Tentukan nilai diskrit untuk ditampilkan (pisahkan dengan koma):", value="")

            if user_bins:
                # Convert user input into a list of values
                user_bins = list(map(float if is_discrete_column(df[feature_name]) else str, user_bins.split(',')))

                # Filter data based on the selected discrete values
                df_filtered = df[df[feature_name].isin(user_bins)].reset_index(drop=True)
                df_binned = df_filtered[feature_name]

                # Update SHAP values to match filtered data
                shap_values_filtered = [shap_values[i][df[feature_name].isin(user_bins).values] for i in range(len(class_mapping))]

                # Use the selected values as bin midpoints and labels
                bin_midpoints = user_bins
                x_axis_labels = [f"x = {val}" for val in user_bins]
                categories = user_bins

        if user_bins:
            # Step 2: Aggregate SHAP values by feature bin and class
            feature_index = df_filtered.columns.get_loc(feature_name)

            # Extract SHAP values for the feature for each class
            shap_values_feature = [shap_values_filtered[i][:, feature_index] for i in range(len(class_mapping))]

            # Store the average SHAP value per bin and class
            shap_means_per_bin = []
            for i in range(len(class_mapping)):
                shap_means = []
                for bin_val in categories:
                    if is_continuous:
                        bin_mask = df_binned == bin_val
                    else:
                        bin_mask = df_binned == bin_val

                    if bin_mask.any():
                        mean_shap = shap_values_feature[i][bin_mask].mean()
                        shap_means.append(mean_shap)
                    else:
                        shap_means.append(np.nan)
                shap_means_per_bin.append(shap_means)

            # Step 3: Plot the SHAP values per bin and class
            plt.figure(figsize=(10, 6))

            # Add lines and markers for SHAP values
            for i, class_name in enumerate(class_mapping.values()):
                plt.plot(
                    bin_midpoints,
                    shap_means_per_bin[i],
                    marker='o',
                    linestyle='-',
                    label=class_name,
                )

            # Customize the x-axis ticks
            plt.xticks(bin_midpoints, labels=x_axis_labels, rotation=15, fontsize=12)

            # Set plot labels and title
            plt.title(f"Hubungan Fitur {feature_name} dengan Target", fontsize=16)
            plt.ylabel("Nilai SHAP Rata-rata", fontsize=14)

            # Move the legend to the top corner
            plt.legend(title="Kelas", loc='upper right')

            # Remove grid
            plt.grid(False)

            # Adjust layout and show the plot
            plt.tight_layout()
            st.pyplot(plt)

            st.write("""
                **Cara Membaca Plot Hubungan Nilai Fitur dengan Target**:
                - **Sumbu X**: Interval atau nilai diskrit dari fitur yang dipilih.
                - **Sumbu Y**: Nilai SHAP rata-rata untuk setiap kelas.
                - Garis menunjukkan pengaruh fitur terhadap kelas target pada interval yang berbeda.
            """)

            # Allow the user to provide feedback
            explanation_feedback = st.selectbox("Beri feedback untuk plot ini:", [
                "Penjelasan Bagus", "Terlalu Rumit", "Fitur Tidak Relevan", "Penjelasan Tidak Jelas", "Saya pikir ini salah."
            ])

            if st.button("Kirim Feedback Penjelasan"):
                new_explanation_feedback = pd.DataFrame({
                    'instance_index': ['Global Feature Relationship'],
                    'class': [feature_name],
                    'feedback': [explanation_feedback]
                })
                st.session_state.explanation_feedback = pd.concat([st.session_state.explanation_feedback, new_explanation_feedback], ignore_index=True)
                st.write("Feedback Anda telah dikirim.")
