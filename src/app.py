import pickle

import numpy as np
import streamlit as st
import utils

print(utils.load_model_from_disk)

best_mrnet_model_name = 'MRNet_Model3'
best_mrnet_model_cutoff_threshold = 0.429860
mrnet_model = utils.load_model_from_disk(best_mrnet_model_name)

best_kneemri_model_name = 'kneeMRI_Model6'
kneemri_model = utils.load_model_from_disk(best_kneemri_model_name)

st.set_page_config(
    page_title="Medical Image Models",
)

st.title("DETECCIÓN DE DESGARRO DE LIGAMENTO CRUZADO ANTERIOR")

st.subheader("Basado en deep learning")

st.text('''
    Tomás Agustín Barak
    Instituto Politécnico Modelo
    ''')

mri_file = st.file_uploader("Subir RM",
                            type=['npy', 'pck'],
                            key="mri_file")

mrnet_label = {0: 'Sano', 1: 'Desgarro de LCA'}
kneemri_label = {0: 'Sano', 1: 'Desgarro parcial de LCA', 2: 'Desgarro completo de LCA'}

if mri_file is not None:
    if mri_file.name.endswith('.npy'):
        mri_vol = np.load(mri_file)
    elif mri_file.name.endswith('.pck'):
        mri_vol = pickle.load(mri_file)

    mri_vol = mri_vol.astype(np.float64)

    predict_button = st.button('Analizar')

    if predict_button:
        with st.spinner('Preprocesando...'):
            preprocessed_mri_vol = utils.preprocess_mri(mri_vol)

        with st.spinner('Analizando...'):
            mri_vol = np.expand_dims(mri_vol, axis=3)  # Dimension extra para compatibilidad
            # mri_vol.shape
            mrnet_pred_prob = mrnet_model.predict(np.array([preprocessed_mri_vol]))
            print(mrnet_pred_prob)
            mrnet_pred_label = (mrnet_pred_prob[0] >= best_mrnet_model_cutoff_threshold).astype('int')
            print(mrnet_pred_label)

            kneemri_pred_prob = kneemri_model.predict(np.array([preprocessed_mri_vol]))
            print(kneemri_pred_prob)
            kneemri_pred_label = kneemri_pred_prob[0].argmax(axis=-1)
            print(kneemri_pred_label)

            if mrnet_pred_label == 1 and kneemri_pred_label == 0:
                if mrnet_pred_prob[0] > kneemri_pred_prob[0][kneemri_pred_label]:
                    st.write(f'Prediccion de desgarro del LCA: **{mrnet_label[mrnet_pred_label[0]]}**')
                    st.warning("Posible desgarro del LCA, sin certeza sobre el grado del mismo.")
            elif mrnet_pred_label == 0 and kneemri_pred_label > 0:
                if mrnet_pred_prob[0] < kneemri_pred_prob[0][kneemri_pred_label]:
                    st.write(f'Prediccion del grado de desgarro del LCA: **{kneemri_label[kneemri_pred_label]}**')
                    st.warning("Posibilidad de desgarro del LCA")
            else:
                st.write(f'Prediccion de desgarro del LCA: **{mrnet_label[mrnet_pred_label[0]]}**')
                st.write(f'Prediccion del grado de desgarro del LCA: **{kneemri_label[kneemri_pred_label]}**')

    slice_number = st.slider('Corte de la RM', min_value=1,
                             max_value=mri_vol.shape[0], value=1) - 1

    img = mri_vol[slice_number, :, :]
    normalized_image_data = (img - img.min()) / (img.max() - img.min())

    with st.columns(3)[1]:
        st.image(normalized_image_data, width=300)

    # st.error('Disclaimer : The model predictions are just for reference. Please consult your doctor for treatment.')
