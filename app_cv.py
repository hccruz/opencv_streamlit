import cv2 as cv
import streamlit as st
from PIL import Image
import numpy as np
import skimage as ski


def brilho_imagem(imagem, resultado):
    img_brilho = cv.convertScaleAbs(imagem, beta=resultado)
    return img_brilho


def borra_imagem(imagem, resultado):
    img_borrada = cv.GaussianBlur(imagem, (7, 7), resultado)
    return img_borrada


def melhora_detalhe(imagem):
    img_melhorada = cv.detailEnhance(imagem, sigma_s=34, sigma_r=0.5)
    return img_melhorada


def escala_cinza(imagem):
    img_cinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    return img_cinza


def principal():
    st.title("OpenCV Data App")
    st.subheader("Esse aplicativo web permite integrar processamento de \
                 imagens com OpenCV.")
    st.text("Streamlit com OpenCV")

    arquivo_img = st.file_uploader("Envie a sua imagem",
                                   type=["jpg", "png", "jpeg"])

    taxa_borrao = st.sidebar.slider("Borrão", min_value=0.2,
                                    max_value=3.5)
    qtd_brilho = st.sidebar.slider("Brilho", min_value=-50,
                                   max_value=50, value=0)
    filtro_aprimoramento = st.sidebar.checkbox("Melhorar Detalhes da Imagem")
    img_cinza = st.sidebar.checkbox("Converte para Escala de Cinza")
    img_erosao = st.sidebar.checkbox("Aplicar o filtro erosão")
    img_dilatacao = st.sidebar.checkbox("Aplicar o filtro dilatação")
    img_edge = st.sidebar.checkbox("Aplicar o filtro edge")

    if not arquivo_img:
        return None

    imagem_original = Image.open(arquivo_img)
    imagem_original = np.array(imagem_original)

    imagem_processada = borra_imagem(imagem_original, taxa_borrao)
    imagem_processada = brilho_imagem(imagem_processada, qtd_brilho)

    if filtro_aprimoramento:
        imagem_processada = melhora_detalhe(imagem_processada)

    if img_cinza:
        imagem_processada = escala_cinza(imagem_processada)

    if img_erosao:
        imagem_processada = ski.morphology.erosion(imagem_processada)

    if img_dilatacao:
        imagem_processada = ski.morphology.dilation(imagem_processada)

    if img_edge:
        imagem_processada = ski.filters.sobel(imagem_processada)

    st.text("Imagem Original vs Imagem Processada")

    st.image([imagem_original, imagem_processada])


if __name__ == "__main__":
    principal()
