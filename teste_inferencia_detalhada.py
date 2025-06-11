#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este script realiza inferência com um modelo YOLOv8 treinado para detecção de vagas de estacionamento,
analisando e exibindo informações detalhadas sobre as detecções em imagens de teste.

Funcionalidades:
- Carrega um modelo YOLOv8 treinado.
- Define um conjunto de imagens de teste.
- Realiza inferência nas imagens, obtendo resultados detalhados (coordenadas, classes, confianças).
- Desenha bounding boxes e rótulos nas imagens originais.
- Imprime informações sobre cada detecção (classe, confiança, coordenadas).
- Salva as imagens com as detecções em um diretório específico.

Requisitos:
- Ultralytics YOLO: pip install ultralytics
- OpenCV: pip install opencv-python
- Modelo treinado (best.pt) no diretório especificado.
- Imagens de teste no diretório especificado.
"""

import cv2
import os
import sys # Adicionado import sys
from ultralytics import YOLO

# Caminho para o modelo treinado
MODEL_PATH = r"C:\\Users\\chenr\\Documents\\GitHub\\trabalho_inteligencia_artificial\\projeto_final\\yolo_vagas\\weights\\best.pt"

# Diretório com as imagens de teste (reais)
IMAGE_DIR = r"C:\\Users\\chenr\\Documents\\GitHub\\trabalho_inteligencia_artificial\\demo_final\\teste_best_model"

# Diretório para salvar as imagens com as detecções (reais)
OUTPUT_DIR = r"C:\\Users\\chenr\\Documents\\GitHub\\trabalho_inteligencia_artificial\\resultados\\predicoes_detalhadas_real"

# Verificar se o diretório de saída existe, senão criar
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Carregar o modelo YOLO
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    sys.exit(1) # Corrigido para sys.exit()

# Listar todas as imagens no diretório de teste
image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]

if not image_files:
    print(f"Nenhuma imagem encontrada em {IMAGE_DIR}")
    sys.exit(1) # Corrigido para sys.exit()

print(f"Encontradas {len(image_files)} imagens para processar.")

# Processar cada imagem
for image_path in image_files:
    print(f"\nProcessando imagem: {image_path}")
    img = None  # Inicializar img como None

    # Carregar a imagem
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erro ao carregar a imagem: {image_path}. Pulando esta imagem.")
            continue
    except Exception as e:
        print(f"Erro ao carregar a imagem {image_path}: {e}. Pulando esta imagem.")
        continue

    # Realizar a inferência
    try:
        results = model(img, verbose=False)  # verbose=False para menos output no console
    except Exception as e:
        print(f"Erro durante a inferência na imagem {image_path}: {e}. Pulando esta imagem.")
        continue

    # Verificar se houve detecções
    if not results or len(results) == 0 or len(results[0].boxes) == 0:
        print("Nenhuma detecção encontrada nesta imagem.")
        if img is not None: # Verificar se a imagem foi carregada antes de salvar
            output_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
            try:
                cv2.imwrite(output_image_path, img)
                print(f"Imagem original (sem detecções) salva em: {output_image_path}")
            except Exception as e:
                print(f"Erro ao salvar a imagem original {output_image_path}: {e}")
        else:
            print(f"Imagem {os.path.basename(image_path)} não pôde ser carregada, então não será salva.")
        continue

    print(f"Detecções encontradas: {len(results[0].boxes)}")

    # Iterar sobre as detecções
    for i, box in enumerate(results[0].boxes):
        # Coordenadas da bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Classe e confiança
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{model.names[cls]} {conf:.2f}"
        print(f"  Detecção {i+1}: Classe={model.names[cls]} (ID: {cls}), Confiança={conf:.2f}, Coordenadas=[({x1},{y1}), ({x2},{y2})]")

        # Desenhar a bounding box na imagem
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Adicionar o rótulo (classe e confiança)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Salvar a imagem com as detecções
    output_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    try:
        cv2.imwrite(output_image_path, img)
        print(f"Imagem com detecções salva em: {output_image_path}")
    except Exception as e:
        print(f"Erro ao salvar a imagem {output_image_path}: {e}")

print("\nProcessamento concluído.")
