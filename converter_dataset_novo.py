#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversor de Dataset: XML (CVAT) -> YOLO
Converte o dataset_novo para formato YOLO
"""

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import shutil

def converter_dataset_novo():
    print("🔄 CONVERTENDO DATASET_NOVO PARA YOLO")
    print("=" * 50)
    
    # Caminhos
    source_dir = "dataset_novo"
    target_dir = "dataset_yolo_novo"
    
    xml_file = os.path.join(source_dir, "annotations.xml")
    images_dir = os.path.join(source_dir, "images")
    
    # Criar estrutura YOLO
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, 'labels'), exist_ok=True)
    
    # Ler XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Mapeamento de classes
    class_mapping = {
        'free_parking_space': 0,           # vaga livre
        'not_free_parking_space': 1,       # vaga ocupada
        'partially_free_parking_space': 1  # parcialmente ocupada = ocupada
    }
    
    print(f"📋 Classes: {class_mapping}")
    
    # Processar cada imagem
    image_annotations = {}
    
    # Extrair anotações por imagem
    for image_elem in root.findall('.//image'):
        image_id = image_elem.get('id')
        image_name = image_elem.get('name')
        image_width = int(image_elem.get('width'))
        image_height = int(image_elem.get('height'))
        
        print(f"📸 Processando: {image_name} ({image_width}x{image_height})")
        
        annotations = []
        
        # Processar polígonos
        for polygon in image_elem.findall('.//polygon'):
            label = polygon.get('label')
            points_str = polygon.get('points')
            
            if label in class_mapping and points_str:
                class_id = class_mapping[label]
                
                # Converter pontos do polígono
                points = []
                for point_str in points_str.split(';'):
                    x, y = map(float, point_str.split(','))
                    points.append([x, y])
                
                points = np.array(points)
                
                # Calcular bounding box do polígono
                x_min = np.min(points[:, 0])
                y_min = np.min(points[:, 1])
                x_max = np.max(points[:, 0])
                y_max = np.max(points[:, 1])
                
                # Converter para formato YOLO (normalizado)
                center_x = (x_min + x_max) / 2 / image_width
                center_y = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                # Garantir que está dentro dos limites
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                annotations.append((class_id, center_x, center_y, width, height))
        
        if annotations:
            image_annotations[image_name] = {
                'annotations': annotations,
                'width': image_width,
                'height': image_height
            }
            print(f"  ✅ {len(annotations)} vagas detectadas")
        else:
            print(f"  ⚠️ Nenhuma anotação encontrada")
    
    # Dividir dataset: 70% treino, 15% val, 15% teste
    image_names = list(image_annotations.keys())
    np.random.seed(42)  # Para reprodutibilidade
    np.random.shuffle(image_names)
    
    n_total = len(image_names)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_images = image_names[:n_train]
    val_images = image_names[n_train:n_train+n_val]
    test_images = image_names[n_train+n_val:]
    
    print(f"\n📊 Distribuição:")
    print(f"  🏋️ Treino: {len(train_images)} imagens")
    print(f"  ✅ Validação: {len(val_images)} imagens")
    print(f"  🧪 Teste: {len(test_images)} imagens")
    
    # Copiar imagens e criar labels
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images    }
    
    total_annotations = 0
    
    for split_name, split_images in splits.items():
        print(f"\n📁 Processando split: {split_name}")
        
        for image_name in split_images:
            # Extrair apenas o nome do arquivo (sem 'images/' prefix)
            clean_image_name = image_name.replace('images/', '')
            
            # Copiar imagem
            src_img = os.path.join(images_dir, clean_image_name)
            dst_img = os.path.join(target_dir, split_name, 'images', clean_image_name)
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                
                # Criar arquivo de label
                label_name = clean_image_name.replace('.png', '.txt').replace('.jpg', '.txt')
                label_path = os.path.join(target_dir, split_name, 'labels', label_name)
                
                if image_name in image_annotations:
                    with open(label_path, 'w') as f:
                        for annotation in image_annotations[image_name]['annotations']:
                            class_id, cx, cy, w, h = annotation
                            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                            total_annotations += 1
                
                print(f"  ✅ {clean_image_name}")
            else:
                print(f"  ❌ Imagem não encontrada: {src_img}")
    
    # Criar data.yaml
    yaml_content = f"""# Dataset YOLO Novo - Parking Detection
path: {os.path.abspath(target_dir)}
train: train/images
val: val/images
test: test/images

# Classes
nc: 2
names: ['free_parking_space', 'not_free_parking_space']

# Dataset Info
description: "Dataset de estacionamento com anotações precisas"
total_images: {len(image_names)}
total_annotations: {total_annotations}
train_images: {len(train_images)}
val_images: {len(val_images)}
test_images: {len(test_images)}
"""
    
    yaml_path = os.path.join(target_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n🎉 CONVERSÃO CONCLUÍDA!")
    print(f"📁 Dataset YOLO criado em: {target_dir}")
    print(f"📄 Configuração: {yaml_path}")
    print(f"📊 Total: {len(image_names)} imagens, {total_annotations} anotações")
    
    return yaml_path

if __name__ == "__main__":
    converter_dataset_novo()
