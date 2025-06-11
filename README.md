# Trabalho Final - Inteligência Artificial
## Treinamento de Modelo YOLO v12 para Detecção de Vagas de Estacionamento

Este projeto implementa um sistema completo de **treinamento do zero** de um modelo YOLO v12 para detectar e classificar vagas de estacionamento como vazias ou ocupadas.

## 🎯 Objetivo do Projeto

Desenvolver um modelo de Deep Learning personalizado utilizando YOLO v12, treinado com dataset próprio, seguindo as melhores práticas de Machine Learning:
- **Divisão adequada** dos dados (Train/Validation/Test)
- **Treinamento completo** do modelo do zero
- **Avaliação rigorosa** com métricas apropriadas
- **Visualização** dos resultados e performance

## 🔧 Funcionalidades Implementadas

- ✅ **Download automático** de dataset via Roboflow
- ✅ **Configuração do YOLO v12** com parâmetros otimizados
- ✅ **Treinamento personalizado** do modelo
- ✅ **Validação e teste** em conjuntos separados
- ✅ **Métricas completas** (mAP, Precision, Recall, F1-Score)
- ✅ **Visualização** de curvas de treinamento e resultados
- ✅ **Contagem automática** de vagas livres/ocupadas
- ✅ **Dataset alternativo** para casos de falha no download

## 🛠️ Tecnologias Utilizadas

- **YOLO v12** (Ultralytics) - Modelo de detecção de objetos
- **Roboflow** - Plataforma para datasets de Computer Vision
- **Python 3.13** - Linguagem principal
- **Jupyter Notebook** - Ambiente de desenvolvimento
- **Matplotlib/Seaborn** - Visualização de dados
- **PyTorch** - Framework de Deep Learning (via Ultralytics)

## 📋 Estrutura do Projeto

```
trabalho_inteligencia_artificial/
├── trabalho_final_inteligencia_artificial.ipynb  # Notebook principal
├── README.md                                      # Documentação
├── parking_dataset/                              # Dataset (criado automaticamente)
│   ├── train/                                    # Dados de treinamento
│   ├── val/                                      # Dados de validação
│   └── test/                                     # Dados de teste
├── parking_detection/                            # Modelos treinados
│   └── yolo_parking_v1/                         # Experimento principal
└── predictions/                                  # Resultados das predições
```

## 🚀 Como Executar

### **OPÇÃO 1: 1 COMANDO APENAS ⚡ (Recomendado)**

**No Windows:**
```batch
EXECUTAR_PROJETO.bat
```

**No PowerShell:**
```powershell
.\EXECUTAR_PROJETO.ps1
```

**Ou diretamente:**
```bash
python run_projeto_completo.py
```

**✅ Isso executa TUDO automaticamente:**
- Instala dependências
- Baixa dataset
- Treina modelo
- Valida resultados  
- Gera relatório final

**⏱️ Tempo total: 30-60 minutos**

---

### **OPÇÃO 2: NOTEBOOK INTERATIVO 📱**

1. **Abra o notebook** `trabalho_final_inteligencia_artificial.ipynb`
2. **Execute célula por célula** com `Shift + Enter`
3. **Acompanhe o progresso** em tempo real

3. **As etapas são executadas automaticamente:**
   - **Etapa 1**: Setup do ambiente e download do dataset
   - **Etapa 2**: Configuração e treinamento do YOLO v12
   - **Etapa 3**: Validação do modelo treinado
   - **Etapa 4**: Teste e inferência
   - **Etapa 5**: Visualização dos resultados

## 📊 Etapas do Desenvolvimento

### 1. 📦 Setup e Preparação
- Instalação das bibliotecas necessárias
- Download do dataset via Roboflow
- Verificação da estrutura dos dados

### 2. 🤖 Treinamento do Modelo
- Configuração do YOLO v12 com parâmetros otimizados
- Treinamento por 50 épocas com early stopping
- Salvamento dos checkpoints intermediários

### 3. 📊 Validação e Métricas
- Avaliação no conjunto de teste
- Cálculo de mAP, Precision, Recall, F1-Score
- Análise por classe (vagas livres vs ocupadas)

### 4. 🎯 Teste e Aplicação
- Inferência em imagens reais
- Contagem automática de vagas
- Cálculo da taxa de ocupação do estacionamento

### 5. 📈 Análise dos Resultados
- Visualização das curvas de treinamento
- Matriz de confusão
- Exemplos de detecções
- Gráficos de distribuição

## 📈 Resultados Esperados

O modelo treinado será capaz de:
- **Detectar vagas** com alta precisão (mAP > 0.8)
- **Classificar corretamente** vagas livres vs ocupadas
- **Processar imagens** em tempo real
- **Fornecer métricas** confiáveis de ocupação

## 🔄 Alternativas de Dataset

Caso o download automático falhe, o projeto inclui:
- Instruções para dataset manual
- Links para fontes de dados públicas
- Configuração alternativa do ambiente

## ⚙️ Configurações do Modelo

- **Arquitetura**: YOLO v12 (YOLOv8n como base)
- **Resolução**: 640x640 pixels
- **Batch Size**: 16
- **Épocas**: 50 (com early stopping)
- **Otimizador**: AdamW (padrão do Ultralytics)
- **Classes**: 2 (empty, occupied)

## 📚 Recursos Adicionais

- **Datasets**: Roboflow, Kaggle, GitHub
- **Ferramentas**: labelImg, CVAT, Roboflow Annotate  
- **Documentação**: Ultralytics YOLO, PyTorch
- **Métricas**: mAP, IoU, NMS, Confidence Thresholding

## 👨‍🎓 Autor

Desenvolvido como trabalho final da disciplina de Inteligência Artificial.

**Diferencial deste projeto:**
- ✅ Treinamento completo do zero (não apenas inferência)
- ✅ Dataset personalizado e bem estruturado
- ✅ Métricas de avaliação rigorosas
- ✅ Divisão adequada Train/Val/Test
- ✅ Visualizações profissionais dos resultados
- ✅ Código bem documentado e reproduzível
