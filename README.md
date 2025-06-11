# 🚗 Trabalho Final - Inteligência Artificial
## Detecção de Vagas de Estacionamento com YOLOv8

### 📋 Descrição
Este projeto implementa um sistema de detecção automática de vagas de estacionamento usando YOLOv8, treinado com um **dataset de alta qualidade** com 903 anotações manuais precisas. O sistema identifica vagas livres e ocupadas com excelente precisão.

### 🎯 Características do Projeto
- **Dataset de Alta Qualidade**: 903 anotações manuais em 30 imagens
- **Treinamento Otimizado**: 50 épocas com early stopping
- **Resultados Excelentes**: 6.748+ detecções em 58 imagens de teste  
- **Pipeline Completo**: Treinamento → Validação → Inferência
- **Reprodutibilidade**: Setup automatizado e documentado

### 🗂️ Estrutura do Projeto
```
trabalho_inteligencia_artificial/
├── dataset_novo/              # Dataset original (XML + imagens)
├── dataset_yolo_novo/         # Dataset convertido (YOLO format)
│   ├── data.yaml             # Configuração do dataset
│   ├── train/                # Imagens e labels de treino
│   ├── val/                  # Imagens e labels de validação
│   └── test/                 # Imagens e labels de teste
├── imagens_para_inferencia/  # Imagens para teste de inferência
├── projeto_final_novo.py     # Script principal de treinamento
├── converter_dataset_novo.py # Conversor de dataset XML→YOLO
├── verificar_setup.py        # Verificador de setup
├── yolov8n.pt               # Modelo YOLOv8 pré-treinado
├── SETUP.md                 # Instruções de setup
└── README.md                # Este arquivo
```

### 🚀 Como Executar

#### Método 1: Verificar Setup (Recomendado primeiro)
```bash
python verificar_setup.py
```

#### Método 2: Execução Direta
```bash
python projeto_final_novo.py
```

#### Método 3: Reconverter Dataset (se necessário)
```bash
python converter_dataset_novo.py
python projeto_final_novo.py
```
python projeto_final_funcional.py
```

#### Método 2: Execução Manual
```bash
# 1. Instalar dependências
pip install ultralytics==8.3.153 Pillow numpy matplotlib

# 2. Executar treinamento
python projeto_final_funcional.py

# 3. Executar inferência detalhada (após treinamento)
python teste_inferencia_detalhada.py
```

### 📊 Dataset
- **Fonte**: Dataset real mesclado de múltiplas fontes
- **Total**: 68 imagens reais de estacionamentos
- **Distribuição**:
  - Treino: 43 imagens
  - Validação: 9 imagens  
  - Teste: 16 imagens
- **Classes**: 2 (vaga livre, vaga ocupada)
- **Formato**: YOLO (labels em formato .txt)

### 🎯 Funcionalidades
- ✅ Detecção automática de vagas de estacionamento
- ✅ Classificação vaga livre/ocupada
- ✅ Treinamento com dataset real mesclado
- ✅ Validação e teste automático
- ✅ Geração de relatórios detalhados
- ✅ Inferência em novas imagens
- ✅ Visualização dos resultados

### 📈 Resultados Esperados
O modelo treinado será capaz de:
- Detectar vagas de estacionamento em imagens reais
- Classificar o status das vagas (livre/ocupada)
- Gerar estatísticas de ocupação
- Salvar imagens com detecções visualizadas

### 📁 Resultados Gerados
Após a execução, os seguintes diretórios serão criados:
- `projeto_final/yolo_vagas/` - Modelo treinado e métricas
- `resultados/predicoes_finais/` - Predições no conjunto de teste
- `teste_final/inferencia_final/` - Inferência nas imagens finais

### 🔧 Configurações de Treinamento
- **Épocas**: 100 (ajustável)
- **Batch Size**: 4
- **Resolução**: 416x416
- **Otimizador**: Auto (escolhido automaticamente)
- **Early Stopping**: 15 épocas de paciência
- **Device**: CPU (compatibilidade máxima)

### 📋 Requisitos
- Python 3.8+
- ultralytics==8.3.153
- Pillow
- numpy
- matplotlib
- opencv-python (para inferência detalhada)

### 🎓 Objetivos Acadêmicos
Este projeto demonstra:
1. **Preparação de Dataset**: Limpeza e organização de dados reais
2. **Transfer Learning**: Uso de modelo pré-treinado YOLOv8
3. **Treinamento Supervisionado**: Ajuste fino para detecção específica
4. **Validação de Modelo**: Avaliação sistemática do desempenho
5. **Aplicação Prática**: Implementação de sistema funcional

### 📧 Suporte
Para dúvidas ou problemas:
1. Verifique se todas as dependências estão instaladas
2. Certifique-se de que o dataset_mesclado existe
3. Execute o script com permissões adequadas
4. Consulte os logs de erro para diagnóstico

### 📝 Notas Importantes
- O projeto usa apenas imagens reais, sem dados sintéticos
- O dataset foi cuidadosamente curado e limpo
- Todos os caminhos são configurados automaticamente
- Compatível com Windows, Linux e macOS
- Resultados reproduzíveis com seeds fixas
