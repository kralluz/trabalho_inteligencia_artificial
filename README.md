# Trabalho Final - Inteligência Artificial
## Detecção de Vagas de Estacionamento com YOLO

Este projeto utiliza o modelo YOLO (You Only Look Once) para detectar e classificar vagas de estacionamento como vazias ou ocupadas em tempo real.

## Descrição do Projeto

O sistema é capaz de:
- Detectar vagas de estacionamento em imagens
- Classificar cada vaga como "empty" (vazia) ou ocupada
- Contar o número total de vagas livres disponíveis
- Processar múltiplas imagens de teste

## Tecnologias Utilizadas

- **YOLO (Ultralytics)**: Para detecção e classificação de objetos
- **Python**: Linguagem de programação principal
- **Jupyter Notebook**: Ambiente de desenvolvimento
- **OpenCV**: Para processamento de imagens

## Como Executar

1. Clone o repositório de dados:
   ```bash
   git clone https://github.com/Detopall/parking-lot-prediction.git
   ```

2. Instale as dependências:
   ```bash
   pip install ultralytics
   ```

3. Execute o notebook `trabalho_final_inteligencia_artificial.ipynb`

## Estrutura do Projeto

- `trabalho_final_inteligencia_artificial.ipynb`: Notebook principal com o código
- `README.md`: Documentação do projeto

## Resultados Esperados

O modelo processará as imagens de teste e retornará:
- Detecções visuais das vagas identificadas
- Contagem total de vagas livres
- Nível de confiança das predições

## Autor

Desenvolvido como trabalho final da disciplina de Inteligência Artificial.
