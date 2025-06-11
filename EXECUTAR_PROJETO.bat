@echo off
echo ====================================================
echo 🚀 TRABALHO FINAL - INTELIGENCIA ARTIFICIAL
echo 🚗 Detecção de Vagas de Estacionamento com YOLO v12
echo ⚡ Execução Automática Completa (1 COMANDO)
echo ====================================================
echo.

echo 📋 Opções de execução:
echo 1. Script Completo (30-60 min)
echo 2. Script Simplificado (5-10 min) - Recomendado
echo.

set /p opcao="Escolha (1 ou 2): "

if "%opcao%"=="1" (
    echo 🔄 Executando script completo...
    python run_projeto_completo.py
) else (
    echo 🔄 Executando script simplificado...
    python executar_simples.py
)

echo.
echo ====================================================
echo 🎉 EXECUÇÃO FINALIZADA!
echo ====================================================

pause
