# Dev Container Setup Guide

## Como Usar Dev Containers

### 1. Reabrir no Container

**Opção A: Command Palette**
```
Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

**Opção B: Notificação**
- VS Code detecta `.devcontainer/` e mostra popup
- Clique em "Reopen in Container"

### 2. O que acontece?

 VS Code constrói/inicia container Docker 
 Monta workspace dentro do container 
 Instala extensões Python/Jupyter automaticamente 
 Configura kernel Python do container 
 Sincroniza código (volumes montados) 

### 3. Executar Notebooks

1. Abrir `notebooks/01-stdp_example.ipynb`
2. VS Code detecta kernel automaticamente
3. Executar células com `Shift+Enter`
4. Debug com breakpoints funciona!

### 4. Terminal Integrado

- Terminal já está DENTRO do container
- Python path: `/opt/venv/bin/python`
- Todas as bibliotecas disponíveis

```bash
# Verificar ambiente
python --version
pip list

# Executar código
python src/main.py
```

### 5. Debug Python

Criar `.vscode/launch.json`:

```json
{
 "version": "0.2.0",
 "configurations": [
 {
 "name": "Python: Current File",
 "type": "debugpy",
 "request": "launch",
 "program": "${file}",
 "console": "integratedTerminal",
 "justMyCode": false
 }
 ]
}
```

### 6. Comandos Úteis

**Reconstruir container:**
```
Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

**Voltar ao local:**
```
Ctrl+Shift+P → "Dev Containers: Reopen Folder Locally"
```

**Ver logs:**
```
Ctrl+Shift+P → "Dev Containers: Show Container Log"
```

### 7. Estrutura de Arquivos

```
.devcontainer/
 devcontainer.json # Configuração do container
docker-compose.dev.yml # Docker Compose para dev
notebooks/ # Notebooks Jupyter (montado)
data/ # Datasets (montado)
models/ # Modelos treinados (montado)
```

### 8. Vantagens

 **Sem Jupyter Web** - Apenas VS Code 
 **IntelliSense completo** - Autocomplete de brian2, torch, etc. 
 **Debug nativo** - Breakpoints em notebooks 
 **Git integrado** - Commits direto do VS Code 
 **Ambiente isolado** - Não afeta sistema local 
 **Reprodutível** - Mesmo ambiente para todos 

### 9. Troubleshooting

**Container não inicia:**
```bash
# Limpar containers antigos
docker compose -f docker-compose.dev.yml down
docker system prune -f

# Rebuild
Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

**Kernel não encontrado:**
```bash
# Dentro do container
/opt/venv/bin/pip install ipykernel
/opt/venv/bin/python -m ipykernel install --user
```

**Extensões não carregam:**
- Verificar `.devcontainer/devcontainer.json`
- Rebuild container

---

**Pronto!** Agora você pode desenvolver Python, executar notebooks e debugar tudo diretamente no VS Code usando o ambiente Docker! 
