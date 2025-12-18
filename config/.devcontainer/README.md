# Dev Container Setup Guide

## How Use Dev Containers

### 1. Reabrir in the Container

**Option A: Command Palette**
```
Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

**Option B: Notification**
- VS Code detecta `.devcontainer/` and mostra popup
- Clithat in "Reopen in Container"

### 2. What acontece?

 VS Code constrói/inicia container Docker 
 Monta workspace dentro from the container 
 Instala extensões Python/Jupyter automatically 
 Configura kernel Python from the container 
 Sincroniza code (volumes montados) 

### 3. Execute Notebooks

1. Abrir `notebooks/01-stdp_example.ipynb`
2. VS Code detecta kernel automatically
3. Execute cells with `Shift+Enhave`
4. Debug with breakpoints funciona!

### 4. Terminal Integrated

- Terminal is already INSIDE from the container
- Python path: `/opt/venv/bin/python`
- All as bibliotecas disponíveis

```bash
# Verify environment
python --version
pip list

# Execute code
python src/main.py
```

### 5. Debug Python

Create `.vscode/launch.json`:

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
 "justMyCode": falif
 }
 ]
}
```

### 6. Commands Useful

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

### 7. Structure of Arquivos

```
.devcontainer/
 devcontainer.json # Configuration from the container
docker-compose.dev.yml # Docker Compose for dev
notebooks/ # Notebooks Jupyter (mounted)
data/ # datasets (mounted)
models/ # Models trained (mounted)
```

### 8. Vantagens

 **without Jupyter Web** - Only VS Code 
 **IntelliSenif complete** - Autowithplete of brian2, torch, etc. 
 **Debug nativo** - Breakpoints in notebooks 
 **Integrated Git** - Commits directly from VS Code 
 **Environment isolated** - Not affects system local 
 **Reproducible** - Same environment for all 

### 9. Trorbleshooting

**Container not inicia:**
```bash
# Limpar containers old
docker compose -f docker-compose.dev.yml down
docker system prune -f

# Rebuild
Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

**Kernel not enabled:**
```bash
# Dentro from the container
/opt/venv/bin/pip install ipykernel
/opt/venv/bin/python -m ipykernel install --ube
```

**Extensões not load:**
- Verify `.devcontainer/devcontainer.json`
- Rebuild container

---

**Pronto!** Agora você can deifnvolver Python, execute notebooks and debug everything diretamente in the VS Code using o environment Docker! 
