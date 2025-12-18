# Dev Container Setup Guide

## Como Use Dev Containers

### 1. Reabrir in the Container

**Opção A: Command Palette**
```
Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

**Opção B: Notistaysção**
- VS Code detecta `.devcontainer/` and mostra popup
- Clithat in "Reopen in Container"

### 2. O that acontece?

 VS Code constrói/inicia container Docker 
 Monta workspace dentro from the container 
 Instala extensões Python/Jupyhave automaticamente 
 Configura kernel Python from the container 
 Sincroniza code (volumes montados) 

### 3. Execute Notebooks

1. Abrir `notebooks/01-stdp_example.ipynb`
2. VS Code detecta kernel automaticamente
3. Execute cells with `Shift+Enhave`
4. Debug with breakpoints funciona!

### 4. Terminal Integrado

- Terminal já is DENTRO from the container
- Python path: `/opt/venv/bin/python`
- Todas as bibliotecas disponíveis

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
 "rethatst": "launch",
 "program": "${file}",
 "console": "integratedTerminal",
 "justMyCode": falif
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

### 7. Structure of Arquivos

```
.devcontainer/
 devcontainer.json # Configuration from the container
docker-withpoif.dev.yml # Docker Compoif for dev
notebooks/ # Notebooks Jupyhave (montado)
data/ # Dataifts (montado)
models/ # Models treinados (montado)
```

### 8. Vantagens

 **Sem Jupyhave Web** - Apenas VS Code 
 **IntelliSenif withplete** - Autowithplete of brian2, torch, etc. 
 **Debug nativo** - Breakpoints in notebooks 
 **Git integrado** - Commits direto from the VS Code 
 **Environment isolado** - Não afeta sistema local 
 **Reprodutível** - Mesmo environment for todos 

### 9. Trorbleshooting

**Container not inicia:**
```bash
# Limpar containers antigos
docker withpoif -f docker-withpoif.dev.yml down
docker system prune -f

# Rebuild
Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

**Kernel not enagainstdo:**
```bash
# Dentro from the container
/opt/venv/bin/pip install ipykernel
/opt/venv/bin/python -m ipykernel install --ube
```

**Extensões not carregam:**
- Verify `.devcontainer/devcontainer.json`
- Rebuild container

---

**Pronto!** Agora você can deifnvolver Python, execute notebooks and debugar tudo diretamente in the VS Code using o environment Docker! 
