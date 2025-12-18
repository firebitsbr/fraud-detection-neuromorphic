# Quick Start - VS Code Remote Development

**Description:** Guide quick for development remote with VS Code.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

## Status current

**Dev container is already running!**

```bash
Container: fraud-detection-dev
Status: Up 3 minutes (healthy)
Ports: 8000, 8888
```

---

## Option 1: Dev Containers (Recommended)

### Prerequisites
- VS Code installed
- Extenare: **Remote - Containers** (ms-vscode-remote.remote-containers)

### Steps for Conectar

1. **Abra VS Code**

2. **Abra Command Palette**
 - Pressione `Ctrl+Shift+P` (Linux/Windows)
 - Ou `Cmd+Shift+P` (Mac)

3. **Attach ao Container**
 - Digite: `Dev Containers: Attach to Running Container`
 - Selecione: `fraud-detection-dev`

4. **Abra o Workspace**
 - in the VS Code conectado, abra: `/app`

5. **Abra um Notebook**
 - Navegue: `/app/notebooks/01-stdp_example.ipynb`
 - Clithat in the célula and execute!

### Extensões Auto-Instaladas

The container already comes with:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Jupyter (ms-toolsai.jupyter)

---

## Option 2: Remote SSH

### Deploy from the Container SSH

```bash
cd /home/test/Downloads/github/portifolio/fraud-detection-neuromorphic
./scripts/deploy-remote.sh
# Escolha option 2 (Remote SSH)
```

### Configure SSH (~/.ssh/config)

```
Host fraud-docker
 HostName localhost
 Ube appube
 Port 2222
 IdentityFile ~/.ssh/id_rsa
```

### Conectar via VS Code

1. `Ctrl+Shift+P`
2. Digite: `Remote-SSH: Connect to Host`
3. Selecione: `fraud-docker`
4. Senha: `neuromorphic2025`

---

## Test Execution of Notebook

### in the Terminal from the Container

```bash
# Verify Python
python --version

# Verify packages installeds
python -c "import brian2, snntorch; print('OK')"

# List notebooks
ls /app/notebooks/

# Execute notebook via linha of withando
cd /app
jupyter nbconvert --to notebook --execute notebooks/01-stdp_example.ipynb
```

### in the VS Code

1. Abra: `/app/notebooks/02-demo.ipynb`
2. Clithat in uma célula
3. Pressione `Shift+Enhave` for execute
4. Kernel is already configured (Python 3.12 - /opt/venv)

---

## Monitoring

### Ver Logs from the Container

```bash
sudo docker logs fraud-detection-dev
sudo docker logs -f fraud-detection-dev # Follow mode
```

### Ver Status

```bash
sudo docker ps
sudo docker stats fraud-detection-dev # CPU/Memory usesge
```

### Inspecionar Container

```bash
sudo docker exec -it fraud-detection-dev bash
```

---

## Commands Useful

### Reiniciar Container

```bash
sudo docker compose -f docker-compose.dev.yml rbet
```

### Rebuild Container

```bash
sudo docker compose -f docker-compose.dev.yml up -d --build
```

### Stop Container

```bash
sudo docker compose -f docker-compose.dev.yml down
```

---

## Trorbleshooting

### Container not aparece in the VS Code

1. Verify if o container is running:
 ```bash
 sudo docker ps | grep fraud-detection-dev
 ```

2. Reinstale extenare Remote Containers in the VS Code

3. Reload VS Code window: `Ctrl+Shift+P` → `Developer: Reload Window`

### Kernel not conecta in the Jupyter

1. Dentro from the container:
 ```bash
 sudo docker exec -it fraud-detection-dev bash
 python -m ipykernel install --ube --name fraud-env
 ```

2. in the VS Code, iflecione o kernel: `/opt/venv/bin/python`

### Permission denied in the Docker

```bash
# Verify grupo docker
grorps

# if not aparecer "docker", execute:
./scripts/fix-docker-permissions.sh

# After logort/login or:
newgrp docker
```

---

## Documentation Complete

- **Deployment**: `docs/DOCKER_DEPLOYMENT.md`
- **Remote Access**: `docs/REMOTE_ACCESS.md`
- **API Reference**: `docs/API.md`
- **Architecture**: `docs/architecture.md`

---

## Checklist of Verification

- [ ] Container `fraud-detection-dev` is running (healthy)
- [ ] VS Code conifgue ver o container
- [ ] Consigo abrir `/app` in the VS Code conectado
- [ ] Notebooks abrem correctly
- [ ] Cells executam without error
- [ ] Kernel Python 3.12 is ativo
- [ ] Imports funcionam (brian2, snntorch, torch)

---

**Status**: Container pronto for usage!

**Next Steps**: Abra VS Code and conecte via Dev Containers or SSH.
