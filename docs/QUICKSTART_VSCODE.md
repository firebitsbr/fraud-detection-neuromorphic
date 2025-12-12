# Quick Start - VS Code Remote Development

**Descrição:** Guia rápido para desenvolvimento remoto com VS Code.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025

## Status Atual

**Container dev já está rodando!**

```bash
Container: fraud-detection-dev
Status: Up 3 minutes (healthy)
Ports: 8000, 8888
```

---

## Opção 1: Dev Containers (Recomendado)

### Pré-requisitos
- VS Code instalado
- Extensão: **Remote - Containers** (ms-vscode-remote.remote-containers)

### Passos para Conectar

1. **Abra VS Code**

2. **Abra Command Palette**
 - Pressione `Ctrl+Shift+P` (Linux/Windows)
 - Ou `Cmd+Shift+P` (Mac)

3. **Attach ao Container**
 - Digite: `Dev Containers: Attach to Running Container`
 - Selecione: `fraud-detection-dev`

4. **Abra o Workspace**
 - No VS Code conectado, abra: `/app`

5. **Abra um Notebook**
 - Navegue: `/app/notebooks/01-stdp_example.ipynb`
 - Clique na célula e execute!

### Extensões Auto-Instaladas

O container já vem com:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Jupyter (ms-toolsai.jupyter)

---

## Opção 2: Remote SSH

### Deploy do Container SSH

```bash
cd /home/test/Downloads/github/portifolio/fraud-detection-neuromorphic
./scripts/deploy-remote.sh
# Escolha opção 2 (Remote SSH)
```

### Configure SSH (~/.ssh/config)

```
Host fraud-docker
 HostName localhost
 User appuser
 Port 2222
 IdentityFile ~/.ssh/id_rsa
```

### Conectar via VS Code

1. `Ctrl+Shift+P`
2. Digite: `Remote-SSH: Connect to Host`
3. Selecione: `fraud-docker`
4. Senha: `neuromorphic2025`

---

## Testar Execução de Notebook

### No Terminal do Container

```bash
# Verificar Python
python --version

# Verificar pacotes instalados
python -c "import brian2, snntorch; print('OK')"

# Listar notebooks
ls /app/notebooks/

# Executar notebook via linha de comando
cd /app
jupyter nbconvert --to notebook --execute notebooks/01-stdp_example.ipynb
```

### No VS Code

1. Abra: `/app/notebooks/02-demo.ipynb`
2. Clique em uma célula
3. Pressione `Shift+Enter` para executar
4. Kernel já está configurado (Python 3.12 - /opt/venv)

---

## Monitoramento

### Ver Logs do Container

```bash
sudo docker logs fraud-detection-dev
sudo docker logs -f fraud-detection-dev # Follow mode
```

### Ver Status

```bash
sudo docker ps
sudo docker stats fraud-detection-dev # CPU/Memory usage
```

### Inspecionar Container

```bash
sudo docker exec -it fraud-detection-dev bash
```

---

## Comandos Úteis

### Reiniciar Container

```bash
sudo docker compose -f docker-compose.dev.yml restart
```

### Rebuild Container

```bash
sudo docker compose -f docker-compose.dev.yml up -d --build
```

### Parar Container

```bash
sudo docker compose -f docker-compose.dev.yml down
```

---

## Troubleshooting

### Container não aparece no VS Code

1. Verifique se o container está rodando:
 ```bash
 sudo docker ps | grep fraud-detection-dev
 ```

2. Reinstale extensão Remote Containers no VS Code

3. Reload VS Code window: `Ctrl+Shift+P` → `Developer: Reload Window`

### Kernel não conecta no Jupyter

1. Dentro do container:
 ```bash
 sudo docker exec -it fraud-detection-dev bash
 python -m ipykernel install --user --name fraud-env
 ```

2. No VS Code, selecione o kernel: `/opt/venv/bin/python`

### Permission denied no Docker

```bash
# Verificar grupo docker
groups

# Se não aparecer "docker", executar:
./scripts/fix-docker-permissions.sh

# Depois logout/login ou:
newgrp docker
```

---

## Documentação Completa

- **Deployment**: `docs/DOCKER_DEPLOYMENT.md`
- **Remote Access**: `docs/REMOTE_ACCESS.md`
- **API Reference**: `docs/API.md`
- **Architecture**: `docs/architecture.md`

---

## Checklist de Verificação

- [ ] Container `fraud-detection-dev` está rodando (healthy)
- [ ] VS Code consegue ver o container
- [ ] Consigo abrir `/app` no VS Code conectado
- [ ] Notebooks abrem corretamente
- [ ] Células executam sem erro
- [ ] Kernel Python 3.12 está ativo
- [ ] Imports funcionam (brian2, snntorch, torch)

---

**Status**: Container pronto para uso!

**Next Steps**: Abra VS Code e conecte via Dev Containers ou SSH.
