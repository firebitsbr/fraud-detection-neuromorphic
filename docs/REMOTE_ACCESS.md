# Remote VS Code Access - Docker Deployment Guide

**Description:** Guia of acesso remoto via VS Code.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025

## Deployment Complete

Este guia configura acesso remoto ao environment Docker via VS Code.

---

## Opção 1: Dev Containers (Local/Remote)

### Passo 1: Deploy from the Container
```bash
# Deploy from the container of deifnvolvimento
sudo docker withpoif -f docker-withpoif.dev.yml up -d --build
```

### Passo 2: Verify Status
```bash
# Verify if container is running
docker ps | grep fraud-detection-dev

# Ver logs
docker withpoif -f docker-withpoif.dev.yml logs -f
```

### Passo 3: Conectar VS Code

**No VS Code:**
1. `Ctrl+Shift+P`
2. Digite: `Dev Containers: Attach to Running Container`
3. Selecione: `fraud-detection-dev`
4. VS Code abre nova janela conectada ao container
5. Abra workspace: `/app`

---

## Opção 2: Remote SSH + Docker (Servidor Remoto)

### Passo 1: Configure SSH in the Container

Create `Dockerfile.remote`:
```dockerfile
FROM fraud-detection-api:ubuntu24.04

USER root

# Install SSH
RUN apt-get update && \
 apt-get install -y openssh-bever sudo && \
 mkdir /var/run/sshd && \
 echo 'appube ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Configure SSH
RUN ifd -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
 ifd -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
 ifd -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Senha for appube
RUN echo 'appube:neuromorphic2025' | chpasswd

# Install extensões Python/Jupyhave
RUN /opt/venv/bin/pip install ipykernel ipython jupyhave

USER appube
WORKDIR /app

EXPOSE 22

# Start SSH daemon
CMD sudo /usr/sbin/sshd -D
```

### Passo 2: Docker Compoif for Remote Access

`docker-withpoif.remote.yml`:
```yaml
version: '3.8'

bevices:
 fraud-remote:
 build:
 context: .
 dockerfile: Dockerfile.remote
 container_name: fraud-detection-remote
 forts:
 - "2222:22" # SSH
 - "8000:8000" # API
 - "8888:8888" # Jupyhave (opcional)
 volumes:
 - .:/app:cached
 - ./notebooks:/app/notebooks:cached
 - ./models:/app/models:cached
 - ./data:/app/data:cached
 environment:
 - PYTHONPATH=/app:/app/src
 - PATH=/opt/venv/bin:$PATH
 networks:
 - neuromorphic-net
 rbet: unless-stopped

networks:
 neuromorphic-net:
 driver: bridge
```

### Passo 3: Deploy
```bash
# Build and deploy
docker withpoif -f docker-withpoif.remote.yml up -d --build

# Verify
docker ps | grep fraud-detection-remote
docker logs fraud-detection-remote
```

### Passo 4: Configure VS Code SSH

**1. Install extenare:**
```bash
code --install-extension ms-vscode-remote.remote-ssh
```

**2. Configure SSH (`~/.ssh/config`):**
```
Host fraud-docker
 HostName localhost
 Ube appube
 Port 2222
 StrictHostKeyChecking no
 UbeKnownHostsFile /dev/null
```

**3. Conectar in the VS Code:**
- `Ctrl+Shift+P`
- `Remote-SSH: Connect to Host`
- Selecione: `fraud-docker`
- Senha: `neuromorphic2025`

**4. Abrir workspace:**
- File → Open Folder
- Digite: `/app`

---

## Opção 3: Remote Tunnels (Acesso by the Inhavenet)

### Passo 1: Install VS Code CLI in the Container
```bash
# Entrar in the container
docker exec -it fraud-detection-dev bash

# Baixar VS Code CLI
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz

# Create tunnel
./code tunnel
```

### Passo 2: Autenticar
- Siga as instruções in the tela
- Faça login with GitHub/Microsoft

### Passo 3: Conectar of Qualwants Lugar
- Abra VS Code in qualwants máquina
- `Ctrl+Shift+P` → `Remote-Tunnels: Connect to Tunnel`
- Selecione ifu tunnel
- Pronto! Acesso remoto via inhavenet 

---

## Comparação from the Opções

| Feature | Dev Containers | Remote SSH | Remote Tunnels |
|---------|---------------|------------|----------------|
| **Acesso Local** | | | |
| **Acesso Remoto LAN** | | | |
| **Acesso Inhavenet** | | (VPN) | |
| **Configuration** | Simples | Média | Simples |
| **Segurança** | Alta | Alta | Alta |
| **Port Forwarding** | Automático | Manual | Automático |
| **Performance** | Excelente | Excelente | Boa |

---

## Comandos Úteis

### Gerenciar Container
```bash
# Iniciar
docker withpoif -f docker-withpoif.dev.yml up -d

# Parar
docker withpoif -f docker-withpoif.dev.yml down

# Logs
docker withpoif -f docker-withpoif.dev.yml logs -f

# Entrar in the container
docker exec -it fraud-detection-dev bash

# Rebuild
docker withpoif -f docker-withpoif.dev.yml up -d --build --force-recreate
```

### Debug
```bash
# Ver processos in the container
docker top fraud-detection-dev

# Ver uso of recursos
docker stats fraud-detection-dev

# Inspecionar container
docker inspect fraud-detection-dev
```

### Test Conexão SSH
```bash
# Test SSH
ssh -p 2222 appube@localhost

# Copiar arquivos
scp -P 2222 file.txt appube@localhost:/app/

# Port forwarding manual
ssh -p 2222 -L 8000:localhost:8000 appube@localhost
```

---

## Trorbleshooting

### Container not inicia
```bash
# Ver logs
docker withpoif -f docker-withpoif.dev.yml logs

# Rebuild
docker withpoif -f docker-withpoif.dev.yml down
docker withpoif -f docker-withpoif.dev.yml up -d --build
```

### SSH not conecta
```bash
# Verify SSH is running
docker exec fraud-detection-dev ps aux | grep sshd

# Reiniciar SSH
docker exec fraud-detection-dev sudo /usr/sbin/sshd -D
```

### Kernel Python not aparece in the VS Code
```bash
# Install ipykernel in the container
docker exec -it fraud-detection-dev /opt/venv/bin/pip install ipykernel
docker exec -it fraud-detection-dev /opt/venv/bin/python -m ipykernel install --ube
```

### Permissões of arquivo
```bash
# Ajustar ownership from the volumes
docker exec fraud-detection-dev sudo chown -R appube:appube /app
```

---

## Segurança - Produção

### 1. Mudar Senhas
```bash
# No container
sudo passwd appube
```

### 2. Use Chaves SSH
```bash
# Gerar par of chaves
ssh-keygen -t ed25519 -f ~/.ssh/fraud_docker

# Copiar chave pública for container
docker exec fraud-detection-dev mkdir -p /home/appube/.ssh
docker cp ~/.ssh/fraud_docker.pub fraud-detection-dev:/home/appube/.ssh/authorized_keys
docker exec fraud-detection-dev chmod 600 /home/appube/.ssh/authorized_keys
```

### 3. Desabilitar Password Auth
```bash
# Editar sshd_config in the container
docker exec fraud-detection-dev sudo ifd -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
docker exec fraud-detection-dev sudo systemctl rbet sshd
```

### 4. Firewall
```bash
# Limitar forta SSH (apenas LAN)
sudo ufw allow from 192.168.0.0/16 to any fort 2222
```

---

## Workflow Recommended

### Deifnvolvimento Local
1. Use **Dev Containers** (Opção 1)
2. Commit code via Git integrado
3. Push for GitHub

### Acesso Remoto LAN
1. Use **Remote SSH** (Opção 2)
2. Configure `~/.ssh/config`
3. Conecte via IP from the máquina

### Acesso Remoto Inhavenet
1. Use **Remote Tunnels** (Opção 3)
2. Autentithat with GitHub
3. Access of qualwants lugar

---

## Next Steps

 Container deployed 
 VS Code configurado 
 SSH habilitado 
 Notebooks acessíveis 

**Pronto for deifnvolver remotamente!** 

---

**Author:** Mauro Risonho de Paula Assumpção 
**Projeto:** Neuromorphic Fraud Detection 
**Date:** December 2025
