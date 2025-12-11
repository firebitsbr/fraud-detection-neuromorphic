# Remote VS Code Access - Docker Deployment Guide

**Descrição:** Guia de acesso remoto via VS Code.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025

## 🚀 Deployment Completo

Este guia configura acesso remoto ao ambiente Docker via VS Code.

---

## Opção 1: Dev Containers (Local/Remote)

### Passo 1: Deploy do Container
```bash
# Deploy do container de desenvolvimento
sudo docker compose -f docker-compose.dev.yml up -d --build
```

### Passo 2: Verificar Status
```bash
# Verificar se container está rodando
docker ps | grep fraud-detection-dev

# Ver logs
docker compose -f docker-compose.dev.yml logs -f
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

### Passo 1: Configurar SSH no Container

Criar `Dockerfile.remote`:
```dockerfile
FROM fraud-detection-api:ubuntu24.04

USER root

# Instalar SSH
RUN apt-get update && \
    apt-get install -y openssh-server sudo && \
    mkdir /var/run/sshd && \
    echo 'appuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Configurar SSH
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Senha para appuser
RUN echo 'appuser:neuromorphic2025' | chpasswd

# Instalar extensões Python/Jupyter
RUN /opt/venv/bin/pip install ipykernel ipython jupyter

USER appuser
WORKDIR /app

EXPOSE 22

# Start SSH daemon
CMD sudo /usr/sbin/sshd -D
```

### Passo 2: Docker Compose para Remote Access

`docker-compose.remote.yml`:
```yaml
version: '3.8'

services:
  fraud-remote:
    build:
      context: .
      dockerfile: Dockerfile.remote
    container_name: fraud-detection-remote
    ports:
      - "2222:22"      # SSH
      - "8000:8000"    # API
      - "8888:8888"    # Jupyter (opcional)
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
    restart: unless-stopped

networks:
  neuromorphic-net:
    driver: bridge
```

### Passo 3: Deploy
```bash
# Build e deploy
docker compose -f docker-compose.remote.yml up -d --build

# Verificar
docker ps | grep fraud-detection-remote
docker logs fraud-detection-remote
```

### Passo 4: Configurar VS Code SSH

**1. Instalar extensão:**
```bash
code --install-extension ms-vscode-remote.remote-ssh
```

**2. Configurar SSH (`~/.ssh/config`):**
```
Host fraud-docker
    HostName localhost
    User appuser
    Port 2222
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

**3. Conectar no VS Code:**
- `Ctrl+Shift+P`
- `Remote-SSH: Connect to Host`
- Selecione: `fraud-docker`
- Senha: `neuromorphic2025`

**4. Abrir workspace:**
- File → Open Folder
- Digite: `/app`

---

## Opção 3: Remote Tunnels (Acesso pela Internet)

### Passo 1: Instalar VS Code CLI no Container
```bash
# Entrar no container
docker exec -it fraud-detection-dev bash

# Baixar VS Code CLI
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz

# Criar tunnel
./code tunnel
```

### Passo 2: Autenticar
- Siga as instruções na tela
- Faça login com GitHub/Microsoft

### Passo 3: Conectar de Qualquer Lugar
- Abra VS Code em qualquer máquina
- `Ctrl+Shift+P` → `Remote-Tunnels: Connect to Tunnel`
- Selecione seu tunnel
- Pronto! Acesso remoto via internet ✅

---

## Comparação das Opções

| Feature | Dev Containers | Remote SSH | Remote Tunnels |
|---------|---------------|------------|----------------|
| **Acesso Local** | ✅ | ✅ | ✅ |
| **Acesso Remoto LAN** | ❌ | ✅ | ✅ |
| **Acesso Internet** | ❌ | ⚠️ (VPN) | ✅ |
| **Configuração** | Simples | Média | Simples |
| **Segurança** | Alta | Alta | Alta |
| **Port Forwarding** | Automático | Manual | Automático |
| **Performance** | Excelente | Excelente | Boa |

---

## Comandos Úteis

### Gerenciar Container
```bash
# Iniciar
docker compose -f docker-compose.dev.yml up -d

# Parar
docker compose -f docker-compose.dev.yml down

# Logs
docker compose -f docker-compose.dev.yml logs -f

# Entrar no container
docker exec -it fraud-detection-dev bash

# Rebuild
docker compose -f docker-compose.dev.yml up -d --build --force-recreate
```

### Debug
```bash
# Ver processos no container
docker top fraud-detection-dev

# Ver uso de recursos
docker stats fraud-detection-dev

# Inspecionar container
docker inspect fraud-detection-dev
```

### Testar Conexão SSH
```bash
# Testar SSH
ssh -p 2222 appuser@localhost

# Copiar arquivos
scp -P 2222 file.txt appuser@localhost:/app/

# Port forwarding manual
ssh -p 2222 -L 8000:localhost:8000 appuser@localhost
```

---

## Troubleshooting

### Container não inicia
```bash
# Ver logs
docker compose -f docker-compose.dev.yml logs

# Rebuild
docker compose -f docker-compose.dev.yml down
docker compose -f docker-compose.dev.yml up -d --build
```

### SSH não conecta
```bash
# Verificar SSH está rodando
docker exec fraud-detection-dev ps aux | grep sshd

# Reiniciar SSH
docker exec fraud-detection-dev sudo /usr/sbin/sshd -D
```

### Kernel Python não aparece no VS Code
```bash
# Instalar ipykernel no container
docker exec -it fraud-detection-dev /opt/venv/bin/pip install ipykernel
docker exec -it fraud-detection-dev /opt/venv/bin/python -m ipykernel install --user
```

### Permissões de arquivo
```bash
# Ajustar ownership dos volumes
docker exec fraud-detection-dev sudo chown -R appuser:appuser /app
```

---

## Segurança - Produção

### 1. Mudar Senhas
```bash
# No container
sudo passwd appuser
```

### 2. Usar Chaves SSH
```bash
# Gerar par de chaves
ssh-keygen -t ed25519 -f ~/.ssh/fraud_docker

# Copiar chave pública para container
docker exec fraud-detection-dev mkdir -p /home/appuser/.ssh
docker cp ~/.ssh/fraud_docker.pub fraud-detection-dev:/home/appuser/.ssh/authorized_keys
docker exec fraud-detection-dev chmod 600 /home/appuser/.ssh/authorized_keys
```

### 3. Desabilitar Password Auth
```bash
# Editar sshd_config no container
docker exec fraud-detection-dev sudo sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
docker exec fraud-detection-dev sudo systemctl restart sshd
```

### 4. Firewall
```bash
# Limitar porta SSH (apenas LAN)
sudo ufw allow from 192.168.0.0/16 to any port 2222
```

---

## Workflow Recomendado

### Desenvolvimento Local
1. Use **Dev Containers** (Opção 1)
2. Commit código via Git integrado
3. Push para GitHub

### Acesso Remoto LAN
1. Use **Remote SSH** (Opção 2)
2. Configure `~/.ssh/config`
3. Conecte via IP da máquina

### Acesso Remoto Internet
1. Use **Remote Tunnels** (Opção 3)
2. Autentique com GitHub
3. Acesse de qualquer lugar

---

## Próximos Passos

✅ Container deployed  
✅ VS Code configurado  
✅ SSH habilitado  
✅ Notebooks acessíveis  

**Pronto para desenvolver remotamente!** 🚀

---

**Autor:** Mauro Risonho de Paula Assumpção  
**Projeto:** Neuromorphic Fraud Detection  
**Data:** Dezembro 2025
