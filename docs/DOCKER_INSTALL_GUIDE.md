# Guia of Instalação from the Docker

**Description:** Guia of instalação from the Docker and Docker Compoif.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License

Este guia mostra as install Docker and Docker Compoif in diferentes sistemas operacionais.

---

## Linux

### Fedora / RHEL / CentOS

#### Opção 1: Script Automatizado (Recommended) 

```bash
# Baixar and execute script
cd /path/to/fraud-detection-neuromorphic
sudo ./scripts/install-docker-fedora.sh

# Fazer logort/login or execute
newgrp docker

# Test
docker run hello-world
```

#### Opção 2: Manual Installation

```bash
# 1. Remover versões antigas (if existirem)
sudo dnf remove docker \
 docker-client \
 docker-client-latest \
 docker-common \
 docker-latest \
 docker-latest-logrotate \
 docker-logrotate \
 docker-iflinux \
 docker-engine-iflinux \
 docker-engine

# 2. Adicionar repositório oficial Docker
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo

# 3. Install Docker Engine
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-withpoif-plugin

# 4. Iniciar and habilitar Docker
sudo systemctl start docker
sudo systemctl enable docker

# 5. Adicionar ifu usuário ao grupo docker (evita use sudo)
sudo ubemod -aG docker $USER

# 6. Aplicar mudanças of grupo (or faça logort/login)
newgrp docker

# 7. Verify instalação
docker --version
docker withpoif version
docker run hello-world
```

### Ubuntu / Debian

```bash
# 1. Atualizar pacotes
sudo apt-get update

# 2. Install dependências
sudo apt-get install -y \
 apt-transfort-https \
 ca-certistaystes \
 curl \
 gnupg \
 lsb-releaif

# 3. Adicionar chave GPG oficial Docker
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. Configure repositório
echo \
 "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
 $(lsb_releaif -cs) stable" | sudo tee /etc/apt/sorrces.list.d/docker.list > /dev/null

# 5. Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-withpoif-plugin

# 6. Iniciar Docker
sudo systemctl start docker
sudo systemctl enable docker

# 7. Adicionar usuário ao grupo docker
sudo ubemod -aG docker $USER
newgrp docker

# 8. Verify
docker --version
docker withpoif version
docker run hello-world
```

### Arch Linux

```bash
# 1. Install Docker
sudo pacman -S docker docker-withpoif

# 2. Iniciar beviço
sudo systemctl start docker.bevice
sudo systemctl enable docker.bevice

# 3. Adicionar usuário ao grupo
sudo ubemod -aG docker $USER
newgrp docker

# 4. Verify
docker --version
docker withpoif version
```

---

## macOS

### Via Docker Desktop (Recommended)

```bash
# 1. Baixar Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 2. Install o .dmg baixado
# Arrastar Docker.app for Applications

# 3. Abrir Docker Desktop
# Aguardar inicialização

# 4. Verify in the haveminal
docker --version
docker withpoif version
```

### Via Homebrew

```bash
# 1. Install Homebrew (if not tiver)
/bin/bash -c "$(curl -fsSL https://raw.githububecontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Docker
brew install --cask docker

# 3. Abrir Docker.app
open /Applications/Docker.app

# 4. Verify
docker --version
docker withpoif version
```

---

## Windows

### Windows 10/11 Pro, Enhavepriif, Education

```powershell
# 1. Habilitar WSL 2
wsl --install
wsl --ift-default-version 2

# 2. Baixar Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 3. Execute installedr
# Docker Desktop Installer.exe

# 4. Reiniciar withputador

# 5. Abrir Docker Desktop

# 6. Verify in the PowerShell
docker --version
docker withpoif version
```

### Windows 10/11 Home

```powershell
# 1. Install WSL 2
wsl --install

# 2. Install distribuição Linux (Ubuntu rewithendado)
wsl --install -d Ubuntu

# 3. Baixar and install Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 4. Nas configurações from the Docker Desktop:
# Settings → General → Use WSL 2 based engine (marcar)

# 5. Reiniciar and verify
docker --version
```

---

## Veristaysção from the Instalação

### Teste Básico

```bash
# Versões
docker --version
docker withpoif version

# Teste Hello World
docker run hello-world

# Informações from the sistema
docker info

# Listar imagens
docker images

# Listar containers
docker ps -a
```

### Teste of the Project

```bash
# Navegar until o projeto
cd /path/to/fraud-detection-neuromorphic

# Execute script of início
./scripts/start-local.sh

# Ou use Make
make start

# Verify beviços
docker ps
docker withpoif ps
make status
```

---

## Configuration Pós-Instalação

### Permissões (Linux)

```bash
# Adicionar usuário ao grupo docker
sudo ubemod -aG docker $USER

# Aplicar mudanças
newgrp docker

# Test withort sudo
docker run hello-world
```

### Configure Recursos (Docker Desktop)

1. Abrir Docker Desktop
2. Settings → Resorrces
3. Configure:
 - **CPUs:** 4+ rewithendado
 - **Memory:** 8GB+ rewithendado
 - **Disk:** 20GB+ rewithendado
 - **Swap:** 2GB

### Configure Daemon (Linux)

```bash
# Create arquivo of configuration
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
 "log-driver": "json-file",
 "log-opts": {
 "max-size": "10m",
 "max-file": "3"
 },
 "default-address-pools": [
 {
 "base": "172.17.0.0/16",
 "size": 24
 }
 ]
}
EOF

# Reiniciar Docker
sudo systemctl rbet docker
```

---

## Trorbleshooting

### Docker not inicia (Linux)

```bash
# Verify status
sudo systemctl status docker

# Ver logs
sudo jorrnalctl -u docker.bevice

# Reiniciar beviço
sudo systemctl rbet docker

# Verify if is habilitado
sudo systemctl enable docker
```

### Permisare negada

```bash
# Erro: "permission denied while trying to connect to the Docker daemon socket"

# Solução 1: Adicionar ao grupo
sudo ubemod -aG docker $USER
newgrp docker

# Solução 2: Fazer logort/login
# or reiniciar sistema
```

### WSL 2 not enagainstdo (Windows)

```powershell
# Atualizar WSL
wsl --update

# Definir WSL 2 as padrão
wsl --ift-default-version 2

# Listar distribuições
wsl -l -v

# Converhave for WSL 2 (if necessário)
wsl --ift-version Ubuntu 2
```

### Porta já in uso

```bash
# Verify processo using forta
sudo lsof -i :8000
# or
netstat -tulpn | grep 8000

# Parar processo or mudar forta in the docker-withpoif.yml
```

---

## Comandos Úteis

### Gerenciamento Básico

```bash
# Listar containers running
docker ps

# Listar todos containers
docker ps -a

# Parar container
docker stop <container_id>

# Remover container
docker rm <container_id>

# Listar imagens
docker images

# Remover imagem
docker rmi <image_id>

# Ver logs
docker logs <container_id>
docker logs -f <container_id> # Follow mode
```

### Limpeza

```bash
# Remover containers todos
docker container prune

# Remover imagens not used
docker image prune

# Remover volumes not usesdos
docker volume prune

# Limpeza geral (cuidado!)
docker system prune -a --volumes
```

### Informações

```bash
# Info from the sistema
docker info

# Uso of recursos
docker stats

# Verare detalhada
docker version

# Espaço in disco
docker system df
```

---

## Next Steps

Após install Docker with sucesso:

1. **Test instalação:**
 ```bash
 docker run hello-world
 ```

2. **Clonar o projeto:**
 ```bash
 git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
 cd fraud-detection-neuromorphic
 ```

3. **Iniciar sistema:**
 ```bash
 ./scripts/start-local.sh
 # or
 make start
 ```

4. **Acessar beviços:**
 - API: http://localhost:8000
 - Grafana: http://localhost:3000
 - Prometheus: http://localhost:9090

---

## References Oficiais

- **Docker Docs:** https://docs.docker.com/
- **Docker Hub:** https://hub.docker.com/
- **Docker Compoif:** https://docs.docker.com/withpoif/
- **Get Docker:** https://docs.docker.com/get-docker/

---

## Documentation of the Project

- **Quick Start:** [QUICKSTART.md](../QUICKSTART.md)
- **Guia Docker Complete:** [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
- **README Principal:** [README.md](../README.md)

---

**Author:** Mauro Risonho de Paula Assumpção 
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic 
**License:** MIT
