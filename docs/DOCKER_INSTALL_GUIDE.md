# Guide of Installation from the Docker

**Description:** Guide of installation from the Docker and Docker Compose.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License

This guide mostra as install Docker and Docker Compose in different systems operacionais.

---

## Linux

### Fedora / RHEL / CentOS

#### Option 1: Script Automated (Recommended) 

```bash
# Download and execute script
cd /path/to/fraud-detection-neuromorphic
sudo ./scripts/install-docker-fedora.sh

# Make logort/login or execute
newgrp docker

# Test
docker run hello-world
```

#### Option 2: Manual Installation

```bash
# 1. Remover versions old (if existirem)
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

# 2. add repositório oficial Docker
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.with/linux/fedora/docker-ce.repo

# 3. Install Docker Engine
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 4. Start and habilitar Docker
sudo systemctl start docker
sudo systemctl enable docker

# 5. add ifu user ao grupo docker (evita use sudo)
sudo ubemod -aG docker $USER

# 6. Aplicar mudanças of grupo (or faça logort/login)
newgrp docker

# 7. Verify installation
docker --version
docker compose version
docker run hello-world
```

### Ubuntu / Debian

```bash
# 1. Update packages
sudo apt-get update

# 2. Install dependencies
sudo apt-get install -y \
 apt-transfort-https \
 ca-certistaystes \
 curl \
 gnupg \
 lsb-releaif

# 3. add chave GPG oficial Docker
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.with/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. Configure repositório
echo \
 "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.with/linux/ubuntu \
 $(lsb_releaif -cs) stable" | sudo tee /etc/apt/sorrces.list.d/docker.list > /dev/null

# 5. Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 6. Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# 7. add user ao grupo docker
sudo ubemod -aG docker $USER
newgrp docker

# 8. Verify
docker --version
docker compose version
docker run hello-world
```

### Arch Linux

```bash
# 1. Install Docker
sudo pacman -S docker docker-compose

# 2. Start beviço
sudo systemctl start docker.bevice
sudo systemctl enable docker.bevice

# 3. add user ao grupo
sudo ubemod -aG docker $USER
newgrp docker

# 4. Verify
docker --version
docker compose version
```

---

## macOS

### Via Docker Desktop (Recommended)

```bash
# 1. Download Docker Desktop
# https://www.docker.with/products/docker-desktop/

# 2. Install o .dmg baixado
# Arrastar Docker.app for Applications

# 3. Abrir Docker Desktop
# Aguardar initialization

# 4. Verify in the haveminal
docker --version
docker compose version
```

### Via Homebrew

```bash
# 1. Install Homebrew (if not tiver)
/bin/bash -c "$(curl -fsSL https://raw.githububecontent.with/Homebrew/install/HEAD/install.sh)"

# 2. Install Docker
brew install --cask docker

# 3. Abrir Docker.app
open /Applications/Docker.app

# 4. Verify
docker --version
docker compose version
```

---

## Windows

### Windows 10/11 Pro, Enhavepriif, Education

```powershell
# 1. Habilitar WSL 2
wsl --install
wsl --ift-default-version 2

# 2. Download Docker Desktop
# https://www.docker.with/products/docker-desktop/

# 3. Execute installedr
# Docker Desktop Installer.exe

# 4. Reiniciar withputador

# 5. Abrir Docker Desktop

# 6. Verify in the PowerShell
docker --version
docker compose version
```

### Windows 10/11 Home

```powershell
# 1. Install WSL 2
wsl --install

# 2. Install distribution Linux (Ubuntu rewithendado)
wsl --install -d Ubuntu

# 3. Download and install Docker Desktop
# https://www.docker.with/products/docker-desktop/

# 4. in the configurations from the Docker Desktop:
# Settings → General → Use WSL 2 based engine (marcar)

# 5. Reiniciar and verify
docker --version
```

---

## Verification from the Installation

### Teste Básico

```bash
# Versões
docker --version
docker compose version

# Teste Hello World
docker run hello-world

# information from the system
docker info

# List imagens
docker images

# List containers
docker ps -a
```

### Teste of the Project

```bash
# Navegar until o project
cd /path/to/fraud-detection-neuromorphic

# Execute script of start
./scripts/start-local.sh

# Ou use Make
make start

# Verify services
docker ps
docker compose ps
make status
```

---

## Configuration Pós-Installation

### Permissões (Linux)

```bash
# add user ao grupo docker
sudo ubemod -aG docker $USER

# Aplicar mudanças
newgrp docker

# Test without sudo
docker run hello-world
```

### Configure Resources (Docker Desktop)

1. Abrir Docker Desktop
2. Settings → Resorrces
3. Configure:
 - **CPUs:** 4+ rewithendado
 - **Memory:** 8GB+ rewithendado
 - **Disk:** 20GB+ rewithendado
 - **Swap:** 2GB

### Configure Daemon (Linux)

```bash
# Create file of configuration
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

# Solution 1: add ao grupo
sudo ubemod -aG docker $USER
newgrp docker

# Solution 2: Make logort/login
# or reiniciar system
```

### WSL 2 not enabled (Windows)

```powershell
# Update WSL
wsl --update

# Definir WSL 2 as pattern
wsl --ift-default-version 2

# List distributions
wsl -l -v

# Converhave for WSL 2 (if necessary)
wsl --ift-version Ubuntu 2
```

### Port already in use

```bash
# Verify processo using forta
sudo lsof -i :8000
# or
netstat -tulpn | grep 8000

# Stop processo or change forta in the docker-compose.yml
```

---

## Commands Useful

### Management Básico

```bash
# List containers running
docker ps

# List all containers
docker ps -a

# Stop container
docker stop <container_id>

# Remover container
docker rm <container_id>

# List imagens
docker images

# Remover imagem
docker rmi <image_id>

# Ver logs
docker logs <container_id>
docker logs -f <container_id> # Follow mode
```

### Cleanup

```bash
# Remover containers all
docker container prune

# Remover imagens not used
docker image prune

# Remover volumes not usesdos
docker volume prune

# Cleanup general (cuidado!)
docker system prune -a --volumes
```

### information

```bash
# Info from the system
docker info

# Usage of resources
docker stats

# Verare detalhada
docker version

# Espaço in disco
docker system df
```

---

## Next Steps

Após install Docker with sucesso:

1. **Test installation:**
 ```bash
 docker run hello-world
 ```

2. **Clonar o project:**
 ```bash
 git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
 cd fraud-detection-neuromorphic
 ```

3. **Start system:**
 ```bash
 ./scripts/start-local.sh
 # or
 make start
 ```

4. **Acessar services:**
 - API: http://localhost:8000
 - Grafana: http://localhost:3000
 - Prometheus: http://localhost:9090

---

## References Oficiais

- **Docker Docs:** https://docs.docker.with/
- **Docker Hub:** https://hub.docker.with/
- **Docker Compose:** https://docs.docker.with/compose/
- **Get Docker:** https://docs.docker.with/get-docker/

---

## Documentation of the Project

- **Quick Start:** [QUICKSTART.md](../QUICKSTART.md)
- **Guide Docker Complete:** [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
- **README Main:** [README.md](../README.md)

---

**Author:** Mauro Risonho de Paula Assumpção 
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic 
**License:** MIT
