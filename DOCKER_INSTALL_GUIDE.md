# 🐳 Guia de Instalação do Docker

**Autor:** Mauro Risonho de Paula Assumpção  
**Data:** 5 de Dezembro de 2025  
**Licença:** MIT License

Este guia mostra como instalar Docker e Docker Compose em diferentes sistemas operacionais.

---

## 🐧 Linux

### Fedora / RHEL / CentOS

```bash
# 1. Remover versões antigas (se existirem)
sudo dnf remove docker \
    docker-client \
    docker-client-latest \
    docker-common \
    docker-latest \
    docker-latest-logrotate \
    docker-logrotate \
    docker-selinux \
    docker-engine-selinux \
    docker-engine

# 2. Adicionar repositório oficial Docker
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo

# 3. Instalar Docker Engine
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 4. Iniciar e habilitar Docker
sudo systemctl start docker
sudo systemctl enable docker

# 5. Adicionar seu usuário ao grupo docker (evita usar sudo)
sudo usermod -aG docker $USER

# 6. Aplicar mudanças de grupo (ou faça logout/login)
newgrp docker

# 7. Verificar instalação
docker --version
docker compose version
docker run hello-world
```

### Ubuntu / Debian

```bash
# 1. Atualizar pacotes
sudo apt-get update

# 2. Instalar dependências
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 3. Adicionar chave GPG oficial Docker
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. Configurar repositório
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. Instalar Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 6. Iniciar Docker
sudo systemctl start docker
sudo systemctl enable docker

# 7. Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
newgrp docker

# 8. Verificar
docker --version
docker compose version
docker run hello-world
```

### Arch Linux

```bash
# 1. Instalar Docker
sudo pacman -S docker docker-compose

# 2. Iniciar serviço
sudo systemctl start docker.service
sudo systemctl enable docker.service

# 3. Adicionar usuário ao grupo
sudo usermod -aG docker $USER
newgrp docker

# 4. Verificar
docker --version
docker compose version
```

---

## 🍎 macOS

### Via Docker Desktop (Recomendado)

```bash
# 1. Baixar Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 2. Instalar o .dmg baixado
# Arrastar Docker.app para Applications

# 3. Abrir Docker Desktop
# Aguardar inicialização

# 4. Verificar no terminal
docker --version
docker compose version
```

### Via Homebrew

```bash
# 1. Instalar Homebrew (se não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Instalar Docker
brew install --cask docker

# 3. Abrir Docker.app
open /Applications/Docker.app

# 4. Verificar
docker --version
docker compose version
```

---

## 🪟 Windows

### Windows 10/11 Pro, Enterprise, Education

```powershell
# 1. Habilitar WSL 2
wsl --install
wsl --set-default-version 2

# 2. Baixar Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 3. Executar instalador
# Docker Desktop Installer.exe

# 4. Reiniciar computador

# 5. Abrir Docker Desktop

# 6. Verificar no PowerShell
docker --version
docker compose version
```

### Windows 10/11 Home

```powershell
# 1. Instalar WSL 2
wsl --install

# 2. Instalar distribuição Linux (Ubuntu recomendado)
wsl --install -d Ubuntu

# 3. Baixar e instalar Docker Desktop
# https://www.docker.com/products/docker-desktop/

# 4. Nas configurações do Docker Desktop:
# Settings → General → Use WSL 2 based engine (marcar)

# 5. Reiniciar e verificar
docker --version
```

---

## ✅ Verificação da Instalação

### Teste Básico

```bash
# Versões
docker --version
docker compose version

# Teste Hello World
docker run hello-world

# Informações do sistema
docker info

# Listar imagens
docker images

# Listar containers
docker ps -a
```

### Teste do Projeto

```bash
# Navegar até o projeto
cd /path/to/fraud-detection-neuromorphic

# Executar script de início
./start-local.sh

# Ou usar Make
make start

# Verificar serviços
docker ps
docker compose ps
make status
```

---

## 🔧 Configuração Pós-Instalação

### Permissões (Linux)

```bash
# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER

# Aplicar mudanças
newgrp docker

# Testar sem sudo
docker run hello-world
```

### Configurar Recursos (Docker Desktop)

1. Abrir Docker Desktop
2. Settings → Resources
3. Configurar:
   - **CPUs:** 4+ recomendado
   - **Memory:** 8GB+ recomendado
   - **Disk:** 20GB+ recomendado
   - **Swap:** 2GB

### Configurar Daemon (Linux)

```bash
# Criar arquivo de configuração
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
sudo systemctl restart docker
```

---

## 🐛 Troubleshooting

### Docker não inicia (Linux)

```bash
# Verificar status
sudo systemctl status docker

# Ver logs
sudo journalctl -u docker.service

# Reiniciar serviço
sudo systemctl restart docker

# Verificar se está habilitado
sudo systemctl enable docker
```

### Permissão negada

```bash
# Erro: "permission denied while trying to connect to the Docker daemon socket"

# Solução 1: Adicionar ao grupo
sudo usermod -aG docker $USER
newgrp docker

# Solução 2: Fazer logout/login
# ou reiniciar sistema
```

### WSL 2 não encontrado (Windows)

```powershell
# Atualizar WSL
wsl --update

# Definir WSL 2 como padrão
wsl --set-default-version 2

# Listar distribuições
wsl -l -v

# Converter para WSL 2 (se necessário)
wsl --set-version Ubuntu 2
```

### Porta já em uso

```bash
# Verificar processo usando porta
sudo lsof -i :8000
# ou
netstat -tulpn | grep 8000

# Parar processo ou mudar porta no docker-compose.yml
```

---

## 🔍 Comandos Úteis

### Gerenciamento Básico

```bash
# Listar containers rodando
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
docker logs -f <container_id>  # Follow mode
```

### Limpeza

```bash
# Remover containers parados
docker container prune

# Remover imagens não usadas
docker image prune

# Remover volumes não usados
docker volume prune

# Limpeza geral (cuidado!)
docker system prune -a --volumes
```

### Informações

```bash
# Info do sistema
docker info

# Uso de recursos
docker stats

# Versão detalhada
docker version

# Espaço em disco
docker system df
```

---

## 📚 Próximos Passos

Após instalar Docker com sucesso:

1. **Testar instalação:**
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
   ./start-local.sh
   # ou
   make start
   ```

4. **Acessar serviços:**
   - API: http://localhost:8000
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

---

## 🔗 Referências Oficiais

- **Docker Docs:** https://docs.docker.com/
- **Docker Hub:** https://hub.docker.com/
- **Docker Compose:** https://docs.docker.com/compose/
- **Get Docker:** https://docs.docker.com/get-docker/

---

## 📖 Documentação do Projeto

- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Guia Docker Completo:** [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
- **README Principal:** [README.md](README.md)

---

**Autor:** Mauro Risonho de Paula Assumpção  
**Repositório:** https://github.com/maurorisonho/fraud-detection-neuromorphic  
**Licença:** MIT
