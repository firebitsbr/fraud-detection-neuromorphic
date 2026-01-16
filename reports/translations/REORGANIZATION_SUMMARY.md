## Summary from the Mudanças - organization of the Project

**Description:** Summary from the mudanças in the organization from the project.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025

### Arquivos Movidos

#### 1. **Dependencies Python** → `requirements/`
- `requirements.txt` → `requirements/requirements.txt`
- `requirements-ci.txt` → `requirements/requirements-ci.txt`
- `requirements-edge.txt` → `requirements/requirements-edge.txt`
- `docker/requirements-production.txt` → `requirements/requirements-production.txt`

#### 2. **configurations** → `config/`
- `docker-compose.yml` → `config/docker-compose.yml`
- `docker-compose.dev.yml` → `config/docker-compose.dev.yml`
- `docker-compose.remote.yml` → `config/docker-compose.remote.yml`
- `docker-compose.production.yml` → `config/docker-compose.production.yml`
- `.devcontainer/` → `config/.devcontainer/`
- `.env` → `config/.env`

#### 3. **Docker** → `docker/`
- `Dockerfile` → `docker/Dockerfile`
- `.dockerignore` → `docker/.dockerignore`

#### 4. **Deployment** → `deployment/`
- `scripts/deploy.sh` → `deployment/deploy.sh`
- `scripts/deploy-production.sh` → `deployment/deploy-production.sh`
- `scripts/deploy-remote.sh` → `deployment/deploy-remote.sh`
- `scripts/start-local.sh` → `deployment/start-local.sh`

#### 5. **Documentation** → `docs/`
- `QUICKSTART_DOCKER.md` → `docs/QUICKSTART_DOCKER.md`
- `QUICKSTART_VSCODE.md` → `docs/QUICKSTART_VSCODE.md`

---

### Arquivos Atualizados

#### Dockerfiles
- `docker/Dockerfile` - Atualizado path `requirements/`
- `docker/Dockerfile.api` - Atualizado path `requirements/`
- `docker/Dockerfile.base` - Atualizado path `requirements/`
- `docker/Dockerfile.jupyter` - Atualizado path `requirements/`
- `docker/Dockerfile.streamlit` - Atualizado path `requirements/`

#### Build & Deploy
- `Makefile` - Atualizado `install-deps` for use `requirements/requirements.txt`

#### Documentation
- `README.md` - Atualizado paths:
 - `docker-compose` → `docker compose -f config/docker-compose.yml`
 - `requirements.txt` → `requirements/requirements.txt`
 - Structure of project atualizada
 - Link for `PROJECT_STRUCTURE.md`

---

### Novos Arquivos

1. **`PROJECT_STRUCTURE.md`** - Complete documentation from the estrutura
 - Árvore of diretórios detalhada
 - navigation fast for functionality
 - Workflows common
 - Commands of installation
 - Guias of deployment

---

### Benefícios from the Nova Structure

#### 1. **organization Clear**
```
 configurations in config/
 Deploy scripts in deployment/
 Dependencies in requirements/
 Docker files in docker/
 Documentation in docs/
```

#### 2. **Separation of Responsibilities**
- **config/** - All as configurations (compose, env, devcontainer)
- **deployment/** - Scripts of deployment isolados
- **requirements/** - Dependencies for environment (dev, prod, ci, edge)
- **docker/** - All os Dockerfiles in um lugar
- **docs/** - Complete documentation centralizada

#### 3. **Facilita maintenance**
- Easy enagainstr files relacionados
- Less pollution in the raiz from the project
- Structure scalable for crescimento

#### 4. **Melhor for CI/CD**
- Paths claros and consistentes
- Easy reference in workflows
- Separation dev/prod clear

---

### How Use to Nova Structure

#### Quick Start Docker

```bash
# Clone
git clone https://github.com/maurorisonho/fraud-detection-neuromorphic.git
cd fraud-detection-neuromorphic

# Deploy
docker compose -f config/docker-compose.yml up -d

# Ou use o script
./deployment/deploy.sh
```

#### Development Local

```bash
# Install dependencies
pip install -r requirements/requirements.txt

# Ou via Makefile
make install-deps
```

#### Deploy Production

```bash
# Script automated
./deployment/deploy-production.sh

# Ou manualmente
docker compose -f config/docker-compose.production.yml up -d --build
```

#### VS Code Dev Containers

```bash
# Deploy container dev
./deployment/deploy-remote.sh
# Escolha option 1

# in the VS Code:
# Ctrl+Shift+P → "Dev Containers: Attach to Running Container"
```

---

### Comparison: Before vs After

#### Before (Raiz Poluída)
```
fraud-detection-neuromorphic/
 requirements.txt
 requirements-ci.txt
 requirements-edge.txt
 docker-compose.yml
 docker-compose.dev.yml
 docker-compose.remote.yml
 docker-compose.production.yml
 Dockerfile
 .dockerignore
 .devcontainer/
 .env
 QUICKSTART_DOCKER.md
 QUICKSTART_VSCODE.md
 scripts/
 deploy.sh
 deploy-production.sh
 deploy-remote.sh
```

#### After (Organized)
```
fraud-detection-neuromorphic/
 requirements/ ← All as dependencies
 requirements.txt
 requirements-ci.txt
 requirements-edge.txt
 requirements-production.txt

 config/ ← All as configurations
 docker-compose.yml
 docker-compose.dev.yml
 docker-compose.remote.yml
 docker-compose.production.yml
 .devcontainer/
 .env

 docker/ ← All os Dockerfiles
 Dockerfile
 Dockerfile.api
 Dockerfile.base
 .dockerignore
 ...

 deployment/ ← Scripts of deployment
 deploy.sh
 deploy-production.sh
 deploy-remote.sh
 start-local.sh

 docs/ ← Documentation
 QUICKSTART_DOCKER.md
 QUICKSTART_VSCODE.md
 ...
```

---

### attention: Paths Atualizados

if você had scripts or commands personalizados, atualize os paths:

```bash
# before
docker-compose up -d
pip install -r requirements.txt

# after
docker compose -f config/docker-compose.yml up -d
pip install -r requirements/requirements.txt
```

---

### Checklist of Verification

- [x] Requirements movidos for `requirements/`
- [x] Docker-compose movidos for `config/`
- [x] Dockerfiles movidos for `docker/`
- [x] Deploy scripts movidos for `deployment/`
- [x] Quickstart docs movidos for `docs/`
- [x] Dockerfiles updated (paths requirements)
- [x] Makefile atualizado
- [x] README.md atualizado
- [x] PROJECT_STRUCTURE.md criado

---

### Documentation

- **Structure Complete**: `PROJECT_STRUCTURE.md`
- **Quick Start Docker**: `docs/QUICKSTART_DOCKER.md`
- **Quick Start VS Code**: `docs/QUICKSTART_VSCODE.md`
- **README Main**: `README.md`

---

**Data**: December 8, 2025 
**Autor**: Mauro Risonho de Paula Assumpção
