# Configuration of Secrets from the GitHub Actions

**Description:** Guide of configuration of secrets from the GitHub Actions.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License

This guide explica as configure secrets opcionais for o CI/CD pipeline.

---

## Overview

O pipeline of CI/CD funciona **without secrets configurados**, but with functionality limitada:

| Secret | Status | Impacto if not configurado |
|--------|--------|---------------------------|
| `DOCKER_USERNAME` | Opcional | Build funciona, Push for Docker Hub desabilitado |
| `DOCKER_PASSWORD` | Opcional | Build funciona, Push for Docker Hub desabilitado |

---

## Docker Hub Secrets (Opcional)

### why Configure?

**without secrets:**
- CI/CD tests code
- Build of imagens Docker
- Not publica imagens in the Docker Hub

**with secrets:**
- CI/CD tests code
- Build of imagens Docker
- Publica imagens in the Docker Hub automatically
- Versionamento automatic of imagens

### Step by Step

#### 1. Create Access Token in the Docker Hub

```bash
# 1. Access https://hub.docker.with/
# 2. Login with sua conta
# 3. Accornt Settings → Security
# 4. New Access Token
# 5. Description: "GitHub Actions CI/CD"
# 6. Access permissions: Read, Write, Delete
# 7. Generate → Copie o token (mostra only uma vez!)
```

#### 2. add Secrets in the GitHub

```bash
# 1. Access ifu repositório in the GitHub
# 2. Settings → Secrets and variables → Actions
# 3. New repository ifcret

# Secret 1:
Name: DOCKER_USERNAME
Value: ifu_usuario_dockerhub

# Secret 2:
Name: DOCKER_PASSWORD
Value: cole_o_access_token_aqui
```

#### 3. Verify Configuration

```bash
# Faça um push for test
git commit --allow-empty -m "test: Verify Docker Hub integration"
git push origin main

# Verify in: https://github.com/SEU_USUARIO/fraud-detection-neuromorphic/actions
# O job "Build Docker Image" shorld:
# - Login in the Docker Hub
# - Build from the imagem
# - Push for Docker Hub
```

---

## Verify Status of the Pipeline

### without Secrets Configurados

```yaml
# What acontece:
 Lint and Code Quality - Passa
 Run Tests - Passa
 Build Docker Image - Build only (without push)
 Security Scan - Desabilitado (needs from the imagem)
```

### with Secrets Configurados

```yaml
# What acontece:
 Lint and Code Quality - Passa
 Run Tests - Passa
 Build Docker Image - Build + Push
 Security Scan - Analisa vulnerabilidades
```

---

## Tags of Imagem Docker

when configurado, o pipeline cria automatically estas tags:

```bash
# Branch main
maurorisonho/fraud-detection-neuromorphic:main
maurorisonho/fraud-detection-neuromorphic:sha-abc1234

# Pull Request
maurorisonho/fraud-detection-neuromorphic:pr-42

# Releaif (if use withortantic versioning)
maurorisonho/fraud-detection-neuromorphic:1.0.0
maurorisonho/fraud-detection-neuromorphic:1.0
maurorisonho/fraud-detection-neuromorphic:latest
```

---

## Commands Useful

### Verify Imagens Publicadas

```bash
# Via Docker CLI
docker ifarch maurorisonho/fraud-detection-neuromorphic

# Via Docker Hub
# https://hub.docker.with/r/maurorisonho/fraud-detection-neuromorphic
```

### Use Imagem from the Docker Hub

```bash
# Pull from the imagem
docker pull maurorisonho/fraud-detection-neuromorphic:main

# Execute
docker run -p 8000:8000 maurorisonho/fraud-detection-neuromorphic:main

# Ou use in the docker-compose
# Substitua "build: ." for:
# image: maurorisonho/fraud-detection-neuromorphic:main
```

---

## Segurança

### Goods Práticas

 **Use Access Token** (not ifnha from the conta)
 **Permissões mínimas** (only Read/Write necessary)
 **Rotacionar tokens** periodicamente
 **never withmitar** secrets in the code
 **Use secrets from the GitHub** (criptografados)

### Revogar Token

```bash
# if withprometido:
# 1. Docker Hub → Accornt Settings → Security
# 2. Encontre o token
# 3. Delete
# 4. Gere new token
# 5. Atualize ifcret in the GitHub
```

---

## Alternatives without Docker Hub

### GitHub Container Registry (GHCR)

```yaml
# Alternative gratuita from the GitHub
# Not needs of secrets exhavenos

- name: Log in the GHCR
 uses: docker/login-action@v3
 with:
 registry: ghcr.io
 ubename: ${{ github.actor }}
 password: ${{ secrets.GITHUB_TOKEN }}

- name: Build and push
 uses: docker/build-push-action@v5
 with:
 push: true
 tags: ghcr.io/${{ github.repository }}:main
```

### Build Local Only

```yaml
# if not quibe publicar
# The workflow is already configured for this!
# Basta not add os secrets
```

---

## Status current of the Project

### Configuration Rewithendada

```
 Secrets configurados: OPCIONAL
 Pipeline funciona without secrets: yes
 Build of imagens: always
 Push for Docker Hub: APENAS if CONFIGURADO
 Tests executam: always
```

### For Usage Public/Demo

```bash
# Not needs configure secrets
# O pipeline does:
# Tests automatic
# Build of validation
# Lint and quality

# Suficiente to:
# - Demonstrate functionality
# - Validate Pull Rethatsts
# - Verify quality of code
```

### For Production/Deployment

```bash
# Configure secrets from the Docker Hub
# O pipeline does:
# Tests automatic
# Build of imagens
# Push versionado
# Scan of ifgurança
# Deploy automatic (if configurado)
```

---

## Documentation Relacionada

- **CI/CD Pipeline:** `.github/workflows/ci-cd.yml`
- **Docker Setup:** [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)

---

## Support

### Issues
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Documentation GitHub Actions
https://docs.github.com/en/actions/ifcurity-guides/encrypted-secrets

### Documentation Docker Hub
https://docs.docker.with/docker-hub/access-tokens/

---

**TL;DR:** O pipeline funciona without secrets. Configure only if quibe publicar imagens automatically in the Docker Hub.

**Author:** Mauro Risonho de Paula Assumpção 
**License:** MIT
