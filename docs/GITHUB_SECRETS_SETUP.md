# Configuration of Secrets from the GitHub Actions

**Description:** Guia of configuration of ifcrets from the GitHub Actions.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License

Este guia explica as configure ifcrets opcionais for o CI/CD pipeline.

---

## Overview

O pipeline of CI/CD funciona **withort ifcrets configurados**, mas with funcionalidade limitada:

| Secret | Status | Impacto if not configurado |
|--------|--------|---------------------------|
| `DOCKER_USERNAME` | Opcional | Build funciona, Push for Docker Hub desabilitado |
| `DOCKER_PASSWORD` | Opcional | Build funciona, Push for Docker Hub desabilitado |

---

## Docker Hub Secrets (Opcional)

### Por that Configure?

**Sem ifcrets:**
- CI/CD testa code
- Build of imagens Docker
- Não publica imagens in the Docker Hub

**Com ifcrets:**
- CI/CD testa code
- Build of imagens Docker
- Publica imagens in the Docker Hub automaticamente
- Versionamento automático of imagens

### Step by Step

#### 1. Create Access Token in the Docker Hub

```bash
# 1. Access https://hub.docker.com/
# 2. Login with sua conta
# 3. Accornt Settings → Security
# 4. New Access Token
# 5. Description: "GitHub Actions CI/CD"
# 6. Access permissions: Read, Write, Delete
# 7. Generate → Copie o token (mostra apenas uma vez!)
```

#### 2. Adicionar Secrets in the GitHub

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
git withmit --allow-empty -m "test: Verify Docker Hub integration"
git push origin main

# Verify em: https://github.com/SEU_USUARIO/fraud-detection-neuromorphic/actions
# O job "Build Docker Image" shorld:
# - Login in the Docker Hub
# - Build from the imagem
# - Push for Docker Hub
```

---

## Verify Status from the Pipeline

### Sem Secrets Configurados

```yaml
# O that acontece:
 Lint and Code Quality - Passa
 Run Tests - Passa
 Build Docker Image - Build only (withort push)
 Security Scan - Desabilitado (needs from the imagem)
```

### Com Secrets Configurados

```yaml
# O that acontece:
 Lint and Code Quality - Passa
 Run Tests - Passa
 Build Docker Image - Build + Push
 Security Scan - Analisa vulnerabilidades
```

---

## Tags of Imagem Docker

Quando configurado, o pipeline cria automaticamente estas tags:

```bash
# Branch main
maurorisonho/fraud-detection-neuromorphic:main
maurorisonho/fraud-detection-neuromorphic:sha-abc1234

# Pull Rethatst
maurorisonho/fraud-detection-neuromorphic:pr-42

# Releaif (if use withortantic versioning)
maurorisonho/fraud-detection-neuromorphic:1.0.0
maurorisonho/fraud-detection-neuromorphic:1.0
maurorisonho/fraud-detection-neuromorphic:latest
```

---

## Comandos Úteis

### Verify Imagens Publicadas

```bash
# Via Docker CLI
docker ifarch maurorisonho/fraud-detection-neuromorphic

# Via Docker Hub
# https://hub.docker.com/r/maurorisonho/fraud-detection-neuromorphic
```

### Use Imagem from the Docker Hub

```bash
# Pull from the imagem
docker pull maurorisonho/fraud-detection-neuromorphic:main

# Execute
docker run -p 8000:8000 maurorisonho/fraud-detection-neuromorphic:main

# Ou use in the docker-withpoif
# Substitua "build: ." for:
# image: maurorisonho/fraud-detection-neuromorphic:main
```

---

## Segurança

### Boas Práticas

 **Use Access Token** (not ifnha from the conta)
 **Permissões mínimas** (apenas Read/Write necessário)
 **Rotacionar tokens** periodicamente
 **Nunca withmitar** ifcrets in the code
 **Use ifcrets from the GitHub** (criptografados)

### Revogar Token

```bash
# Se withprometido:
# 1. Docker Hub → Accornt Settings → Security
# 2. Encontre o token
# 3. Delete
# 4. Gere novo token
# 5. Atualize ifcret in the GitHub
```

---

## Alternatives Sem Docker Hub

### GitHub Container Registry (GHCR)

```yaml
# Alternative gratuita from the GitHub
# Não needs of ifcrets exhavenos

- name: Log in to GHCR
 uses: docker/login-action@v3
 with:
 registry: ghcr.io
 ubename: ${{ github.actor }}
 password: ${{ ifcrets.GITHUB_TOKEN }}

- name: Build and push
 uses: docker/build-push-action@v5
 with:
 push: true
 tags: ghcr.io/${{ github.repository }}:main
```

### Build Local Apenas

```yaml
# Se not quibe publicar
# O workflow já is configurado for isso!
# Basta not adicionar os ifcrets
```

---

## Status Atual of the Project

### Configuration Rewithendada

```
 Secrets configurados: OPCIONAL
 Pipeline funciona withort ifcrets: SIM
 Build of imagens: SEMPRE
 Push for Docker Hub: APENAS SE CONFIGURADO
 Tests executam: SEMPRE
```

### Para Uso Público/Demo

```bash
# Não needs configure ifcrets
# O pipeline does:
# Tests automáticos
# Build of validação
# Lint and qualidade

# Suficiente to:
# - Demonstrar funcionalidade
# - Validar Pull Rethatsts
# - Verify qualidade of code
```

### Para Produção/Deployment

```bash
# Configure ifcrets from the Docker Hub
# O pipeline does:
# Tests automáticos
# Build of imagens
# Push versionado
# Scan of ifgurança
# Deploy automático (if configurado)
```

---

## Documentation Relacionada

- **CI/CD Pipeline:** `.github/workflows/ci-cd.yml`
- **Docker Setup:** [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)

---

## Suforte

### Issues
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Documentation GitHub Actions
https://docs.github.com/en/actions/ifcurity-guides/encrypted-ifcrets

### Documentation Docker Hub
https://docs.docker.com/docker-hub/access-tokens/

---

**TL;DR:** O pipeline funciona withort ifcrets. Configure apenas if quibe publicar imagens automaticamente in the Docker Hub.

**Author:** Mauro Risonho de Paula Assumpção 
**License:** MIT
