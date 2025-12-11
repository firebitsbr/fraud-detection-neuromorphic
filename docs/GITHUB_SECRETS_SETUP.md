# 🔐 Configuração de Secrets do GitHub Actions

**Descrição:** Guia de configuração de secrets do GitHub Actions.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License

Este guia explica como configurar secrets opcionais para o CI/CD pipeline.

---

## 📋 Visão Geral

O pipeline de CI/CD funciona **sem secrets configurados**, mas com funcionalidade limitada:

| Secret | Status | Impacto se não configurado |
|--------|--------|---------------------------|
| `DOCKER_USERNAME` | ⚠️ Opcional | ✅ Build funciona, ❌ Push para Docker Hub desabilitado |
| `DOCKER_PASSWORD` | ⚠️ Opcional | ✅ Build funciona, ❌ Push para Docker Hub desabilitado |

---

## 🐳 Docker Hub Secrets (Opcional)

### Por que Configurar?

**Sem secrets:**
- ✅ CI/CD testa código
- ✅ Build de imagens Docker
- ❌ Não publica imagens no Docker Hub

**Com secrets:**
- ✅ CI/CD testa código
- ✅ Build de imagens Docker
- ✅ Publica imagens no Docker Hub automaticamente
- ✅ Versionamento automático de imagens

### Passo a Passo

#### 1. Criar Access Token no Docker Hub

```bash
# 1. Acesse https://hub.docker.com/
# 2. Login com sua conta
# 3. Account Settings → Security
# 4. New Access Token
# 5. Description: "GitHub Actions CI/CD"
# 6. Access permissions: Read, Write, Delete
# 7. Generate → Copie o token (mostra apenas uma vez!)
```

#### 2. Adicionar Secrets no GitHub

```bash
# 1. Acesse seu repositório no GitHub
# 2. Settings → Secrets and variables → Actions
# 3. New repository secret

# Secret 1:
Name: DOCKER_USERNAME
Value: seu_usuario_dockerhub

# Secret 2:
Name: DOCKER_PASSWORD
Value: cole_o_access_token_aqui
```

#### 3. Verificar Configuração

```bash
# Faça um push para testar
git commit --allow-empty -m "test: Verify Docker Hub integration"
git push origin main

# Verifique em: https://github.com/SEU_USUARIO/fraud-detection-neuromorphic/actions
# O job "Build Docker Image" deve:
# - ✅ Login no Docker Hub
# - ✅ Build da imagem
# - ✅ Push para Docker Hub
```

---

## 🔍 Verificar Status do Pipeline

### Sem Secrets Configurados

```yaml
# O que acontece:
✅ Lint and Code Quality - Passa
✅ Run Tests - Passa
⚠️  Build Docker Image - Build only (sem push)
⚠️  Security Scan - Desabilitado (precisa da imagem)
```

### Com Secrets Configurados

```yaml
# O que acontece:
✅ Lint and Code Quality - Passa
✅ Run Tests - Passa
✅ Build Docker Image - Build + Push
✅ Security Scan - Analisa vulnerabilidades
```

---

## 🎯 Tags de Imagem Docker

Quando configurado, o pipeline cria automaticamente estas tags:

```bash
# Branch main
maurorisonho/fraud-detection-neuromorphic:main
maurorisonho/fraud-detection-neuromorphic:sha-abc1234

# Pull Request
maurorisonho/fraud-detection-neuromorphic:pr-42

# Release (se usar semantic versioning)
maurorisonho/fraud-detection-neuromorphic:1.0.0
maurorisonho/fraud-detection-neuromorphic:1.0
maurorisonho/fraud-detection-neuromorphic:latest
```

---

## 🛠️ Comandos Úteis

### Verificar Imagens Publicadas

```bash
# Via Docker CLI
docker search maurorisonho/fraud-detection-neuromorphic

# Via Docker Hub
# https://hub.docker.com/r/maurorisonho/fraud-detection-neuromorphic
```

### Usar Imagem do Docker Hub

```bash
# Pull da imagem
docker pull maurorisonho/fraud-detection-neuromorphic:main

# Executar
docker run -p 8000:8000 maurorisonho/fraud-detection-neuromorphic:main

# Ou usar no docker-compose
# Substitua "build: ." por:
# image: maurorisonho/fraud-detection-neuromorphic:main
```

---

## 🔒 Segurança

### Boas Práticas

✅ **Usar Access Token** (não senha da conta)
✅ **Permissões mínimas** (apenas Read/Write necessário)
✅ **Rotacionar tokens** periodicamente
✅ **Nunca commitar** secrets no código
✅ **Usar secrets do GitHub** (criptografados)

### Revogar Token

```bash
# Se comprometido:
# 1. Docker Hub → Account Settings → Security
# 2. Encontre o token
# 3. Delete
# 4. Gere novo token
# 5. Atualize secret no GitHub
```

---

## 🚫 Alternativas Sem Docker Hub

### GitHub Container Registry (GHCR)

```yaml
# Alternativa gratuita do GitHub
# Não precisa de secrets externos

- name: Log in to GHCR
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}

- name: Build and push
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: ghcr.io/${{ github.repository }}:main
```

### Build Local Apenas

```yaml
# Se não quiser publicar
# O workflow já está configurado para isso!
# Basta não adicionar os secrets
```

---

## 📊 Status Atual do Projeto

### Configuração Recomendada

```
✅ Secrets configurados: OPCIONAL
✅ Pipeline funciona sem secrets: SIM
✅ Build de imagens: SEMPRE
✅ Push para Docker Hub: APENAS SE CONFIGURADO
✅ Testes executam: SEMPRE
```

### Para Uso Público/Demo

```bash
# Não precisa configurar secrets
# O pipeline faz:
# ✅ Testes automáticos
# ✅ Build de validação
# ✅ Lint e qualidade

# Suficiente para:
# - Demonstrar funcionalidade
# - Validar Pull Requests
# - Verificar qualidade de código
```

### Para Produção/Deployment

```bash
# Configure secrets do Docker Hub
# O pipeline faz:
# ✅ Testes automáticos
# ✅ Build de imagens
# ✅ Push versionado
# ✅ Scan de segurança
# ✅ Deploy automático (se configurado)
```

---

## 🎓 Documentação Relacionada

- **CI/CD Pipeline:** `.github/workflows/ci-cd.yml`
- **Docker Setup:** [DOCKER_LOCAL_SETUP.md](DOCKER_LOCAL_SETUP.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)

---

## 📞 Suporte

### Issues
https://github.com/maurorisonho/fraud-detection-neuromorphic/issues

### Documentação GitHub Actions
https://docs.github.com/en/actions/security-guides/encrypted-secrets

### Documentação Docker Hub
https://docs.docker.com/docker-hub/access-tokens/

---

**TL;DR:** O pipeline funciona sem secrets. Configure apenas se quiser publicar imagens automaticamente no Docker Hub.

**Autor:** Mauro Risonho de Paula Assumpção  
**Licença:** MIT
