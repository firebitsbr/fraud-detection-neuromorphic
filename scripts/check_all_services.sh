#!/bin/bash
#
# **Descrição:** Script para verificar o status de todos os serviços do projeto.
#
# **Autor:** Mauro Risonho de Paula Assumpção
# **Data de Criação:** 5 de Dezembro de 2025
# **Licença:** MIT License
# **Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
# - Claude Sonnet 4.5
# - Gemini 3 Pro Preview
#

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Verificação de Serviços - Neuromorphic Fraud Detection ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Função para testar endpoint HTTP
test_http() {
    local name=$1
    local url=$2
    local port=$3
    
    printf "${BLUE}%-20s${NC}" "$name:"
    
    if curl -s -f -m 3 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC} (${url})"
        return 0
    else
        echo -e "${RED}✗ FALHOU${NC} (não responde em ${url})"
        return 1
    fi
}

# Função para testar Redis
test_redis() {
    printf "${BLUE}%-20s${NC}" "Redis:"
    
    if docker exec fraud_redis redis-cli PING 2>/dev/null | grep -q PONG; then
        echo -e "${GREEN}✓ OK${NC} (localhost:6379)"
        return 0
    else
        echo -e "${RED}✗ FALHOU${NC} (localhost:6379)"
        return 1
    fi
}

# Contadores
total=0
success=0

echo -e "${YELLOW}Testando serviços...${NC}"
echo ""

# 1. Grafana
((total++))
if test_http "Grafana" "http://localhost:3000/api/health" "3000"; then ((success++)); fi

# 2. Prometheus
((total++))
if test_http "Prometheus" "http://localhost:9090/-/healthy" "9090"; then ((success++)); fi

# 3. Redis
((total++))
if test_redis; then ((success++)); fi

# 4. API Principal
((total++))
if test_http "API (FastAPI)" "http://localhost:8000/health" "8000"; then 
    ((success++))
    
    # Se API funciona, tenta pegar mais detalhes
    echo -e "${CYAN}   └─ Detalhes:${NC}"
    response=$(curl -s http://localhost:8000/health 2>/dev/null)
    echo "$response" | python3 -m json.tool 2>/dev/null | sed 's/^/      /' || echo "      $response"
fi

# 5. Loihi Simulator
((total++))
printf "${BLUE}%-20s${NC}" "Loihi Simulator:"
if docker ps --filter "name=fraud_loihi" --format "{{.Status}}" | grep -q "Up.*healthy"; then
    echo -e "${GREEN}✓ OK${NC} (localhost:8001 - simulador rodando)"
    ((success++))
else
    echo -e "${RED}✗ FALHOU${NC} (container não está healthy)"
fi

# 6. BrainScaleS Simulator
((total++))
printf "${BLUE}%-20s${NC}" "BrainScaleS:"
if docker ps --filter "name=fraud_brainscales" --format "{{.Status}}" | grep -q "Up.*healthy"; then
    echo -e "${GREEN}✓ OK${NC} (localhost:8002 - simulador rodando)"
    ((success++))
else
    echo -e "${RED}✗ FALHOU${NC} (container não está healthy)"
fi

# 7. Cluster Controller (pode não ter endpoint HTTP - é um benchmark)
((total++))
printf "${BLUE}%-20s${NC}" "Cluster Controller:"
status=$(docker ps -a --filter "name=fraud_cluster" --format "{{.Status}}")
if echo "$status" | grep -q "Restarting"; then
    # Verifica se completou benchmark com sucesso
    if docker logs fraud_cluster 2>&1 | grep -q "benchmark complete"; then
        echo -e "${GREEN}✓ OK${NC} (benchmark executado - exit normal)"
        ((success++))
    else
        echo -e "${YELLOW}⚠ RESTART LOOP${NC} ($status)"
    fi
elif docker ps --filter "name=fraud_cluster" --format "{{.Status}}" | grep -q "Up"; then
    echo -e "${GREEN}✓ RODANDO${NC} (container ativo)"
    ((success++))
else
    echo -e "${YELLOW}⚠ STOPPED${NC} ($status)"
fi

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Resumo                                                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

if [ $success -eq $total ]; then
    echo -e "${GREEN}✓ Todos os $total serviços estão funcionando!${NC}"
    exit 0
elif [ $success -gt 0 ]; then
    echo -e "${YELLOW}⚠ $success de $total serviços funcionando${NC}"
    echo -e "${YELLOW}  Serviços com problema: $((total - success))${NC}"
    exit 1
else
    echo -e "${RED}✗ Nenhum serviço está funcionando!${NC}"
    exit 2
fi
