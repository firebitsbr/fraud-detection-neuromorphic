Quero que você atue como gerador automático de projetos completos de portfólio em Computação Neuromórfica aplicada à Cybersecurity Bancária e Fintechs.
Seu objetivo é criar todos os projetos, com código, documentação, Docker, notebooks e exemplos reais.

Use o meu nome: Mauro Risonho de Paula Assumpção e prepare o portfólio para publicação no GitHub.

🎯 OBJETIVO GERAL

Crie 10 projetos completos, cada um representando um caso de uso diferente de Computação Neuromórfica aplicada à Cybersecurity em bancos e fintechs, incluindo:

1-Detecção de Fraude Neuromórfica
2-Cognitive Firewall contra Engenharia Social
3-Monitoramento Temporal de SWIFT/STR/Pix
4-Neuromorphic SIEM
5-Detecção de Malware Bancário (Guildma/Astaroth etc.)
6-EDR neuromórfico para Mobile Banking
7-Proteção de APIs Open Finance
8-UEBA Neuromórfico
9-AML Neuromórfico
10-Zero-Trust Cognitivo em Tempo Real

Cada projeto deve ser criado como um repositório independente dentro de uma pasta /portfolio.

📁 ESTRUTURA DOS PROJETOS (igual para todos)

Para cada caso de uso, gerar automaticamente:

/portfolio/
   /01_fraud_neuromorphic/
       README.md
       src/
          main.py
          encoders.py
          models_snn.py
       notebooks/
          demo.ipynb
          stdp_example.ipynb
       docker/
          Dockerfile
          requirements.txt
       docs/
          architecture.png
          explanation.md

📘 REQUISITOS DE CADA PROJETO
1. README.md

Inclua:

Descrição do caso de uso

Arquitetura neuromórfica

Tecnologias: Lava, NEST, Brian2, PyTorch

Como rodar (Docker + local)

Explicação técnica + executiva

Créditos ao autor: Mauro Risonho de Paula Assumpção

2. SRC / Código Python Completo

Cada projeto deve incluir:

main.py com o pipeline

encoders.py para codificação de spikes

models_snn.py com SNN usando Brian2 ou NEST

scripts de validação

integração com PyTorch para pré-processamento

3. NOTEBOOKS

Criar dois notebooks por projeto:

demo.ipynb — demonstração funcional

stdp_example.ipynb — aprendizado neuromórfico com STDP

Os notebooks devem rodar mesmo sem GPU.

4. DOCKER

Gerar:

Dockerfile contendo

Python 3.10

Lava

NEST

Brian2

JupyterLab

PyTorch (CPU)

requirements.txt

comando de execução automática

5. DOCUMENTAÇÃO

Criar:

Arquitetura visual (ASCII + PNG)

Explicação de como funciona o mecanismo neuromórfico

Diagramas de fluxo

Explicação do caso de uso para bancos e fintechs

📌 OUTRAS INSTRUÇÕES IMPORTANTES
→ Trabalhe projeto por projeto

Antes de gerar todo o portfólio, pergunte:

“Qual dos 10 projetos você deseja gerar primeiro?”

Depois gere a pasta inteira.

→ Qualidade profissional

Os projetos devem parecer criados por um:

Engenheiro de IA

Pesquisador em Neuromorphic Computing

Profissional de Cybersecurity

Especialista em Bancos e Fintechs

→ Não usar texto genérico

Toda documentação deve ser específica, técnica e profunda.

→ Preparar para GitHub

As pastas criadas devem estar prontas para:

git init
git add .
git commit -m "Projeto Neuromórfico X"
