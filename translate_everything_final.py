#!/usr/bin/env python3
"""
TRADUÇÃO DEFINITIVA - Traduz ABSOLUTAMENTE TUDO do português para inglês
"""
import re
from pathlib import Path
import sys

# Dicionário COMPLETO de TODAS as palavras portuguesas
PORTUGUESE_ENGLISH = {
    # Substantivos comuns
    r'\bImplementação\b': 'Implementation',
    r'\bimplementação\b': 'implementation',
    r'\bResumo\b': 'Summary',
    r'\bresumo\b': 'summary',
    r'\bAnálise\b': 'Analysis',
    r'\banálise\b': 'analysis',
    r'\bAnáliif\b': 'Analysis',
    r'\banáliif\b': 'analysis',
    r'\bCrítica\b': 'Critical',
    r'\bcrítica\b': 'critical',
    r'\bVisão\b': 'Vision',
    r'\bvisão\b': 'vision',
    r'\bGeral\b': 'General',
    r'\bgeral\b': 'general',
    r'\bGuia\b': 'Guide',
    r'\bguia\b': 'guide',
    r'\bInstalação\b': 'Installation',
    r'\binstalação\b': 'installation',
    r'\bConfiguração\b': 'Configuration',
    r'\bconfiguração\b': 'configuration',
    r'\bAutomação\b': 'Automation',
    r'\bautomação\b': 'automation',
    r'\bGerenciamento\b': 'Management',
    r'\bgerenciamento\b': 'management',
    r'\bVisualização\b': 'Visualization',
    r'\bvisualization\b': 'visualization',
    r'\bLimpeza\b': 'Cleanup',
    r'\blimpeza\b': 'cleanup',
    r'\bMitigação\b': 'Mitigation',
    r'\bmitigação\b': 'mitigation',
    r'\bMatriz\b': 'Matrix',
    r'\bmatriz\b': 'matrix',
    r'\bRiscos\b': 'Risks',
    r'\briscos\b': 'risks',
    r'\bRewithmendations\b': 'Recommendations',
    r'\brewithmendations\b': 'recommendations',
    r'\bRecomendações\b': 'Recommendations',
    r'\brecomendações\b': 'recommendations',
    r'\bPrioritárias\b': 'Priority',
    r'\bprioritárias\b': 'priority',
    r'\bArquitetura\b': 'Architecture',
    r'\barquitetura\b': 'architecture',
    r'\bTécnica\b': 'Technical',
    r'\btécnica\b': 'technical',
    r'\bSólida\b': 'Solid',
    r'\bsólida\b': 'solid',
    r'\bPontos\b': 'Points',
    r'\bpontos\b': 'points',
    r'\bFortes\b': 'Strong',
    r'\bfortes\b': 'strong',
    r'\bIdentistaysdos\b': 'Identified',
    r'\bidentistaysdos\b': 'identified',
    r'\bIdentificados\b': 'Identified',
    r'\bidentificados\b': 'identified',
    r'\bSetoção\b': 'Separation',
    r'\bsetoção\b': 'separation',
    r'\bSeparação\b': 'Separation',
    r'\bseparação\b': 'separation',
    r'\bresponsabilidades\b': 'responsibilities',
    r'\bResponsabilidades\b': 'Responsibilities',
    r'\bEvidências\b': 'Evidence',
    r'\bevidências\b': 'evidence',
    r'\bestruturado\b': 'structured',
    r'\bEstruturado\b': 'Structured',
    r'\borganizado\b': 'organized',
    r'\bOrganizado\b': 'Organized',
    r'\bunitários\b': 'unit',
    r'\bUnitários\b': 'Unit',
    r'\bpreifntes\b': 'present',
    r'\bPreifntes\b': 'Present',
    r'\bpresentes\b': 'present',
    r'\bPresentes\b': 'Present',
    r'\bDocumentação\b': 'Documentation',
    r'\bdocumentação\b': 'documentation',
    r'\bmanutenibilidade\b': 'maintainability',
    r'\bManutenibilidade\b': 'Maintainability',
    
    # Adjetivos e descrições
    r'\bCompleta\b': 'Complete',
    r'\bcompleta\b': 'complete',
    r'\bPrincipal\b': 'Main',
    r'\bprincipal\b': 'main',
    r'\borthatstrados\b': 'orchestrated',
    r'\bOrthatsrados\b': 'Orchestrated',
    r'\bautomáticos\b': 'automatic',
    r'\bAutomáticos\b': 'Automatic',
    r'\bpersistentes\b': 'persistent',
    r'\bPersistentes\b': 'Persistent',
    r'\bisolada\b': 'isolated',
    r'\bIsolada\b': 'Isolated',
    r'\bautomatizada\b': 'automated',
    r'\bAutomatizada\b': 'Automated',
    r'\bcolorida\b': 'colored',
    r'\bColorida\b': 'Colored',
    r'\bSimplistaysdos\b': 'Simplified',
    r'\bsimplistaysdos\b': 'simplified',
    r'\bSimplificados\b': 'Simplified',
    r'\bsimplificados\b': 'simplified',
    r'\búteis\b': 'useful',
    r'\bÚteis\b': 'Useful',
    r'\bmodular\b': 'modular',
    r'\bModular\b': 'Modular',
    r'\bextensível\b': 'extensible',
    r'\bExtensível\b': 'Extensible',
    r'\bclara\b': 'clear',
    r'\bClara\b': 'Clear',
    r'\bFácil\b': 'Easy',
    r'\bfácil\b': 'easy',
    
    # Verbos
    r'\bPrerequisitos\b': 'Prerequisites',
    r'\bprerequisitos\b': 'prerequisites',
    r'\bpré-requisitos\b': 'prerequisites',
    r'\bPré-requisitos\b': 'Prerequisites',
    r'\bInicialização\b': 'Initialization',
    r'\binicialização\b': 'initialization',
    r'\bVerification\b': 'Verification',
    r'\bverification\b': 'verification',
    r'\bVerificação\b': 'Verification',
    r'\bverificação\b': 'verification',
    r'\bComandos\b': 'Commands',
    r'\bcomandos\b': 'commands',
    r'\bTargets\b': 'Targets',
    r'\btargets\b': 'targets',
    r'\bMonitoramento\b': 'Monitoring',
    r'\bmonitoramento\b': 'monitoring',
    r'\bTests\b': 'Tests',
    r'\btests\b': 'tests',
    r'\bTestes\b': 'Tests',
    r'\btestes\b': 'tests',
    r'\bbenchmarks\b': 'benchmarks',
    r'\bBenchmarks\b': 'Benchmarks',
    r'\bbackup\b': 'backup',
    r'\bBackup\b': 'Backup',
    
    # Expressões portuguesas
    r'\bO that\b': 'What',
    r'\bo that\b': 'what',
    r'\bdar certo\b': 'go right',
    r'\bDar Certo\b': 'Go Right',
    r'\bdar errado\b': 'go wrong',
    r'\bDar Errado\b': 'Go Wrong',
    r'\bPlano of\b': 'Plan for',
    r'\bplano of\b': 'plan for',
    r'\bby that\b': 'why',
    r'\bBy that\b': 'Why',
    r'\bvai dar\b': 'will',
    r'\bVai dar\b': 'Will',
    r'\bfor melhor\b': 'for better',
    r'\bFor melhor\b': 'For better',
    
    # Adjetivos compostos
    r'\bwell estruturado\b': 'well structured',
    r'\bWell estruturado\b': 'Well structured',
    r'\bwell organizado\b': 'well organized',
    r'\bWell organizado\b': 'Well organized',
    r'\binline clara\b': 'inline clear',
    r'\bInline clara\b': 'Inline clear',
    
    # Mais substantivos
    r'\bLimites\b': 'Limits',
    r'\blimites\b': 'limits',
    r'\brecursos\b': 'resources',
    r'\bRecursos\b': 'Resources',
    r'\bVolumes\b': 'Volumes',
    r'\bvolumes\b': 'volumes',
    r'\bRede\b': 'Network',
    r'\brede\b': 'network',
    r'\bScript\b': 'Script',
    r'\bscript\b': 'script',
    r'\bservices\b': 'services',
    r'\bServices\b': 'Services',
    r'\bserviços\b': 'services',
    r'\bServiços\b': 'Services',
    r'\blogs\b': 'logs',
    r'\bLogs\b': 'Logs',
    r'\bInterface\b': 'Interface',
    r'\binterface\b': 'interface',
    r'\bdevelopment\b': 'development',
    r'\bDevelopment\b': 'Development',
    
    # Palavras com acentos
    r'\bMétricas\b': 'Metrics',
    r'\bmétricas\b': 'metrics',
    r'\bQualidade\b': 'Quality',
    r'\bqualidade\b': 'quality',
    r'\bExplicabilidade\b': 'Explainability',
    r'\bexplicabilidade\b': 'explainability',
    r'\bInhavepretabilidade\b': 'Interpretability',
    r'\binhavepretabilidade\b': 'interpretability',
    r'\bInterpretabilidade\b': 'Interpretability',
    r'\binterpretabilidade\b': 'interpretability',
    r'\bDepende\b': 'Depends',
    r'\bdepende\b': 'depends',
    r'\bLatência\b': 'Latency',
    r'\blatência\b': 'latency',
    r'\bSimular\b': 'Simulate',
    r'\bsimular\b': 'simulate',
    r'\bAcesso\b': 'Access',
    r'\bacesso\b': 'access',
    r'\bremoto\b': 'remote',
    r'\bRemoto\b': 'Remote',
    r'\bAumenta\b': 'Increases',
    r'\baumenta\b': 'increases',
    r'\bEsparsidade\b': 'Sparsity',
    r'\besparsidade\b': 'sparsity',
    r'\bRepreifntação\b': 'Representation',
    r'\brepreifntação\b': 'representation',
    r'\bRepresentação\b': 'Representation',
    r'\brepresentação\b': 'representation',
    r'\bAjustar\b': 'Adjust',
    r'\bajustar\b': 'adjust',
    r'\bIfveridade\b': 'Severity',
    r'\bifveridade\b': 'severity',
    r'\bSeveridade\b': 'Severity',
    r'\bseveridade\b': 'severity',
    r'\bDiagnósticos\b': 'Diagnostics',
    r'\bdiagnósticos\b': 'diagnostics',
    r'\bapropriado\b': 'appropriate',
    r'\bApropriado\b': 'Appropriate',
    r'\botimizada\b': 'optimized',
    r'\bOtimizada\b': 'Optimized',
    r'\bRecommendation\b': 'Recommendation',
    r'\brecommendation\b': 'recommendation',
    r'\bInvestigar\b': 'Investigate',
    r'\binvestigar\b': 'investigate',
    r'\bvalidar\b': 'validate',
    r'\bValidar\b': 'Validate',
    r'\bParâmetros\b': 'Parameters',
    r'\bparâmetros\b': 'parameters',
    r'\bPrebevam\b': 'Preserve',
    r'\bprebevam\b': 'preserve',
    r'\bFuncionalidade\b': 'Functionality',
    r'\bfuncionalidade\b': 'functionality',
    r'\bInhavena\b': 'Internal',
    r'\binhavena\b': 'internal',
    r'\bInterna\b': 'Internal',
    r'\binterna\b': 'internal',
    r'\bIfcrets\b': 'Secrets',
    r'\bifcrets\b': 'secrets',
    r'\bActions\b': 'Actions',
    r'\bactions\b': 'actions',
    r'\bAtualizar\b': 'Update',
    r'\batualizar\b': 'update',
    r'\bContribuição\b': 'Contribution',
    r'\bcontribuição\b': 'contribution',
    r'\bDiretrizes\b': 'Guidelines',
    r'\bdiretrizes\b': 'guidelines',
    r'\bEstilo\b': 'Style',
    r'\bestilo\b': 'style',
    r'\bInício\b': 'Start',
    r'\binício\b': 'start',
    r'\bRápido\b': 'Quick',
    r'\brápido\b': 'quick',
    r'\bTudo\b': 'Everything',
    r'\btudo\b': 'everything',
    r'\bDeployment\b': 'Deployment',
    r'\bdeployment\b': 'deployment',
    r'\bExecução\b': 'Execution',
    r'\bexecução\b': 'execution',
    r'\bLocal\b': 'Local',
    r'\blocal\b': 'local',
    
    # Palavras corruptas anteriores
    r'\bof the\b': 'of the',
    r'\bOf the\b': 'Of the',
    r'\bfrom the\b': 'from the',
    r'\bFrom the\b': 'From the',
    r'\bfor the\b': 'for the',
    r'\bFor the\b': 'For the',
    r'\bwith the\b': 'with the',
    r'\bWith the\b': 'With the',
    r'\bin the\b': 'in the',
    r'\bIn the\b': 'In the',
    r'\bto the\b': 'to the',
    r'\bTo the\b': 'To the',
}

def translate_file(file_path):
    """Traduz um arquivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Aplicar todas as traduções
        for pt_pattern, en_replacement in PORTUGUESE_ENGLISH.items():
            content = re.sub(pt_pattern, en_replacement, content, flags=re.MULTILINE)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"ERROR {file_path}: {e}")
        return False

def main():
    print("=" * 80)
    print(" TRADUÇÃO DEFINITIVA - PORTUGUÊS → INGLÊS")
    print("=" * 80)
    print()
    
    # Procurar todos os arquivos
    patterns = ['**/*.md', '**/*.py']
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    skip_files = {'translate_', 'cleanup', 'polish', 'fix_', 'ultra_', 'aggressive'}
    
    files_changed = 0
    base = Path('.')
    
    for pattern in patterns:
        for file_path in base.glob(pattern):
            # Skip diretórios
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Skip scripts de tradução
            if any(skip in file_path.name for skip in skip_files):
                continue
            
            if file_path.is_file():
                if translate_file(file_path):
                    print(f"✓ {file_path}")
                    files_changed += 1
    
    print()
    print("=" * 80)
    print(f" COMPLETO! Traduzidos {files_changed} arquivos.")
    print("=" * 80)

if __name__ == '__main__':
    main()
