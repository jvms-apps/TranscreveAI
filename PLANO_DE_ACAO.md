# Plano de Ação — Transcrição Inteligente de Áudios e Vídeos (TranscreveAI)

**Pasta do projeto:** `G:\Meu Drive\Antigravity Projects\TranscricaoAudios`  
**Idiomas prioritários:** Português (PT-BR/PT) e Inglês (EN), com suporte a detecção automática  
**Formatos de entrada:** Vídeos (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.wmv`, `.flv`, `.m4v`) e Áudios (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma`, `.opus`)  
**Formatos de saída:** `.txt`, `.docx` e `.pdf`

---

## 1. Escolha da Linguagem e Arquitetura

### Por que Python?
**Python 3.12** é a linguagem padrão-ouro para processamento de áudio, inteligência artificial de fala (ASR), diarização de locutores e geração de documentos estruturados:
- Acesso nativo ao motor de transcrição **Faster-Whisper (CTranslate2)** com aceleração por GPU **NVIDIA CUDA** (aproveitando a GPU **RTX 3050 6GB** disponível nesta máquina).
- Diarização de locutores de alto nível via **Sherpa-ONNX** (combinando segmentação baseada no **Pyannote 3.0** + extração de embeddings biométricos de voz **WeSpeaker / NeMo TitaNet**), rodando **100% localmente, gratuitamente e sem necessidade de conta ou token do HuggingFace**.
- Bibliotecas maduras e confiáveis para exportação de documentos: `python-docx` (Word `.docx`) e `reportlab` (PDF vetorial com fontes TrueType e suporte total a acentuação em português).
- Backend web leve e rápido com **FastAPI**, servindo uma interface web moderna (HTML5/CSS3/JS) direto no navegador local — muito mais rica e confortável que janelas desktop antigas, permitindo arrastar e soltar arquivos, ouvir o áudio/vídeo sincronizado com cada parágrafo, renomear locutores antes de exportar e acompanhar o progresso em tempo real.

### Decisão de Engenharia Importante: Google Drive vs. Ambiente Virtual
Como a pasta do projeto está no **Google Drive** (`G:\Meu Drive\...`), manter ambientes Python gigantes (`.venv` com ~3 GB de bibliotecas e modelos de IA) dentro do Drive causaria sincronização lenta e milhares de arquivos temporários na nuvem.

Portanto, adotamos a seguinte separação limpa:
- **No Google Drive (`G:\Meu Drive\Antigravity Projects\TranscricaoAudios`):**
  - Todo o código-fonte (`app/`, `transcrever.py`), documentação (`PLANO_DE_ACAO.md`, `README.md`), lista de dependências (`requirements.txt`), scripts de inicialização (`iniciar.bat`, `instalar.ps1`), histórico de transcrições (`dados/historico/`) e exportações salvas (`exportacoes/`).
- **Em disco local rápido (`%LOCALAPPDATA%\TranscricaoAudios`):**
  - Ambiente virtual Python (`venv/`), cache dos modelos Whisper/ONNX (`models/`) e arquivos temporários de conversão de áudio (`temp/`).

---

## 2. Arquitetura Técnica da Aplicação

```text
TranscricaoAudios/
├── PLANO_DE_ACAO.md          # Este plano de ação e arquitetura
├── README.md                 # Manual de uso rápido (Web + CLI)
├── requirements.txt          # Dependências Python
├── instalar.ps1              # Instalador automático (ambiente local + dependências)
├── iniciar.bat               # Atalho duplo-clique para abrir a aplicação Web
├── transcrever.py            # Interface de linha de comando (CLI)
├── app/
│   ├── __init__.py
│   ├── config.py             # Configurações, caminhos (Drive vs LocalAppData), GPU/CPU
│   ├── main.py               # Servidor FastAPI, rotas REST e fluxo de tarefas
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio.py          # Extração e conversão de vídeo/áudio para WAV 16kHz mono (FFmpeg)
│   │   ├── transcriber.py    # Transcrição com Faster-Whisper (CUDA/CPU, VAD, word timestamps)
│   │   ├── diarizer.py       # Diarização local sem token (Sherpa-ONNX: Pyannote 3.0 + WeSpeaker)
│   │   ├── aligner.py        # Alinhamento palavra-locutor e formação de parágrafos coerentes
│   │   └── pipeline.py       # Orquestrador completo com callbacks de progresso por etapa
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── txt_exporter.py   # Geração de arquivo .txt formatado (UTF-8)
│   │   ├── docx_exporter.py  # Geração de documento Word .docx estilizado por locutor
│   │   └── pdf_exporter.py   # Geração de documento PDF vetorial com fontes TTF (acentuação PT-BR)
│   ├── storage.py            # Persistência do histórico de transcrições (JSON)
│   └── static/
│       ├── index.html        # Interface web moderna (upload drag-and-drop, editor, player, export)
│       ├── styles.css        # Design limpo, responsivo e profissional
│       └── app.js            # Lógica do frontend, progresso em tempo real, edição e exportação
├── dados/
│   └── historico/            # Transcrições salvas para reabertura/edição/re-exportação
└── exportacoes/              # Cópia local dos arquivos exportados (TXT, DOCX, PDF)
```

---

## 3. Pipeline de Processamento (Passo a Passo)

1. **Ingestão e Extração de Áudio (`audio.py`)**
   - Recebe qualquer arquivo de vídeo ou áudio suportado pelo FFmpeg.
   - Converte a faixa de áudio para **WAV PCM 16 kHz Mono** (formato ótimo tanto para o Whisper quanto para o modelo de diarização) e mede a duração exata da mídia.
2. **Transcrição de Fala (`transcriber.py`)**
   - Carrega o modelo **Faster-Whisper** selecionado:
     - `large-v3-turbo` *(Padrão recomendado — excelente equilíbrio entre precisão altíssima em PT/EN e velocidade na RTX 3050 6GB usando `int8_float16`)*
     - `medium` *(Rápido e leve)*
     - `small` *(Ultrarrápido para testes ou áudios limpos)*
     - `large-v3` *(Precisão máxima para áudios difíceis/técnicos)*
   - Aplica **Silero VAD** (`vad_filter=True`) para ignorar silêncios/ruídos e evitar alucinações.
   - Extrai **timestamps no nível da palavra** (`word_timestamps=True`), essencial para saber exatamente onde termina um locutor e começa outro.
   - Permite escolher: **Detecção Automática**, **Português (`pt`)** ou **Inglês (`en`)**.
3. **Diarização de Locutores (`diarizer.py`)** *(opcional — ativada por padrão)*
   - Baixa automaticamente (apenas na primeira execução) os modelos ONNX abertos do projeto `k2-fsa/sherpa-onnx`:
     - **Segmentação de fala:** `sherpa-onnx-pyannote-segmentation-3-0`
     - **Embeddings de voz:** `wespeaker_en_voxceleb_resnet34_LM.onnx` (independente de idioma — identifica o timbre acústico de cada voz em PT e EN).
   - Agrupa segmentos por identidade vocal:
     - **Modo Automático:** estima automaticamente quantos locutores falaram usando limiar de similaridade (`threshold` ajustável).
     - **Número Definido:** o usuário pode informar exatamente quantas pessoas estão na conversa (ex: entrevista com 2 pessoas), aumentando ainda mais a precisão.
4. **Alinhamento e Divisão em Parágrafos (`aligner.py`)**
   - Cruza cada palavra transcrita com os intervalos de locutores retornados pela diarização (por maior sobreposição temporal ou proximidade mínima).
   - Aplica **suavização inteligente** para eliminar trocas espúrias de 1–2 palavras no meio de uma sentença e alinhar trocas de locutor às fronteiras naturais de pontuação.
   - Agrupa falas consecutivas do mesmo locutor em **parágrafos coesos** (com opção de quebrar parágrafos muito longos em pausas naturais > 2.5s), mantendo o horário inicial e final de cada bloco.
5. **Revisão Interativa e Exportação (`exporters/` + Interface Web)**
   - Na tela de resultados, o usuário pode:
     - Reproduzir o áudio/vídeo original clicando em qualquer parágrafo (pula exatamente para aquele segundo).
     - **Renomear os locutores** (ex: `Locutor 1` → `Entrevistador`, `Locutor 2` → `Dr. Carlos`) com atualização imediata em todos os parágrafos.
     - Editar o texto de qualquer parágrafo ou reatribuir o locutor de um parágrafo específico.
     - Escolher se deseja incluir timestamps e cabeçalho de metadados.
     - Exportar em **TXT**, **DOCX** ou **PDF** (download imediato no navegador + opção de salvar direto na pasta `exportacoes/`).

---

## 4. Cronograma de Execução

| Etapa | Descrição | Status |
| :--- | :--- | :--- |
| **1** | Planejamento, arquitetura e criação da estrutura na pasta do Drive | Concluída |
| **2** | Instalação do Python 3.12 oficial, `uv`, ambiente virtual local e bibliotecas (CUDA + Sherpa-ONNX + FFmpeg + FastAPI + Docx/PDF) | Concluída |
| **3** | Desenvolvimento do núcleo de extração de áudio, transcrição GPU, diarização local e alinhamento de parágrafos | Concluída |
| **4** | Implementação dos exportadores profissionais `.txt`, `.docx` e `.pdf` | Concluída |
| **5** | Criação da interface Web (FastAPI + UI moderna), CLI (`transcrever.py`) e atalho `iniciar.bat` | Concluída |
| **6** | Testes automatizados e teste real ponta a ponta com vídeo/áudio multi-locutor em Português e Inglês | Concluída |
