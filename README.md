# TranscreveAI — Transcrição Inteligente de Áudios e Vídeos

Aplicação local em **Python** para transcrever arquivos de **vídeo e áudio** (com prioridade para **Português** e **Inglês**), identificar automaticamente os **locutores** dividindo a fala em parágrafos limpos, revisar com player sincronizado e exportar para **TXT**, **DOCX (Word)** e **PDF**.

100% gratuito, offline (após download automático dos modelos na 1ª vez), **sem API paga e sem necessidade de token do HuggingFace**.

---

## Principais Recursos

- **Suporte Amplo a Mídias:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.wmv`, `.flv`, `.m4v`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.wma`, `.opus` (via FFmpeg embutido).
- **Motor OpenAI Whisper em Máxima Qualidade (`faster-whisper` / CTranslate2):**
  - Modelos disponíveis: **`large-v3-turbo`** (padrão recomendado), **`large-v3`** (precisão máxima), **`large-v2`** (alta estabilidade em PT-BR), `medium`, `small` e `base`.
  - **Modo Qualidade Máxima:** executa em **FP16 nativo na GPU NVIDIA** (com fallback automático para `int8_float16` / CPU), `beam_size=8`, fallback de temperatura e proteção anti-alucinação em silêncios (VAD).
  - **Aprimoramento de Áudio Opcional (FFmpeg):** filtro de voz (70 Hz–7,8 kHz) + normalização de volume EBU R128 para gravações baixas ou ruidosas.
- **Identificação de Locutores (Diarização) Local e Sem Token:**
  - Combina segmentação **Pyannote 3.0 ONNX** + embeddings de voz **WeSpeaker** (`ResNet34-LM` padrão ou `ResNet293-LM` alta precisão) via `sherpa-onnx`.
  - Alinhamento palavra-a-palavra com suavização de fronteiras de frase e divisão automática em parágrafos por locutor.
- **Interface Web Moderna:**
  - Arraste e solte arquivos com barra de progresso por etapa em tempo real.
  - Player de áudio/vídeo sincronizado (clique em qualquer parágrafo para ouvir o trecho).
  - Renomeação instantânea de locutores (`Locutor 1` → nome real) e edição direta de texto antes de exportar.
- **Exportação Profissional:** `.txt` (UTF-8), `.docx` (Word estilizado com cores por locutor) e `.pdf` (A4 vetorial com fontes TrueType e acentuação perfeita).

---

## Dicas para Máxima Qualidade em Português e Inglês

1. **Escolha o modelo certo:**
   - **`large-v3-turbo`**: melhor equilíbrio geral (95–98% da fidelidade do `large-v3` sendo muito mais rápido).
   - **`large-v3`**: precisão máxima absoluta para reuniões técnicas, sotaques difíceis ou múltiplos falantes.
   - **`large-v2`**: ótima alternativa muito estável para PT-BR caso encontre áudios com repetições.
2. **Defina o idioma explicitamente** (`Português` ou `Inglês`) quando o áudio tiver um idioma predominante — isso reduz ambiguidades.
3. **Mantenha o Modo `Qualidade Máxima (FP16 + Beam 8)` ativado.**
4. **Ative `Aprimorar áudio`** se a gravação tiver volume baixo, eco ou ruído de ambiente.
5. **Use o campo `Vocabulário de apoio`** para listar nomes próprios, marcas ou siglas técnicas mencionadas no áudio.

---

## Download Rápido para Windows (Sem Instalar Nada)

Para amigos ou usuários que só querem usar o aplicativo sem instalar Python nem FFmpeg:

1. Acesse a aba **[Releases](https://github.com/jvms-apps/TranscreveAI/releases)** e baixe **`TranscreveAI-Windows-Universal.zip`**.
2. Extraia toda a pasta do `.zip`.
3. Dê duplo clique em **`TranscreveAI.exe`** — o navegador abrirá automaticamente.
   *(Caso o Windows SmartScreen exiba um aviso por ser software open-source sem certificado pago, clique em **"Mais informações" → "Executar assim mesmo"**.)*

---

## Como Executar (Modo Desenvolvedor / Local)

### Windows (Duplo Clique)
1. Clone o repositório:
   ```powershell
   git clone https://github.com/jvms-apps/TranscreveAI.git
   ```
2. Dê duplo clique em **`iniciar.bat`** (na primeira execução, `instalar.ps1` cria o ambiente Python 3.12 local em `%LOCALAPPDATA%\TranscricaoAudios\venv` e instala tudo automaticamente).
3. O navegador abrirá em `http://127.0.0.1:8765`.

### Linha de Comando (CLI)
```powershell
& "$env:LOCALAPPDATA\TranscricaoAudios\venv\Scripts\python.exe" transcrever.py "reuniao.mp4" --idioma pt --modelo large-v3 --qualidade high --melhorar-audio --formato all
```

---

## Gerar Executável (`.exe`) para Compartilhar com Amigos

Você pode empacotar o TranscreveAI em um `.zip` standalone contendo `TranscreveAI.exe` — seus amigos **não precisam instalar Python nem FFmpeg**:

```powershell
# Versão Universal (recomendada para compartilhar — ~200 MB, roda em qualquer Windows)
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -Mode cpu

# Versão com DLLs NVIDIA CUDA 12 embutidas (~1.3 GB, para amigos com placa NVIDIA)
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -Mode gpu
```

O pacote `.zip` pronto será salvo na pasta `releases/`.

### Publicação Automática no GitHub Releases
Este projeto inclui um workflow em [`.github/workflows/release-windows.yml`](.github/workflows/release-windows.yml). Ao publicar uma tag (por exemplo `v1.0.0`) no GitHub, o GitHub Actions compila automaticamente o executável Windows e disponibiliza `TranscreveAI-Windows-Universal.zip` na aba **Releases** para download direto.

---

## Licença
Distribuído sob a licença [MIT](LICENSE).
