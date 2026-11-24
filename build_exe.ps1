# ==============================================================================
# Gerador de Executável Standalone Windows (.exe + .zip) — TranscreveAI
#
# Modos disponíveis:
#   .\build_exe.ps1 -Mode cpu  (Padrão - ~200 MB zipado, roda em QUALQUER PC Windows)
#   .\build_exe.ps1 -Mode gpu  (~1.3 GB zipado, inclui DLLs NVIDIA CUDA 12)
#
# O build intermediário ocorre fora do Google Drive (%LOCALAPPDATA%) e apenas
# o pacote .zip final pronto para compartilhar é gravado em .\releases\
# ==============================================================================

param(
    [ValidateSet("cpu", "gpu")]
    [string]$Mode = "cpu"
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalDataDir = Join-Path $env:LOCALAPPDATA "TranscricaoAudios"
$VenvPython = Join-Path $LocalDataDir "venv\Scripts\python.exe"
$BuildRoot = Join-Path $LocalDataDir "build_exe_$Mode"
$WorkDir = Join-Path $BuildRoot "work"
$DistDir = Join-Path $BuildRoot "dist"
$SpecDir = Join-Path $BuildRoot "spec"
$ReleasesDir = Join-Path $ProjectDir "releases"

if (-not (Test-Path $VenvPython)) {
    throw "Ambiente virtual não encontrado em $VenvPython. Execute 'instalar.ps1' primeiro."
}

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  TranscreveAI - Build de Executável Windows ($Mode)" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 1. Garantir PyInstaller no ambiente virtual (sem escrever erro em stderr)
$pyiCheck = & $VenvPython -c "import importlib.util; print('OK' if importlib.util.find_spec('PyInstaller') else 'MISSING')"
if ($pyiCheck -ne "OK") {
    Write-Host "[1/4] Instalando PyInstaller..." -ForegroundColor Yellow
    $uvCmd = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv*" -Recurse -Filter "uv.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if ($uvCmd) {
        & $uvCmd.FullName pip install --python $VenvPython pyinstaller
    } else {
        & $VenvPython -m pip install pyinstaller
    }
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar PyInstaller." }
} else {
    Write-Host "[1/4] PyInstaller já instalado." -ForegroundColor Green
}

# 2. Limpar diretório de build anterior
Write-Host "[2/4] Preparando diretório de build em $BuildRoot..." -ForegroundColor Yellow
if (Test-Path $BuildRoot) {
    Remove-Item -Recurse -Force $BuildRoot
}
New-Item -ItemType Directory -Force -Path $WorkDir, $DistDir, $SpecDir, $ReleasesDir | Out-Null

# 3. Montar argumentos PyInstaller (--onedir: inicialização rápida e menor taxa de falso-positivo)
$StaticSrc = Join-Path $ProjectDir "app\static"
$LauncherPy = Join-Path $ProjectDir "launcher.py"

$PyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--console",
    "--name", "TranscreveAI",
    "--workpath", $WorkDir,
    "--distpath", $DistDir,
    "--specpath", $SpecDir,
    "--add-data", "$StaticSrc;app/static",
    "--collect-all", "faster_whisper",
    "--collect-all", "ctranslate2",
    "--collect-all", "sherpa_onnx",
    "--collect-all", "sherpa_onnx_core",
    "--collect-all", "imageio_ffmpeg",
    "--collect-all", "onnxruntime",
    "--collect-all", "soundfile",
    "--collect-all", "docx",
    "--collect-all", "reportlab",
    "--hidden-import", "uvicorn.logging",
    "--hidden-import", "uvicorn.loops.auto",
    "--hidden-import", "uvicorn.protocols.http.auto",
    "--hidden-import", "uvicorn.protocols.websockets.auto",
    "--hidden-import", "uvicorn.lifespan.on",
    "--hidden-import", "multipart"
)

if ($Mode -eq "gpu") {
    Write-Host "      Modo GPU: incluindo bibliotecas NVIDIA CUDA 12 (cuBLAS + cuDNN)..." -ForegroundColor Yellow
    $PyInstallerArgs += @(
        "--collect-all", "nvidia.cublas",
        "--collect-all", "nvidia.cudnn"
    )
} else {
    Write-Host "      Modo CPU Universal: excluindo DLLs NVIDIA gigantes (~1.2 GB) para pacote compacto..." -ForegroundColor Yellow
    $PyInstallerArgs += @(
        "--exclude-module", "nvidia",
        "--exclude-module", "nvidia.cublas",
        "--exclude-module", "nvidia.cudnn",
        "--exclude-module", "nvidia.cuda_nvrtc"
    )
}

$PyInstallerArgs += $LauncherPy

Write-Host "[3/4] Compilando executável com PyInstaller (aguarde ~1-2 min)..." -ForegroundColor Yellow
Push-Location $ProjectDir
try {
    # PyInstaller grava logs informativos (INFO:) em stderr, então mantemos ErrorActionPreference=Continue na chamada
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $VenvPython @PyInstallerArgs
    $pyiExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($pyiExit -ne 0) {
        throw "PyInstaller falhou com código $pyiExit"
    }
} finally {
    Pop-Location
}

# Copia um guia rápido LEIA-ME.txt para dentro da pasta distribuível
$BundleDir = Join-Path $DistDir "TranscreveAI"
$ReadmeDist = @"
============================================================
  TranscreveAI — Transcrição de Áudios e Vídeos (Windows)
============================================================

COMO USAR:
1. Extraia TODA esta pasta do arquivo .zip antes de executar.
2. Dê duplo clique em "TranscreveAI.exe".
3. O navegador abrirá automaticamente na interface da aplicação.
4. Na primeira transcrição, os modelos de IA serão baixados
   automaticamente para sua pasta local (precisa de internet só na 1ª vez).

OBSERVAÇÃO DO WINDOWS:
Por ser um projeto open-source sem certificado pago de assinatura digital,
o Windows Defender SmartScreen pode exibir um alerta na 1ª abertura.
Clique em "Mais informações" -> "Executar assim mesmo".

Exportações (TXT, DOCX e PDF) ficam na pasta "exportacoes" ao lado do .exe.
"@
Set-Content -Path (Join-Path $BundleDir "LEIA-ME.txt") -Value $ReadmeDist -Encoding UTF8

# 4. Compactar em .zip pronto para compartilhar com amigos / GitHub Releases
$Suffix = if ($Mode -eq "gpu") { "Windows-NVIDIA-GPU" } else { "Windows-Universal" }
$ZipName = "TranscreveAI-$Suffix.zip"
$ZipPath = Join-Path $ReleasesDir $ZipName
if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}

Write-Host "[4/4] Gerando arquivo ZIP distribuível em: $ZipPath..." -ForegroundColor Yellow
Compress-Archive -Path $BundleDir -DestinationPath $ZipPath -CompressionLevel Optimal

$ZipSizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "  BUILD CONCLUÍDO COM SUCESSO!" -ForegroundColor Green
Write-Host "  Executável : $BundleDir\TranscreveAI.exe" -ForegroundColor Green
Write-Host "  Pacote ZIP : $ZipPath ($ZipSizeMB MB)" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
