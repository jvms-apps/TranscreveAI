# ==============================================================================
# Instalador Automático - TranscreveAI
# Usa Python 3.12 oficial (assinado pela Python Software Foundation, compatível
# com o Smart App Control do Windows 11) e cria o venv fora do Google Drive.
# ==============================================================================

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalDataDir = Join-Path $env:LOCALAPPDATA "TranscricaoAudios"
$VenvDir = Join-Path $LocalDataDir "venv"
$ReqFile = Join-Path $ProjectDir "requirements.txt"

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  TranscreveAI - Instalacao do Ambiente Local" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "Projeto : $ProjectDir"
Write-Host "Ambiente: $VenvDir"
Write-Host ""

New-Item -ItemType Directory -Force -Path $LocalDataDir | Out-Null

function Find-OfficialPython312 {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Program Files\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Program Files\Python311\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $wingetPkg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv*" -Recurse -Filter "uv.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($wingetPkg) { return $wingetPkg.FullName }
    return $null
}

# 1. Garantir Python 3.12 oficial
$BasePython = Find-OfficialPython312
if (-not $BasePython) {
    Write-Host "[1/4] Instalando Python 3.12 oficial via winget..." -ForegroundColor Yellow
    winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
    $BasePython = Find-OfficialPython312
}
if (-not $BasePython) {
    throw "Nao foi possivel localizar o Python 3.12 oficial apos a instalacao."
}
Write-Host "[1/4] Python oficial encontrado em: $BasePython" -ForegroundColor Green

# 2. Recriar venv oficial se necessario
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$needRecreate = $true
if (Test-Path $VenvPython) {
    try {
        $verOut = & $VenvPython --version 2>&1
        if ($LASTEXITCODE -eq 0 -and ($verOut -match "Python 3\.1[12]")) {
            $cfgFile = Join-Path $VenvDir "pyvenv.cfg"
            if ((Test-Path $cfgFile) -and ((Get-Content $cfgFile -Raw) -match "Programs\\Python\\Python31[12]|Program Files\\Python31[12]")) {
                $needRecreate = $false
            }
        }
    } catch {
        $needRecreate = $true
    }
}

if ($needRecreate) {
    Write-Host "[2/4] Criando ambiente virtual com Python oficial..." -ForegroundColor Yellow
    if (Test-Path $VenvDir) {
        Remove-Item -Recurse -Force $VenvDir
    }
    & $BasePython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao criar ambiente virtual com $BasePython"
    }
}
Write-Host "[2/4] Ambiente virtual pronto em: $VenvDir" -ForegroundColor Green

# 3. Instalar dependencias (tenta uv para velocidade; fallback automatico para pip)
Write-Host "[3/4] Instalando dependencias (faster-whisper, CUDA runtime, sherpa-onnx, FastAPI, docx, pdf)..." -ForegroundColor Yellow
$UvExe = Find-Uv
$installed = $false
if ($UvExe) {
    & $UvExe pip install --python $VenvPython -r $ReqFile
    if ($LASTEXITCODE -eq 0) {
        $installed = $true
    } else {
        Write-Host "Aviso: uv pip falhou, usando pip padrao..." -ForegroundColor DarkYellow
    }
}
if (-not $installed) {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar dependencias Python."
    }
}
Write-Host "[3/4] Dependencias instaladas com sucesso!" -ForegroundColor Green

# 4. Validacao completa
Write-Host "[4/4] Validando bibliotecas e GPU CUDA..." -ForegroundColor Yellow
& $VenvPython -c "
import os, sys
for pkg in ('nvidia.cublas', 'nvidia.cudnn'):
    try:
        mod = __import__(pkg, fromlist=['__file__'])
        pkg_root = os.path.dirname(mod.__file__) if getattr(mod, '__file__', None) else list(mod.__path__)[0]
        bin_dir = os.path.join(pkg_root, 'bin')
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass
import ctranslate2, faster_whisper, sherpa_onnx, imageio_ffmpeg, fastapi, docx, reportlab
print('Python         :', sys.version.split()[0])
print('CUDA devices   :', ctranslate2.get_cuda_device_count())
print('faster-whisper :', faster_whisper.__version__)
print('sherpa-onnx    :', sherpa_onnx.__version__)
print('FFmpeg         :', imageio_ffmpeg.get_ffmpeg_exe())
"
if ($LASTEXITCODE -ne 0) {
    throw "Falha na validacao das bibliotecas instaladas."
}

Write-Host ""
Write-Host "Instalacao concluida com sucesso! Execute 'iniciar.bat' para abrir o TranscreveAI." -ForegroundColor Green
