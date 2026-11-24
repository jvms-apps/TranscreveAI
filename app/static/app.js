(() => {
  const VIDEO_EXTS = new Set([".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg", ".3gp", ".ts"]);
  const LANG_NAMES = { pt: "Português", en: "Inglês", es: "Espanhol", auto: "Automático" };
  const STAGE_ORDER = ["extracting", "transcribing", "diarizing", "aligning"];

  const state = {
    selectedFile: null,
    history: [],
    currentDoc: null,
    dirty: false,
    activeJobId: null,
    pollTimer: null,
    elapsedTimer: null,
    jobStartTime: null,
  };

  // DOM Elements
  const $ = (id) => document.getElementById(id);

  const viewUpload = $("view-upload");
  const viewEditor = $("view-editor");
  const btnNew = $("btn-new-transcription");
  const hwBadge = $("hardware-badge");
  const hwText = $("hardware-text");
  const historyList = $("history-list");
  const historyCount = $("history-count");
  const historySearch = $("history-search");
  const btnOpenExports = $("btn-open-exports");

  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const dropzoneEmpty = $("dropzone-empty");
  const dropzoneSelected = $("dropzone-selected");
  const selectedFileType = $("selected-file-type");
  const selectedFileName = $("selected-file-name");
  const selectedFileSize = $("selected-file-size");
  const btnClearFile = $("btn-clear-file");
  const optLanguage = $("opt-language");
  const optModel = $("opt-model");
  const optQuality = $("opt-quality");
  const optEnhanceAudio = $("opt-enhance-audio");
  const optDiarization = $("opt-diarization");
  const diarizationOptions = $("diarization-options");
  const optNumSpeakers = $("opt-num-speakers");
  const optDiarizationQuality = $("opt-diarization-quality");
  const optThreshold = $("opt-threshold");
  const thresholdValue = $("threshold-value");
  const optPrompt = $("opt-prompt");
  const btnStart = $("btn-start-transcribe");

  const progressPanel = $("progress-panel");
  const progressTitle = $("progress-title");
  const progressMessage = $("progress-message");
  const progressPercent = $("progress-percent");
  const progressElapsed = $("progress-elapsed");
  const progressBarFill = $("progress-bar-fill");

  const docTitleInput = $("doc-title-input");
  const docBadges = $("doc-badges");
  const btnSaveEdits = $("btn-save-edits");
  const btnDeleteDoc = $("btn-delete-doc");
  const warningBanner = $("diarization-warning-banner");

  const expTimestamps = $("exp-timestamps");
  const expMetadata = $("exp-metadata");
  const expSpeakers = $("exp-speakers");
  const expSaveCopy = $("exp-save-copy");
  const btnCopyText = $("btn-copy-text");

  const mediaCard = $("media-card");
  const videoPlayer = $("video-player");
  const audioPlayer = $("audio-player");
  const speakersCard = $("speakers-card");
  const speakersList = $("speakers-list");
  const transcriptSearch = $("transcript-search");
  const speakerFilter = $("speaker-filter");
  const paragraphsList = $("paragraphs-list");
  const toastContainer = $("toast-container");

  // Helpers
  function showToast(msg, type = "info") {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    toastContainer.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity 0.25s";
      setTimeout(() => el.remove(), 260);
    }, 3400);
  }

  function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let val = bytes;
    while (val >= 1024 && i < units.length - 1) {
      val /= 1024;
      i++;
    }
    return `${val.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function formatTime(sec, forceHours = false) {
    const total = Math.max(0, Math.round(sec || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const pad = (n) => String(n).padStart(2, "0");
    if (forceHours || h > 0) return `${pad(h)}:${pad(m)}:${pad(s)}`;
    return `${pad(m)}:${pad(s)}`;
  }

  function formatDurationHuman(sec) {
    const total = Math.max(0, Math.round(sec || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const parts = [];
    if (h > 0) parts.push(`${h}h`);
    if (m > 0 || h > 0) parts.push(`${m}min`);
    parts.push(`${s}s`);
    return parts.join(" ");
  }

  function switchView(name) {
    viewUpload.classList.toggle("active", name === "upload");
    viewEditor.classList.toggle("active", name === "editor");
    renderHistory();
  }

  function setDirty(isDirty) {
    state.dirty = isDirty;
    btnSaveEdits.disabled = !isDirty;
    btnSaveEdits.textContent = isDirty ? "💾 Salvar Alterações *" : "💾 Salvo";
  }

  function getActivePlayer() {
    if (!state.currentDoc || !state.currentDoc.has_media) return null;
    return state.currentDoc.is_video ? videoPlayer : audioPlayer;
  }

  // Initialization
  async function loadSystemInfo() {
    try {
      const res = await fetch("/api/system-info");
      const data = await res.json();
      if (data.hardware && data.hardware.cuda_available) {
        hwBadge.classList.add("cuda");
        const vram = data.hardware.cuda_free_vram_mb
          ? ` · ${(data.hardware.cuda_free_vram_mb / 1024).toFixed(1)} GB livres`
          : "";
        hwText.textContent = `GPU NVIDIA CUDA ativa${vram}`;
      } else {
        hwBadge.classList.add("cpu");
        hwText.textContent = `Modo CPU (${data.hardware?.cpu_threads || 4} threads)`;
        // Em modo CPU (ex: executável Universal sem DLLs CUDA), recomenda 'small' por padrão para evitar esperas longas
        if (optModel && optModel.value === "large-v3-turbo") {
          optModel.value = "small";
        }
      }
    } catch {
      hwText.textContent = "Servidor local conectado";
    }
  }

  async function loadHistory() {
    try {
      const res = await fetch("/api/transcriptions");
      const data = await res.json();
      state.history = data.items || [];
      historyCount.textContent = String(state.history.length);
      renderHistory();
    } catch (err) {
      console.error(err);
    }
  }

  function renderHistory() {
    const q = (historySearch.value || "").trim().toLowerCase();
    const filtered = state.history.filter(
      (item) =>
        !q ||
        (item.title || "").toLowerCase().includes(q) ||
        (item.filename || "").toLowerCase().includes(q)
    );

    if (!filtered.length) {
      historyList.innerHTML = `<div class="empty-hint">${
        state.history.length ? "Nenhum item encontrado." : "Nenhuma transcrição salva ainda."
      }</div>`;
      return;
    }

    historyList.innerHTML = "";
    for (const item of filtered) {
      const div = document.createElement("div");
      const isActive = viewEditor.classList.contains("active") && state.currentDoc?.id === item.id;
      div.className = `history-item${isActive ? " active" : ""}`;
      div.innerHTML = `
        <div class="history-item-title" title="${escapeHtml(item.title)}">${escapeHtml(item.title)}</div>
        <div class="history-item-meta">
          <span>${formatDurationHuman(item.duration)}</span>
          <span>·</span>
          <span>${(item.language || "-").toUpperCase()}</span>
          ${item.diarization_enabled ? `<span>· ${item.speaker_count} loc.</span>` : ""}
        </div>
      `;
      div.addEventListener("click", () => openTranscription(item.id));
      historyList.appendChild(div);
    }
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // File Selection
  function selectFile(file) {
    if (!file) {
      state.selectedFile = null;
      dropzoneEmpty.classList.remove("hidden");
      dropzoneSelected.classList.add("hidden");
      btnStart.disabled = true;
      fileInput.value = "";
      return;
    }

    state.selectedFile = file;
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    const isVid = VIDEO_EXTS.has(ext) || (file.type || "").startsWith("video/");
    selectedFileType.textContent = isVid ? "VÍDEO" : "ÁUDIO";
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);
    dropzoneEmpty.classList.add("hidden");
    dropzoneSelected.classList.remove("hidden");
    btnStart.disabled = false;
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#btn-clear-file")) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) selectFile(fileInput.files[0]);
  });
  btnClearFile.addEventListener("click", (e) => {
    e.stopPropagation();
    selectFile(null);
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) selectFile(file);
  });

  optDiarization.addEventListener("change", () => {
    diarizationOptions.classList.toggle("hidden", !optDiarization.checked);
  });
  optThreshold.addEventListener("input", () => {
    thresholdValue.textContent = Number(optThreshold.value).toFixed(2);
  });

  // Start Transcription
  btnStart.addEventListener("click", async () => {
    if (!state.selectedFile) return;

    const formData = new FormData();
    formData.append("file", state.selectedFile);
    formData.append("model_name", optModel.value);
    formData.append("language", optLanguage.value);
    formData.append("quality_preset", optQuality.value);
    formData.append("enhance_audio", optEnhanceAudio.checked ? "true" : "false");
    formData.append("enable_diarization", optDiarization.checked ? "true" : "false");
    formData.append("num_speakers", optNumSpeakers.value);
    formData.append("diarization_quality", optDiarizationQuality.value);
    formData.append("diarization_threshold", optThreshold.value);
    formData.append("initial_prompt", optPrompt.value || "");

    btnStart.disabled = true;
    progressPanel.classList.remove("hidden");
    progressTitle.textContent = `Transcrevendo: ${state.selectedFile.name}`;
    progressMessage.textContent = "Enviando arquivo para o servidor local...";
    progressPercent.textContent = "0%";
    progressBarFill.style.width = "2%";
    updateStageSteps("queued");

    state.jobStartTime = Date.now();
    clearInterval(state.elapsedTimer);
    state.elapsedTimer = setInterval(() => {
      const sec = Math.floor((Date.now() - state.jobStartTime) / 1000);
      progressElapsed.textContent = formatTime(sec);
    }, 500);

    try {
      const res = await fetch("/api/transcribe", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Falha ao enviar arquivo." }));
        throw new Error(err.detail || "Falha ao iniciar transcrição.");
      }
      const data = await res.json();
      state.activeJobId = data.job_id;
      pollJobStatus(data.job_id);
    } catch (err) {
      stopTimers();
      btnStart.disabled = false;
      progressMessage.textContent = err.message;
      showToast(err.message, "error");
    }
  });

  function stopTimers() {
    clearInterval(state.pollTimer);
    clearInterval(state.elapsedTimer);
    state.pollTimer = null;
    state.elapsedTimer = null;
  }

  function updateStageSteps(currentStage) {
    const steps = document.querySelectorAll(".stage-step");
    const idx = STAGE_ORDER.indexOf(currentStage);
    steps.forEach((step) => {
      const s = step.dataset.stage;
      const sIdx = STAGE_ORDER.indexOf(s);
      step.classList.remove("active", "done");
      if (currentStage === "completed") {
        step.classList.add("done");
      } else if (idx >= 0) {
        if (sIdx < idx) step.classList.add("done");
        else if (sIdx === idx) step.classList.add("active");
      }
    });
  }

  function pollJobStatus(jobId) {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error("Tarefa não encontrada.");
        const job = await res.json();

        progressPercent.textContent = `${job.progress || 0}%`;
        progressBarFill.style.width = `${Math.max(2, job.progress || 0)}%`;
        progressMessage.textContent = job.message || "Processando...";
        updateStageSteps(job.stage);

        if (job.status === "completed") {
          stopTimers();
          showToast("Transcrição concluída com sucesso!", "success");
          await loadHistory();
          selectFile(null);
          progressPanel.classList.add("hidden");
          await openTranscription(job.transcription_id);
        } else if (job.status === "failed") {
          stopTimers();
          btnStart.disabled = false;
          showToast(job.error || "Erro durante o processamento.", "error");
        }
      } catch (err) {
        stopTimers();
        btnStart.disabled = false;
        showToast(err.message, "error");
      }
    }, 900);
  }

  // Open & Render Transcription Editor
  async function openTranscription(id) {
    try {
      const res = await fetch(`/api/transcriptions/${id}`);
      if (!res.ok) throw new Error("Não foi possível carregar a transcrição.");
      const doc = await res.json();
      state.currentDoc = doc;
      setDirty(false);
      renderEditor();
      switchView("editor");
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  function renderEditor() {
    const doc = state.currentDoc;
    if (!doc) return;

    docTitleInput.value = doc.title || doc.filename || "Transcrição";

    // Badges
    const langLabel = LANG_NAMES[(doc.language || "").toLowerCase()] || (doc.language || "-").toUpperCase();
    const spkCount = Object.keys(doc.speakers || {}).length;
    docBadges.innerHTML = `
      <span class="meta-pill">📄 ${escapeHtml(doc.filename || "")}</span>
      <span class="meta-pill">⏱️ ${formatDurationHuman(doc.duration)}</span>
      <span class="meta-pill">🌐 ${escapeHtml(langLabel)}</span>
      <span class="meta-pill">🧠 Whisper ${escapeHtml(doc.model_name || "")} (${(doc.device_used || "cpu").toUpperCase()})</span>
      <span class="meta-pill">📝 ${doc.total_words || 0} palavras · ${(doc.paragraphs || []).length} parágrafos</span>
      ${doc.diarization_enabled ? `<span class="meta-pill">👥 ${spkCount} locutor(es)</span>` : ""}
    `;

    // Warning banner
    if (doc.diarization_warning) {
      warningBanner.textContent = `⚠️ ${doc.diarization_warning}`;
      warningBanner.classList.remove("hidden");
    } else {
      warningBanner.classList.add("hidden");
    }

    // Media player
    videoPlayer.pause();
    audioPlayer.pause();
    if (doc.has_media) {
      const mediaUrl = `/api/media/${doc.id}`;
      mediaCard.classList.remove("hidden");
      if (doc.is_video) {
        videoPlayer.src = mediaUrl;
        videoPlayer.classList.remove("hidden");
        audioPlayer.classList.add("hidden");
        audioPlayer.removeAttribute("src");
      } else {
        audioPlayer.src = mediaUrl;
        audioPlayer.classList.remove("hidden");
        videoPlayer.classList.add("hidden");
        videoPlayer.removeAttribute("src");
      }
    } else {
      mediaCard.classList.add("hidden");
    }

    renderSpeakersPanel();
    renderSpeakerFilter();
    renderParagraphs();
  }

  function renderSpeakersPanel() {
    const doc = state.currentDoc;
    const speakers = doc.speakers || {};
    const ids = Object.keys(speakers);

    if (!ids.length) {
      speakersCard.classList.add("hidden");
      return;
    }
    speakersCard.classList.remove("hidden");
    speakersList.innerHTML = "";

    for (const spkId of ids) {
      const info = speakers[spkId];
      const row = document.createElement("div");
      row.className = "speaker-row";
      row.innerHTML = `
        <div class="speaker-row-top">
          <span class="speaker-swatch" style="background:${escapeHtml(info.color || "#1D4ED8")}"></span>
          <input class="speaker-name-input" type="text" value="${escapeHtml(info.name || spkId)}" data-speaker-id="${escapeHtml(spkId)}" />
        </div>
        <div class="speaker-row-stats">
          ${info.paragraph_count || 0} parágrafo(s) · ${info.word_count || 0} palavras · ${formatDurationHuman(info.duration || 0)}
        </div>
      `;
      const input = row.querySelector(".speaker-name-input");
      input.addEventListener("input", () => {
        doc.speakers[spkId].name = input.value;
        setDirty(true);
        syncSpeakerNamesInDOM();
      });
      speakersList.appendChild(row);
    }
  }

  function renderSpeakerFilter() {
    const doc = state.currentDoc;
    const currentVal = speakerFilter.value || "all";
    speakerFilter.innerHTML = `<option value="all">Todos os locutores</option>`;
    for (const [spkId, info] of Object.entries(doc.speakers || {})) {
      const opt = document.createElement("option");
      opt.value = spkId;
      opt.textContent = info.name || spkId;
      speakerFilter.appendChild(opt);
    }
    speakerFilter.value = currentVal;
  }

  function syncSpeakerNamesInDOM() {
    renderSpeakerFilter();
    const doc = state.currentDoc;
    paragraphsList.querySelectorAll(".para-speaker-select").forEach((sel) => {
      Array.from(sel.options).forEach((opt) => {
        const info = doc.speakers[opt.value];
        if (info) opt.textContent = info.name || opt.value;
      });
    });
  }

  function renderParagraphs() {
    const doc = state.currentDoc;
    const speakers = doc.speakers || {};
    const speakerIds = Object.keys(speakers);
    const forceHours = (doc.duration || 0) >= 3600;
    const q = (transcriptSearch.value || "").trim().toLowerCase();
    const filterSpk = speakerFilter.value || "all";

    paragraphsList.innerHTML = "";
    const paragraphs = doc.paragraphs || [];

    const visible = paragraphs.filter((p) => {
      if (filterSpk !== "all" && p.speaker_id !== filterSpk) return false;
      if (q && !(p.text || "").toLowerCase().includes(q)) return false;
      return true;
    });

    if (!visible.length) {
      paragraphsList.innerHTML = `<div class="empty-hint">Nenhum parágrafo corresponde ao filtro atual.</div>`;
      return;
    }

    for (const p of visible) {
      const spk = speakers[p.speaker_id] || { name: p.speaker_id, color: "#1D4ED8" };
      const card = document.createElement("article");
      card.className = "para-card";
      card.dataset.start = String(p.start);
      card.dataset.end = String(p.end);
      card.style.borderLeftColor = spk.color || "#1D4ED8";

      const selectOptions = speakerIds
        .map(
          (sid) =>
            `<option value="${escapeHtml(sid)}" ${sid === p.speaker_id ? "selected" : ""}>${escapeHtml(
              speakers[sid]?.name || sid
            )}</option>`
        )
        .join("");

      card.innerHTML = `
        <div class="para-header">
          <div class="para-speaker-group">
            ${
              doc.diarization_enabled && speakerIds.length > 0
                ? `<select class="para-speaker-select" style="color:${escapeHtml(spk.color || "#1D4ED8")}">${selectOptions}</select>`
                : `<strong style="color:${escapeHtml(spk.color || "#1D4ED8")}; font-size: 0.9rem;">${escapeHtml(spk.name || "Transcrição")}</strong>`
            }
          </div>
          <button type="button" class="para-time-btn" title="Ouvir a partir deste trecho">
            ▶ ${formatTime(p.start, forceHours)} – ${formatTime(p.end, forceHours)}
          </button>
        </div>
        <div class="para-text" contenteditable="true" spellcheck="true">${escapeHtml(p.text)}</div>
      `;

      const spkSelect = card.querySelector(".para-speaker-select");
      if (spkSelect) {
        spkSelect.addEventListener("change", () => {
          p.speaker_id = spkSelect.value;
          const newColor = speakers[p.speaker_id]?.color || "#1D4ED8";
          card.style.borderLeftColor = newColor;
          spkSelect.style.color = newColor;
          setDirty(true);
        });
      }

      const timeBtn = card.querySelector(".para-time-btn");
      timeBtn.addEventListener("click", () => {
        const player = getActivePlayer();
        if (!player) {
          showToast("Mídia original não está disponível para reprodução.", "info");
          return;
        }
        player.currentTime = Math.max(0, Number(p.start) || 0);
        player.play().catch(() => {});
      });

      const textEl = card.querySelector(".para-text");
      textEl.addEventListener("input", () => {
        p.text = textEl.innerText.trim();
        setDirty(true);
      });

      paragraphsList.appendChild(card);
    }
  }

  // Highlight active paragraph during playback
  function bindPlayerHighlight(player) {
    player.addEventListener("timeupdate", () => {
      const t = player.currentTime;
      paragraphsList.querySelectorAll(".para-card").forEach((card) => {
        const start = Number(card.dataset.start);
        const end = Number(card.dataset.end);
        card.classList.toggle("playing", t >= start && t <= end + 0.25);
      });
    });
  }
  bindPlayerHighlight(videoPlayer);
  bindPlayerHighlight(audioPlayer);

  // Save Edits
  async function saveCurrentEdits(silent = false) {
    if (!state.currentDoc) return true;
    const doc = state.currentDoc;
    const speakersPayload = {};
    for (const [sid, info] of Object.entries(doc.speakers || {})) {
      speakersPayload[sid] = { name: info.name || sid, color: info.color || "#1D4ED8" };
    }
    const payload = {
      title: docTitleInput.value.trim() || doc.title,
      speakers: speakersPayload,
      paragraphs: (doc.paragraphs || []).map((p) => ({
        id: p.id,
        speaker_id: p.speaker_id,
        start: p.start,
        end: p.end,
        text: p.text,
      })),
    };

    try {
      const res = await fetch(`/api/transcriptions/${doc.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("Falha ao salvar alterações.");
      state.currentDoc = await res.json();
      setDirty(false);
      renderSpeakersPanel();
      await loadHistory();
      if (!silent) showToast("Alterações salvas com sucesso!", "success");
      return true;
    } catch (err) {
      showToast(err.message, "error");
      return false;
    }
  }

  docTitleInput.addEventListener("input", () => {
    if (state.currentDoc) {
      state.currentDoc.title = docTitleInput.value;
      setDirty(true);
    }
  });
  btnSaveEdits.addEventListener("click", () => saveCurrentEdits(false));

  // Export handlers (TXT / DOCX / PDF)
  document.querySelectorAll("[data-export]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!state.currentDoc) return;
      if (state.dirty) {
        const ok = await saveCurrentEdits(true);
        if (!ok) return;
      }
      const fmt = btn.dataset.export;
      const params = new URLSearchParams({
        include_timestamps: expTimestamps.checked ? "true" : "false",
        include_metadata: expMetadata.checked ? "true" : "false",
        include_speakers: expSpeakers.checked ? "true" : "false",
        save_copy: expSaveCopy.checked ? "true" : "false",
      });
      const url = `/api/transcriptions/${state.currentDoc.id}/export/${fmt}?${params.toString()}`;
      const a = document.createElement("a");
      a.href = url;
      a.download = "";
      document.body.appendChild(a);
      a.click();
      a.remove();
      showToast(
        expSaveCopy.checked
          ? `Arquivo .${fmt.toUpperCase()} gerado e salvo também em exportacoes/!`
          : `Download do arquivo .${fmt.toUpperCase()} iniciado!`,
        "success"
      );
    });
  });

  // Copy plain text
  btnCopyText.addEventListener("click", async () => {
    const doc = state.currentDoc;
    if (!doc) return;
    const speakers = doc.speakers || {};
    const forceHours = (doc.duration || 0) >= 3600;
    const lines = (doc.paragraphs || []).map((p) => {
      const spkName = speakers[p.speaker_id]?.name || p.speaker_id;
      const header = [];
      if (expTimestamps.checked) header.push(`[${formatTime(p.start, forceHours)} - ${formatTime(p.end, forceHours)}]`);
      if (expSpeakers.checked && doc.diarization_enabled) header.push(`${spkName}:`);
      return (header.length ? header.join(" ") + "\n" : "") + p.text;
    });
    try {
      await navigator.clipboard.writeText(lines.join("\n\n"));
      showToast("Transcrição copiada para a área de transferência!", "success");
    } catch {
      showToast("Não foi possível copiar automaticamente.", "error");
    }
  });

  // Delete transcription
  btnDeleteDoc.addEventListener("click", async () => {
    if (!state.currentDoc) return;
    if (!confirm(`Excluir a transcrição "${state.currentDoc.title}" do histórico?`)) return;
    try {
      const res = await fetch(`/api/transcriptions/${state.currentDoc.id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Erro ao excluir transcrição.");
      state.currentDoc = null;
      await loadHistory();
      switchView("upload");
      showToast("Transcrição removida.", "info");
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Navigation & filters
  btnNew.addEventListener("click", () => switchView("upload"));
  historySearch.addEventListener("input", renderHistory);
  transcriptSearch.addEventListener("input", renderParagraphs);
  speakerFilter.addEventListener("change", renderParagraphs);

  btnOpenExports.addEventListener("click", async () => {
    await fetch("/api/open-exports-folder", { method: "POST" });
  });

  // Boot
  loadSystemInfo();
  loadHistory();
})();
