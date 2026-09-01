"use strict";

const elements = {
  video: document.querySelector("#camera-video"),
  canvas: document.querySelector("#monitor-canvas"),
  cameraSelect: document.querySelector("#camera-select"),
  interval: document.querySelector("#inference-interval"),
  intervalValue: document.querySelector("#interval-value"),
  threshold: document.querySelector("#display-threshold"),
  thresholdValue: document.querySelector("#threshold-value"),
  start: document.querySelector("#start-monitor"),
  stop: document.querySelector("#stop-monitor"),
  clearEvents: document.querySelector("#clear-events"),
  eventList: document.querySelector("#event-list"),
  serviceDot: document.querySelector("#service-dot"),
  serviceStatus: document.querySelector("#service-status"),
  modelId: document.querySelector("#model-id"),
  cameraState: document.querySelector("#camera-state"),
  monitorState: document.querySelector("#monitor-state"),
  resolution: document.querySelector("#video-resolution"),
  notice: document.querySelector("#notice"),
  noticeText: document.querySelector("#notice-text"),
  statFps: document.querySelector("#stat-fps"),
  statLatency: document.querySelector("#stat-latency"),
  statHelmet: document.querySelector("#stat-helmet"),
  statNoHelmet: document.querySelector("#stat-no-helmet"),
  requestCount: document.querySelector("#request-count"),
  errorCount: document.querySelector("#error-count"),
};

const context = elements.canvas.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d", { alpha: false });

const state = {
  stream: null,
  running: false,
  serviceReady: false,
  inferenceBusy: false,
  inferenceTimer: null,
  animationFrame: null,
  detections: [],
  recentCompletions: [],
  lastEventAt: 0,
};

function setNotice(message, tone = "info") {
  elements.noticeText.textContent = message;
  elements.notice.classList.remove("notice--success", "notice--error");
  if (tone === "success") elements.notice.classList.add("notice--success");
  if (tone === "error") elements.notice.classList.add("notice--error");
}

function setServiceReady(ready, message, modelId = null) {
  state.serviceReady = ready;
  elements.serviceStatus.textContent = message;
  elements.serviceDot.className = `status-dot ${ready ? "status-dot--ready" : "status-dot--error"}`;
  elements.modelId.textContent = `模型：${modelId || "不可用"}`;
}

function drawIdle(message = "等待摄像头授权") {
  const { width, height } = elements.canvas;
  context.fillStyle = "#101412";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#303a35";
  context.lineWidth = 2;
  context.beginPath();
  context.rect(width / 2 - 42, height / 2 - 34, 84, 60);
  context.moveTo(width / 2 - 18, height / 2 - 34);
  context.lineTo(width / 2 - 7, height / 2 - 49);
  context.lineTo(width / 2 + 7, height / 2 - 49);
  context.lineTo(width / 2 + 18, height / 2 - 34);
  context.stroke();
  context.fillStyle = "#87918c";
  context.font = '500 22px "Segoe UI", "Microsoft YaHei", sans-serif';
  context.textAlign = "center";
  context.fillText(message, width / 2, height / 2 + 72);
}

function filteredDetections() {
  const threshold = Number(elements.threshold.value);
  return state.detections.filter((item) => item.confidence >= threshold);
}

function drawDetection(detection) {
  const [x1, y1, x2, y2] = detection.xyxy;
  const color = detection.class_name === "helmet" ? "#20c45a" : "#ff3e3e";
  const label = `${detection.class_name} ${detection.confidence.toFixed(2)}`;
  const scale = Math.max(1, elements.canvas.width / 960);
  context.lineWidth = Math.max(2, 3 * scale);
  context.strokeStyle = color;
  context.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
  context.font = `700 ${Math.round(15 * scale)}px "Segoe UI", sans-serif`;
  const labelWidth = context.measureText(label).width + 12 * scale;
  const labelHeight = 25 * scale;
  const labelY = Math.max(0, y1 - labelHeight);
  context.fillStyle = color;
  context.fillRect(x1, labelY, labelWidth, labelHeight);
  context.fillStyle = "#ffffff";
  context.textAlign = "left";
  context.textBaseline = "middle";
  context.fillText(label, x1 + 6 * scale, labelY + labelHeight / 2);
}

function renderFrame() {
  if (!state.running) return;
  if (elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    context.drawImage(elements.video, 0, 0, elements.canvas.width, elements.canvas.height);
    filteredDetections().forEach(drawDetection);
  }
  state.animationFrame = requestAnimationFrame(renderFrame);
}

function updateCurrentStats() {
  const detections = filteredDetections();
  elements.statHelmet.textContent = String(
    detections.filter((item) => item.class_name === "helmet").length,
  );
  elements.statNoHelmet.textContent = String(
    detections.filter((item) => item.class_name === "no_helmet").length,
  );
}

function appendEvents(detections) {
  const now = Date.now();
  if (now - state.lastEventAt < 1500 || detections.length === 0) return;
  state.lastEventAt = now;
  if (elements.eventList.querySelector(".event-empty")) elements.eventList.innerHTML = "";
  const timestamp = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date());
  detections.slice(0, 4).forEach((detection) => {
    const item = document.createElement("li");
    const classModifier = detection.class_name === "helmet" ? "helmet" : "no-helmet";
    item.innerHTML = `<time>${timestamp}</time><span class="event-class event-class--${classModifier}">${detection.class_name}</span><span>${detection.confidence.toFixed(2)}</span>`;
    elements.eventList.prepend(item);
  });
  while (elements.eventList.children.length > 12) elements.eventList.lastElementChild.remove();
}

function updateFps(completedAt) {
  state.recentCompletions.push(completedAt);
  state.recentCompletions = state.recentCompletions.filter((item) => completedAt - item <= 5000);
  const points = state.recentCompletions;
  const fps = points.length > 1 ? ((points.length - 1) * 1000) / (points.at(-1) - points[0]) : 0;
  elements.statFps.textContent = fps.toFixed(1);
}

function canvasBlob() {
  captureCanvas.width = elements.video.videoWidth;
  captureCanvas.height = elements.video.videoHeight;
  captureContext.drawImage(elements.video, 0, 0, captureCanvas.width, captureCanvas.height);
  return new Promise((resolve, reject) => {
    captureCanvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("无法读取摄像头画面"))),
      "image/jpeg",
      0.82,
    );
  });
}

function scheduleInference(delay = Number(elements.interval.value)) {
  window.clearTimeout(state.inferenceTimer);
  if (!state.running) return;
  state.inferenceTimer = window.setTimeout(runInference, delay);
}

async function runInference() {
  if (!state.running || state.inferenceBusy || elements.video.readyState < 2) {
    scheduleInference(100);
    return;
  }
  state.inferenceBusy = true;
  const startedAt = performance.now();
  try {
    const frame = await canvasBlob();
    const body = new FormData();
    body.append("image", frame, "camera-frame.jpg");
    const response = await fetch("/v1/detections", { method: "POST", body });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `模型服务返回 ${response.status}`);
    }
    const payload = await response.json();
    state.detections = payload.detections;
    elements.statLatency.textContent = payload.inference_ms.toFixed(1);
    updateFps(performance.now());
    updateCurrentStats();
    appendEvents(filteredDetections());
    setNotice(
      `监控运行中：最近一次请求耗时 ${(performance.now() - startedAt).toFixed(0)} ms。`,
      "success",
    );
  } catch (error) {
    state.detections = [];
    updateCurrentStats();
    setNotice(`推理失败：${error.message}`, "error");
  } finally {
    state.inferenceBusy = false;
    scheduleInference();
  }
}

async function refreshCameras(selectedDeviceId = "") {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  elements.cameraSelect.innerHTML = "";
  cameras.forEach((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `摄像头 ${index + 1}`;
    elements.cameraSelect.append(option);
  });
  elements.cameraSelect.disabled = cameras.length === 0;
  if (selectedDeviceId && cameras.some((camera) => camera.deviceId === selectedDeviceId)) {
    elements.cameraSelect.value = selectedDeviceId;
  }
  return cameras;
}

function cameraErrorMessage(error) {
  if (error.name === "NotAllowedError") return "摄像头权限被拒绝，请在浏览器地址栏中允许访问。";
  if (error.name === "NotFoundError") return "未找到可用摄像头，请检查设备连接。";
  if (error.name === "NotReadableError") return "摄像头正被其他程序占用。";
  return `无法启动摄像头：${error.message}`;
}

async function startMonitor() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setNotice("当前浏览器或访问地址不支持摄像头，请使用 localhost 或 HTTPS。", "error");
    return;
  }
  elements.start.disabled = true;
  setNotice("正在请求摄像头权限……");
  try {
    const selectedDeviceId = elements.cameraSelect.value;
    const video = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } };
    state.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    elements.video.srcObject = state.stream;
    await elements.video.play();
    const activeTrack = state.stream.getVideoTracks()[0];
    await refreshCameras(activeTrack.getSettings().deviceId || selectedDeviceId);
    elements.canvas.width = elements.video.videoWidth || 1280;
    elements.canvas.height = elements.video.videoHeight || 720;
    state.running = true;
    state.detections = [];
    state.recentCompletions = [];
    elements.stop.disabled = false;
    elements.cameraState.textContent = "已连接";
    elements.cameraState.classList.add("connection-label--active");
    elements.monitorState.classList.add("monitor-state--active");
    elements.monitorState.lastChild.textContent = " 监控中";
    elements.resolution.textContent = `${elements.canvas.width} × ${elements.canvas.height}`;
    setNotice("摄像头已连接，正在进行模型推理。", "success");
    renderFrame();
    scheduleInference(0);
  } catch (error) {
    elements.start.disabled = false;
    setNotice(cameraErrorMessage(error), "error");
  }
}

function stopMonitor() {
  state.running = false;
  state.inferenceBusy = false;
  window.clearTimeout(state.inferenceTimer);
  cancelAnimationFrame(state.animationFrame);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  elements.video.srcObject = null;
  state.detections = [];
  state.recentCompletions = [];
  elements.start.disabled = false;
  elements.stop.disabled = true;
  elements.cameraState.textContent = "未连接";
  elements.cameraState.classList.remove("connection-label--active");
  elements.monitorState.classList.remove("monitor-state--active");
  elements.monitorState.lastChild.textContent = " 等待启动";
  elements.resolution.textContent = "— × —";
  elements.statFps.textContent = "0.0";
  elements.statLatency.textContent = "—";
  updateCurrentStats();
  drawIdle("监控已停止");
  setNotice("监控已停止，摄像头资源已释放。");
}

async function checkService() {
  try {
    const response = await fetch("/health/ready", { cache: "no-store" });
    if (!response.ok) throw new Error("模型未就绪");
    const payload = await response.json();
    setServiceReady(true, "模型服务就绪", payload.model_id);
  } catch (error) {
    setServiceReady(false, "模型服务不可用");
    if (!state.running) setNotice(`无法连接模型服务：${error.message}`, "error");
  }
}

async function refreshServiceMetrics() {
  try {
    const response = await fetch("/v1/metrics", { cache: "no-store" });
    if (!response.ok) return;
    const metrics = await response.json();
    elements.requestCount.textContent = String(metrics.requests.total);
    elements.errorCount.textContent = String(metrics.requests.error);
  } catch {
    // Readiness polling owns user-visible service errors.
  }
}

elements.interval.addEventListener("input", () => {
  elements.intervalValue.value = `${elements.interval.value} ms`;
});
elements.threshold.addEventListener("input", () => {
  elements.thresholdValue.value = Number(elements.threshold.value).toFixed(2);
  updateCurrentStats();
});
elements.start.addEventListener("click", startMonitor);
elements.stop.addEventListener("click", stopMonitor);
elements.cameraSelect.addEventListener("change", () => {
  if (state.running) {
    stopMonitor();
    startMonitor();
  }
});
elements.clearEvents.addEventListener("click", () => {
  elements.eventList.innerHTML = '<li class="event-empty">监控启动后将在这里显示检测记录</li>';
});
window.addEventListener("beforeunload", () => state.stream?.getTracks().forEach((track) => track.stop()));

drawIdle();
checkService();
refreshServiceMetrics();
if (navigator.mediaDevices?.enumerateDevices) {
  refreshCameras().catch(() => {});
}
window.setInterval(checkService, 5000);
window.setInterval(refreshServiceMetrics, 3000);

