<script lang="ts">
  import { onMount } from "svelte";
  import YOLO from "@ultralytics/yolo";

  // Рамка обнаруженного следа с названием вида
  interface Box {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    conf: number;
    cls: number;
    name: string;
    color: string;
  }

  // Результат обнаружения
  interface PredictResult {
    task: string;
    width: number;
    height: number;
    boxes: Box[];
    obb: unknown[];
    keypoints: unknown[];
    masks: Record<string, unknown>;
    depth: Record<string, unknown>;
    speed: {
      preprocess: number;
      inference: number;
      postprocess: number;
    };
  }

  // Поля ввода
  let fileInput: HTMLInputElement;
  let cameraInput: HTMLInputElement;

  // Входное и выходное изображения
  let selectedImage: string | null = null;
  let annotatedImage: string | null = null;

  // Уверенность модели
  let confidence = 0.25;

  // Состояния
  let isLoading = false;
  let isModelReady = false;
  let errorMessage = "";

  let lastResult: PredictResult | null = null;

  // Модель
  let model: any = null;

  onMount(async () => {
    try {
      // Загрузка модели
      model = await YOLO.load("/models/model.onnx", { device: "cpu" });
      isModelReady = true;
    } catch (e: any) {
      errorMessage = `Ошибка загрузки модели: ${e.message}`;
    }
  });

  // Изменение входного изображения
  function handleFileChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      const reader = new FileReader();
      reader.onload = (e) => {
        selectedImage = e.target?.result as string;
        annotatedImage = null;
        lastResult = null;
      };
      reader.readAsDataURL(input.files[0]);
    }
  }

  // Запуск распознавания
  async function runInference() {
    if (!selectedImage || !model) return;

    isLoading = true;
    errorMessage = "";
    annotatedImage = null;
    lastResult = null;

    try {
      const img = new Image();
      img.src = selectedImage;
      await img.decode();

      // Получение результата распознавания моделью
      const result: PredictResult = await model.predict(img, {
        conf: confidence,
      });

      // Отрисовка рамок
      const canvas = document.createElement("canvas");
      canvas.width = result.width || img.width;
      canvas.height = result.height || img.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const boxes = result.boxes ?? [];

      // Сортировка рамок по возрастанию уверенности
      const boxesToDraw = [...boxes].sort((a, b) => a.conf - b.conf);

      for (const box of boxesToDraw) {
        const { x1, y1, x2, y2, conf, name, color } = box;
        const w = x2 - x1;
        const h = y2 - y1;
        const lineColor = color || "#22c55e";

        // Рамка
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = Math.max(2, Math.round(canvas.width / 400));
        ctx.strokeRect(x1, y1, w, h);

        // Подпись вида
        const label = `${name} ${(conf * 100).toFixed(0)}%`;
        const fontSize = Math.max(14, Math.round(canvas.width / 45));
        ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
        const textWidth = ctx.measureText(label).width;
        const padX = 8;
        const padY = 4;
        const labelH = fontSize + padY * 2;
        const labelY = y1 - labelH < 0 ? y1 : y1 - labelH;

        // Фон подписи
        ctx.fillStyle = lineColor;
        ctx.beginPath();
        const radius = 6;
        roundRect(ctx, x1, labelY, textWidth + padX * 2, labelH, radius);
        ctx.fill();

        // Текст — контрастный к цвету рамки
        ctx.fillStyle = getTextColor(lineColor);
        ctx.textBaseline = "middle";
        ctx.fillText(label, x1 + padX, labelY + labelH / 2);
      }

      annotatedImage = canvas.toDataURL("image/png");

      // Список определенных видов по убыванию увеернности
      const sortedForList = [...boxes].sort((a, b) => b.conf - a.conf);
      lastResult = { ...result, boxes: sortedForList };
    } catch (e: any) {
      errorMessage = `Ошибка распознавания: ${e.message}`;
    } finally {
      isLoading = false;
    }
  }

  // Отрисовка загругленного фона
  function roundRect(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    r: number,
  ) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
  }

  // Получения цвета текста в зависимости от цвета рамки
  function getTextColor(background: string): string {
    let c = background.replace("#", "");
    if (c.length === 3) {
      c = c
        .split("")
        .map((ch) => ch + ch)
        .join("");
    }
    if (c.length !== 6) return "#ffffff";

    const r = parseInt(c.substring(0, 2), 16) / 255;
    const g = parseInt(c.substring(2, 4), 16) / 255;
    const b = parseInt(c.substring(4, 6), 16) / 255;

    const toLinear = (v: number) =>
      v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);

    const R = toLinear(r);
    const G = toLinear(g);
    const B = toLinear(b);

    const luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B;

    return luminance > 0.5 ? "#000000" : "#ffffff";
  }

  // Загрузка результата в память
  function downloadResult() {
    if (!annotatedImage) return;
    const link = document.createElement("a");
    link.href = annotatedImage;
    link.download = `detected_tracks_${Date.now()}.png`;
    link.click();
  }

  // Сброс всех параметров
  function reset() {
    selectedImage = null;
    annotatedImage = null;
    lastResult = null;
    errorMessage = "";
    if (fileInput) fileInput.value = "";
    if (cameraInput) cameraInput.value = "";
  }
</script>

<div class="page">
  <header class="app-header">
    <div class="brand">
      <div>
        <h1>Обнаружение следов животных</h1>
        <p class="subtitle">
          Загрузите изображение - модель найдёт и классифицирует следы
        </p>
      </div>
    </div>
    <div
      class="status-pill"
      class:ready={isModelReady}
      class:loading={!isModelReady}
    >
      {#if isModelReady}
        <span class="dot"></span> Модель загружена
      {:else}
        <span class="dot"></span> Загрузка модели…
      {/if}
    </div>
  </header>

  {#if errorMessage}
    <div class="alert">{errorMessage}</div>
  {/if}

  <main class="card-grid">
    <!-- Панель управления -->
    <section class="card controls">
      <h2>Изображение</h2>

      <div class="upload-area">
        {#if !selectedImage}
          <div class="dropzone">
            <div class="dropzone-icon">🖼️</div>
            <p>Выберите изображение или сделайте фото</p>
            <div class="upload-buttons">
              <button
                class="btn secondary"
                onclick={() => fileInput.click()}
                disabled={!isModelReady}
              >
                📁 Из галереи
              </button>
              <button
                class="btn secondary"
                onclick={() => cameraInput.click()}
                disabled={!isModelReady}
              >
                📷 Камера
              </button>
            </div>
          </div>
        {:else}
          <div class="preview-wrap">
            <img
              src={selectedImage}
              alt="Исходное изображение"
              class="preview"
            />
            <button class="change-btn" onclick={() => fileInput.click()}
              >Заменить</button
            >
          </div>
        {/if}

        <input
          bind:this={fileInput}
          type="file"
          accept="image/*"
          onchange={handleFileChange}
          hidden
        />
        <input
          bind:this={cameraInput}
          type="file"
          accept="image/*"
          capture="environment"
          onchange={handleFileChange}
          hidden
        />
      </div>

      <h2>Порог уверенности</h2>
      <div class="slider-row">
        <input
          id="confidence"
          type="range"
          min="0"
          max="1"
          step="0.01"
          bind:value={confidence}
          disabled={!selectedImage}
        />
        <span class="slider-value">{confidence.toFixed(2)}</span>
      </div>
      <p class="hint">Значения ниже порога будут отброшены моделью.</p>

      <div class="actions">
        <button
          class="btn primary"
          onclick={runInference}
          disabled={isLoading || !isModelReady || !selectedImage}
        >
          {#if isLoading}
            <span class="spinner"></span> Распознавание…
          {:else}
            Начать распознавание
          {/if}
        </button>
        {#if selectedImage}
          <button class="btn ghost" onclick={reset}>Сбросить</button>
        {/if}
      </div>
    </section>

    <!-- Панель результата -->
    <section class="card result">
      <h2>Результат</h2>

      {#if annotatedImage}
        <img src={annotatedImage} alt="Результат" class="preview" />

        {#if lastResult && lastResult.boxes.length > 0}
          <details class="objects">
            <summary>Обнаруженные следы ({lastResult.boxes.length})</summary>
            <ul>
              {#each lastResult.boxes as b}
                <li>
                  <span
                    class="color-dot"
                    style="background: {b.color || '#22c55e'}"
                  ></span>
                  <span class="obj-name">{b.name}</span>
                  <span class="obj-conf">{(b.conf * 100).toFixed(0)}%</span>
                </li>
              {/each}
            </ul>
          </details>
        {/if}

        <button class="btn success" onclick={downloadResult}
          >Скачать изображение</button
        >
      {:else}
        <div class="empty">
          <div class="empty-icon">🔍</div>
          <p>Здесь появится изображение с обнаруженными следами</p>
        </div>
      {/if}
    </section>
  </main>

  <section class="model-info card">
    <h2>Информация о модели</h2>
    <p>
      Данная модель распознавания основана на модели YOLO26 Small и позволяет
      находить на изображениях следы животных на разных поверхностях (грязь,
      снег, песок) и относить их к определенному виду.
    </p>
    <p>
      Всего модель способна распознать следы 43 различных видов животных,
      преимущественно обитающих на территории России:
    </p>
    <ul>
      <li>Медведь</li>
      <li>Барсук</li>
      <li>Бобр</li>
      <li>Крупнорогатый скот</li>
      <li>Баран</li>
      <li>Лось</li>
      <li>Олень</li>
      <li>Косуля</li>
      <li>Заяц</li>
      <li>Кролик</li>
      <li>Волк</li>
      <li>Домашняя собака</li>
      <li>Лисица</li>
      <li>Рысь</li>
      <li>Домашняя кошка</li>
      <li>Выдра</li>
      <li>Куница</li>
      <li>Росомаха</li>
      <li>Норка</li>
      <li>Ласка</li>
      <li>Енот</li>
      <li>Нутрия</li>
      <li>Кабан</li>
      <li>Белка</li>
      <li>Суслик</li>
      <li>Бурундук</li>
      <li>Крыса</li>
      <li>Мышь</li>
      <li>Сурок</li>
      <li>Утка</li>
      <li>Дикая курица</li>
      <li>Рябчик</li>
      <li>Голубь</li>
      <li>Ворон</li>
      <li>Сойка</li>
      <li>Воробей</li>
      <li>Дятел</li>
      <li>Орел</li>
      <li>Сова</li>
      <li>Цапля</li>
      <li>Чайка</li>
      <li>Лягушка</li>
      <li>Жаба</li>
    </ul>
  </section>
</div>
