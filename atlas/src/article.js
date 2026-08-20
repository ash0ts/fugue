const progress = document.querySelector(".reading-progress span");

const updateProgress = () => {
  if (!progress) return;
  const maximum = document.documentElement.scrollHeight - window.innerHeight;
  const fraction = maximum > 0 ? Math.min(1, window.scrollY / maximum) : 0;
  progress.style.transform = `scaleX(${fraction})`;
};

window.addEventListener("scroll", updateProgress, { passive: true });
window.addEventListener("resize", updateProgress);
updateProgress();

const formatTime = (seconds) => {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
};

const players = new Map();

const setupPlayer = (root) => {
  const video = root.querySelector("video");
  const toggle = root.querySelector("[data-film-toggle]");
  const restart = root.querySelector("[data-film-restart]");
  const scrub = root.querySelector("[data-film-scrub]");
  const clock = root.querySelector("[data-film-clock]");
  const screen = root.querySelector("[data-film-screen]");
  if (
    !(video instanceof HTMLVideoElement) ||
    !(toggle instanceof HTMLButtonElement) ||
    !(restart instanceof HTMLButtonElement) ||
    !(scrub instanceof HTMLInputElement) ||
    !(clock instanceof HTMLOutputElement)
  ) {
    return null;
  }

  const update = () => {
    const duration = Number.isFinite(video.duration)
      ? video.duration
      : Number(scrub.max);
    scrub.max = String(duration);
    scrub.value = String(video.currentTime);
    clock.value = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
    toggle.textContent = video.paused ? "Play" : "Pause";
    toggle.setAttribute(
      "aria-label",
      video.paused ? "Play analytical film" : "Pause analytical film",
    );
  };

  toggle.addEventListener("click", () => {
    if (video.paused) {
      video.play().catch(() => undefined);
    } else {
      video.pause();
    }
  });
  restart.addEventListener("click", () => {
    video.currentTime = 0;
    video.play().catch(() => undefined);
  });
  scrub.addEventListener("input", () => {
    video.currentTime = Number(scrub.value);
    update();
  });
  video.addEventListener("loadedmetadata", update);
  video.addEventListener("timeupdate", update);
  video.addEventListener("play", update);
  video.addEventListener("pause", update);
  video.addEventListener("ended", update);
  video.addEventListener("keydown", (event) => {
    if (event.key !== " " && event.key !== "k") return;
    event.preventDefault();
    toggle.click();
  });
  if (screen instanceof HTMLButtonElement) {
    screen.addEventListener("click", () => {
      root.requestFullscreen?.().catch(() => undefined);
    });
  }
  update();

  const api = {
    root,
    video,
    pause: () => video.pause(),
    seek: (time) => {
      video.currentTime = Number(time) || 0;
      update();
    },
    play: () => video.play().catch(() => undefined),
  };
  if (video.id) players.set(video.id, api);
  root.__filmPlayer = api;
  return api;
};

for (const root of document.querySelectorAll("[data-film-player]")) {
  setupPlayer(root);
}

for (const control of document.querySelectorAll("[data-film-seek]")) {
  control.addEventListener("click", () => {
    const player = players.get(control.dataset.filmTarget);
    const time = Number(control.dataset.filmSeek);
    if (!player || !Number.isFinite(time)) return;
    player.seek(time);
    player.play();
    player.video.focus({ preventScroll: true });
  });
}

for (const opener of document.querySelectorAll("[data-film-open]")) {
  opener.addEventListener("click", () => {
    const dialog = document.getElementById(opener.dataset.filmOpen);
    const inlineRoot = opener.closest("[data-film-player]");
    const dialogRoot = dialog?.querySelector("[data-film-dialog-player]");
    const inlinePlayer = inlineRoot?.__filmPlayer;
    const dialogPlayer = dialogRoot?.__filmPlayer;
    if (
      !(dialog instanceof HTMLDialogElement) ||
      !inlinePlayer ||
      !dialogPlayer
    ) {
      return;
    }
    inlinePlayer.pause();
    dialogPlayer.seek(inlinePlayer.video.currentTime);
    dialog.showModal();
    dialog.querySelector("[data-film-close]")?.focus();
  });
}

for (const dialog of document.querySelectorAll(".film-dialog")) {
  const close = dialog.querySelector("[data-film-close]");
  const dialogPlayer = dialog.querySelector("[data-film-dialog-player]")?.__filmPlayer;
  const inlinePlayer = dialog
    .closest(".film-block")
    ?.querySelector(".film-player:not(.is-dialog)")
    ?.__filmPlayer;

  const synchronizeAndClose = () => {
    if (dialogPlayer && inlinePlayer) {
      dialogPlayer.pause();
      inlinePlayer.seek(dialogPlayer.video.currentTime);
    }
    dialog.close();
  };

  close?.addEventListener("click", synchronizeAndClose);
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    synchronizeAndClose();
  });
}
