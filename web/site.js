/* fly-trader.app — site behaviour.
   Edit the CONFIG block at launch time; nothing else needs to change. */

const CONFIG = {
  GITHUB_URL: "https://github.com/bryceweiner/fly-trader",
  X_URL: "https://x.com/i/communities/2038854012578173030",
  FLY_CA: "0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3",
  VENUE_URL: "https://ponsfamily.com/coin/", // CA is appended
};

(function () {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- links driven by CONFIG ---- */
  document.querySelectorAll("[data-link]").forEach((el) => {
    const kind = el.dataset.link;
    if (kind === "github") el.href = CONFIG.GITHUB_URL;
    if (kind === "readme") el.href = CONFIG.GITHUB_URL + "#readme";
    if (kind === "x") {
      if (CONFIG.X_URL) {
        el.href = CONFIG.X_URL;
      } else {
        el.setAttribute("aria-disabled", "true");
        el.title = "X community link coming at launch";
        el.addEventListener("click", (e) => e.preventDefault());
      }
    }
  });

  /* ---- $FLY contract address ---- */
  const caEl = document.getElementById("fly-ca");
  const copyBtn = document.getElementById("copy-ca");
  const buyBtn = document.getElementById("buy-fly");
  if (caEl && copyBtn && buyBtn) {
    if (CONFIG.FLY_CA) {
      caEl.textContent = CONFIG.FLY_CA;
      copyBtn.removeAttribute("aria-disabled");
      buyBtn.removeAttribute("aria-disabled");
      buyBtn.href = CONFIG.VENUE_URL + CONFIG.FLY_CA;
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(CONFIG.FLY_CA);
          copyBtn.textContent = "Copied";
          setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
        } catch (_) {
          copyBtn.textContent = "Select";
          const range = document.createRange();
          range.selectNodeContents(caEl);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        }
      });
    } else {
      buyBtn.addEventListener("click", (e) => e.preventDefault());
    }
  }

  /* ---- mobile menu ---- */
  const burger = document.querySelector(".burger");
  const menu = document.getElementById("mobile-menu");
  if (burger && menu) {
    const close = () => {
      menu.hidden = true;
      burger.setAttribute("aria-expanded", "false");
      burger.setAttribute("aria-label", "Open menu");
    };
    burger.addEventListener("click", () => {
      const open = menu.hidden;
      menu.hidden = !open;
      burger.setAttribute("aria-expanded", String(open));
      burger.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });
    menu.querySelectorAll("a").forEach((a) => a.addEventListener("click", close));
  }

  /* ---- rotating hero headline ---- */
  const headline = document.getElementById("headline");
  if (headline && !reducedMotion) {
    const passages = Array.from(headline.querySelectorAll(".passage"));
    if (passages.length > 1) {
      let i = 0;
      const PERIOD = 4000;
      const LEAVE = 600;
      setInterval(() => {
        const cur = passages[i];
        i = (i + 1) % passages.length;
        const next = passages[i];
        cur.classList.remove("is-active");
        cur.classList.add("is-leaving");
        next.classList.add("is-active");
        setTimeout(() => cur.classList.remove("is-leaving"), LEAVE);
      }, PERIOD);
    }
  }

  /* ---- hero video: fall back to the still when playback is refused ---- */
  const video = document.querySelector(".hero-media video");
  if (video) {
    if (reducedMotion) {
      video.pause();
      video.removeAttribute("autoplay");
    } else {
      const fail = () => document.body.classList.add("no-video");
      video.addEventListener("error", fail, true);
      const p = video.play();
      if (p && typeof p.catch === "function") p.catch(fail);
    }
  }

  /* ---- footer year ---- */
  const y = document.getElementById("year");
  if (y) y.textContent = String(new Date().getFullYear());
})();
