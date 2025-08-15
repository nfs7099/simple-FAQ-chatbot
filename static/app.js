/**
 * /static/app.js
 * - UI boot (brand name + theme)
 * - React App (status polling + modal)
 * - Upload & Reindex controller (vanilla)
 */

/* =========================
   UI BOOT (brand + theme)
   ========================= */
(() => {
  async function loadConfig() {
    try {
      const res = await fetch("/api/ui-config", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch {
      // Safe defaults if backend route is missing
      return {
        appName: "Simple FAQ Chatbot",
        assistantName: "Assistant",
        defaultTheme: "emerald",
        themes: ["emerald", "indigo", "slate", "rose", "amber", "teal"],
      };
    }
  }

  function applyConfig(cfg) {
    // Expose globally for other scripts/components
    window.UI_CONFIG = cfg;

    // Title + header text
    try {
      document.title = cfg.appName || document.title;
      const h1 = document.querySelector("#appTitle, header .logo h1, header h1");
      if (h1) h1.textContent = cfg.appName;
    } catch {}

    // Theme (respect user override in localStorage)
    const saved = localStorage.getItem("theme");
    const theme =
      (saved && cfg.themes?.includes(saved)) ? saved : (cfg.defaultTheme || "emerald");
    document.documentElement.setAttribute("data-theme", theme);

    // Expose a setter for optional theme picker
    window.setTheme = (t) => {
      if (!cfg.themes?.includes(t)) return;
      document.documentElement.setAttribute("data-theme", t);
      localStorage.setItem("theme", t);
    };
  }

  loadConfig().then(applyConfig);
})();

/* =========================
   React App
   ========================= */
const App = () => {
  const [systemStatus, setSystemStatus] = React.useState({
    status: "not ready",               // mirrors API payload
    pdf_count: 0,
    vector_db_initialized: false,
    llm_initialized: false,
    pdfs: [],
  });
  const [isStatusModalOpen, setIsStatusModalOpen] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const mountedRef = React.useRef(true);

  const updateStatusIndicator = React.useCallback((isReady, isLoadingNow) => {
    const dot = document.querySelector(".status-dot");
    const text = document.querySelector(".status-text");
    if (!dot || !text) return;

    dot.classList.remove("online", "offline", "loading");

    if (isLoadingNow) {
      dot.classList.add("loading");
      text.textContent = "Checking system status...";
    } else if (isReady) {
      dot.classList.add("online");
      text.textContent = "System Ready";
    } else {
      dot.classList.add("offline");
      text.textContent = "System Not Ready";
    }
  }, []);

  const fetchSystemStatus = React.useCallback(async () => {
    // immediately show loading state on the badge
    updateStatusIndicator(false, true);
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!mountedRef.current) return;

      setSystemStatus(data);
      setLoading(false);
      updateStatusIndicator(data.status === "ready", false);
    } catch (err) {
      console.error("Error fetching system status:", err);
      if (!mountedRef.current) return;
      setLoading(false);
      updateStatusIndicator(false, false);
    }
  }, [updateStatusIndicator]);

  // initial fetch + 10s polling
  React.useEffect(() => {
    mountedRef.current = true;
    fetchSystemStatus();
    const intervalId = setInterval(fetchSystemStatus, 10000);
    return () => {
      mountedRef.current = false;
      clearInterval(intervalId);
    };
  }, [fetchSystemStatus]);

  // Make refresh callable from other scripts (upload/reindex controller)
  React.useEffect(() => {
    window.refreshStatus = fetchSystemStatus;
    return () => {
      if (window.refreshStatus === fetchSystemStatus) delete window.refreshStatus;
    };
  }, [fetchSystemStatus]);

  // Click-to-open status modal on the indicator (if present)
  React.useEffect(() => {
    const el = document.getElementById("status-indicator");
    if (!el) return;
    const onClick = () => setIsStatusModalOpen(true);
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, []);

  // Debug panel (handy while integrating; remove or hide via CSS later)
  return (
    <React.Fragment>
      <div
        style={{
          padding: "20px",
          backgroundColor: "#f0f0f0",
          marginBottom: "10px",
          borderRadius: "5px",
        }}
      >
        <h3>Debug Info</h3>
        <p>Status: {systemStatus.status}</p>
        <p>PDF Count: {systemStatus.pdf_count}</p>
        <p>Vector DB: {systemStatus.vector_db_initialized ? "Initialized" : "Not Initialized"}</p>
        <p>LLM: {systemStatus.llm_initialized ? "Initialized" : "Not Initialized"}</p>
      </div>

      <ChatInterface systemReady={systemStatus.status === "ready"} />
      <StatusModal
        isOpen={isStatusModalOpen}
        onClose={() => setIsStatusModalOpen(false)}
        status={systemStatus}
      />
    </React.Fragment>
  );
};

// Mount (React 18+ createRoot; fallback to legacy render)
(() => {
  const container = document.getElementById("chat-container");
  if (!container) return;
  if (ReactDOM.createRoot) {
    ReactDOM.createRoot(container).render(<App />);
  } else {
    ReactDOM.render(<App />, container);
  }
})();


// ===== Upload & Reindex Controller (robust binding) =====
(() => {
  const onReady = () => {
    const $ = (s) => document.querySelector(s);
    const fileInput = $("#fileInput");
    const btnUpload = $("#btnUpload");
    const btnReindex = $("#btnReindex");
    const btnReindexFull = $("#btnReindexFull");
    const btnRefresh = $("#btnRefreshStatus");

    // 👇 NEW: show selected filename & enable/disable Upload button
    const fileLabel = document.getElementById("fileLabel");
    const updatePickerUI = () => {
      const name = fileInput?.files && fileInput.files[0] ? fileInput.files[0].name : "";
      if (name) {
        if (fileLabel) fileLabel.textContent = name;
        btnUpload?.removeAttribute("disabled");
      } else {
        if (fileLabel) fileLabel.textContent = "Choose PDF…";
        btnUpload?.setAttribute("disabled", "");
      }
    };
    fileInput?.addEventListener("change", updatePickerUI);
    updatePickerUI(); // initialize on load

    // … keep the rest of your controller as-is …

    async function upload() {
      // (unchanged validation)
      // ...
      try {
        // (send to /api/upload)
        // ...
        // on success, reset picker UI
        if (fileInput) fileInput.value = "";
        updatePickerUI();                 // 👈 reset label + disable button
        window.refreshStatus?.();
      } catch (e) {
        // ...
      } finally {
        btnUpload?.removeAttribute("disabled");
      }
    }

    // bindings (unchanged)
    btnUpload?.addEventListener("click", (e) => { e.preventDefault(); upload(); });
    btnReindex?.addEventListener("click", (e) => { e.preventDefault(); reindex("incremental"); });
    btnReindexFull?.addEventListener("click", (e) => { e.preventDefault(); reindex("full"); });
    btnRefresh?.addEventListener("click", (e) => { e.preventDefault(); window.refreshStatus?.(); });

    // (keep your delegated safety-net click handler if you added one)
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onReady, { once: true });
  } else {
    onReady();
  }
})();
