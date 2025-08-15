/**
 * StatusModal Component
 * Props:
 *  - isOpen: boolean
 *  - onClose: () => void
 *  - status: {
 *      status: "ready" | "not ready" | string,
 *      pdf_count: number,
 *      vector_db_initialized: boolean,
 *      llm_initialized: boolean,
 *      pdfs: string[]
 *    }
 */

const StatusModal = ({ isOpen, onClose, status }) => {
  const UI = window.UI_CONFIG || { appName: "Simple FAQ Chatbot" };
  const closeBtnRef = React.useRef(null);
  const dialogId = "status-dialog-title";

  const safe = status || {
    status: "not ready",
    pdf_count: 0,
    vector_db_initialized: false,
    llm_initialized: false,
    pdfs: [],
  };

  const overallReady = safe.status === "ready";

  const iconClass = (ok, warn = false) =>
    `fas ${ok ? "fa-circle-check" : warn ? "fa-triangle-exclamation" : "fa-circle-xmark"} status-icon ${ok ? "success" : warn ? "warning" : "error"}`;

  React.useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    document.addEventListener("keydown", onKey);
    // focus the close button after paint
    const t = setTimeout(() => closeBtnRef.current?.focus(), 0);
    return () => {
      document.removeEventListener("keydown", onKey);
      clearTimeout(t);
    };
  }, [isOpen, onClose]);

  const stop = (e) => e.stopPropagation();

  if (!isOpen) return null;

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby={dialogId}
      style={{ display: "flex" }}
    >
      <div className="modal-content" onClick={stop}>
        <div className="modal-header">
          <h2 id={dialogId} className="modal-title">
            System Status • {UI.appName}
          </h2>
          <button
            className="close-button"
            id="close-modal"
            onClick={onClose}
            ref={closeBtnRef}
            aria-label="Close status dialog"
            title="Close"
          >
            ×
          </button>
        </div>

        <div className="modal-body" id="modal-body">
          {/* LLM status */}
          <div className="status-item">
            <i className={iconClass(safe.llm_initialized)} aria-hidden="true"></i>
            <div>
              <div className="status-label">LLM Model</div>
              <div>{safe.llm_initialized ? "Loaded and ready" : "Not initialized"}</div>
            </div>
          </div>

          {/* Vector DB status */}
          <div className="status-item">
            <i className={iconClass(safe.vector_db_initialized)} aria-hidden="true"></i>
            <div>
              <div className="status-label">Vector Database</div>
              <div>{safe.vector_db_initialized ? "Initialized" : "Not initialized"}</div>
            </div>
          </div>

          {/* PDFs */}
          <div className="status-item">
            <i
              className={iconClass(safe.pdf_count > 0, safe.pdf_count === 0)}
              aria-hidden="true"
            ></i>
            <div>
              <div className="status-label">PDF Documents</div>
              <div>
                {safe.pdf_count > 0
                  ? `${safe.pdf_count} document${safe.pdf_count !== 1 ? "s" : ""} loaded`
                  : "No documents found"}
              </div>
              {Array.isArray(safe.pdfs) && safe.pdfs.length > 0 && (
                <div className="pdf-list">
                  {safe.pdfs.map((name, i) => (
                    <div className="pdf-item" key={`${name}-${i}`}>
                      {name}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Overall */}
          <div className="status-item">
            <i className={iconClass(overallReady)} aria-hidden="true"></i>
            <div>
              <div className="status-label">Overall Status</div>
              <div>{overallReady ? "System is ready" : "System is not ready"}</div>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="modal-button primary-button" id="close-modal-button" onClick={onClose}>
            Close
          </button>
          <button
            className="modal-button"
            onClick={() => window.refreshStatus?.()}
            title="Refresh status"
          >
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
};
window.StatusModal = StatusModal;