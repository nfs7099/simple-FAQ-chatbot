/**
 * SourceDisplay Component
 * Collapsible list of source snippets for a bot message.
 * Expects: sources = [{ source: "file.pdf", page: 3, content: "..." }, ...]
 */

const SourceDisplay = ({ sources = [] }) => {
  const [open, setOpen] = React.useState(false);
  const [expanded, setExpanded] = React.useState({}); // per-item expand map

  if (!Array.isArray(sources) || sources.length === 0) return null;

  const toggleItem = (idx) =>
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));

  const renderContent = (text = "", isExpanded) => {
    const MAX = 300;
    if (isExpanded || text.length <= MAX) return text;
    return text.slice(0, MAX) + "…";
  };

  return (
    <div className="sources-container">
      <button
        className="sources-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open ? "true" : "false"}
        aria-controls="sources-list"
        title={open ? "Hide sources" : "Show sources"}
      >
        <i className={`fas fa-chevron-${open ? "up" : "down"}`} aria-hidden="true"></i>
        {open ? "Hide" : "Show"} Sources ({sources.length})
      </button>

      {open && (
        <div id="sources-list" className="sources-list">
          {sources.map((s, idx) => {
            const isExpanded = !!expanded[idx];
            return (
              <div className="source-item" key={idx}>
                <div className="source-meta">
                  Source: {s?.source || "Unknown"}
                  {typeof s?.page === "number" ? ` (Page ${s.page})` : ""}
                </div>

                <div className="source-content">
                  {renderContent(s?.content || "", isExpanded)}
                </div>

                {s?.content && s.content.length > 300 && (
                  <button
                    className="sources-toggle small"
                    onClick={() => toggleItem(idx)}
                    aria-expanded={isExpanded ? "true" : "false"}
                    title={isExpanded ? "Show less" : "Show more"}
                  >
                    <i
                      className={`fas fa-chevron-${isExpanded ? "up" : "down"}`}
                      aria-hidden="true"
                    ></i>
                    {isExpanded ? "Show less" : "Show more"}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
window.SourceDisplay = SourceDisplay;