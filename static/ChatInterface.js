/**
 * ChatInterface Component
 * Brand-agnostic chat UI that respects system readiness and shows sources
 */

const ChatInterface = ({ systemReady }) => {
  const UI = window.UI_CONFIG || { appName: "Simple FAQ Chatbot", assistantName: "Assistant" };

  const [messages, setMessages] = React.useState([]);
  const [inputValue, setInputValue] = React.useState("");
  const [isLoading, setIsLoading] = React.useState(false);
  const messagesEndRef = React.useRef(null);

  // Welcome message once on mount
  React.useEffect(() => {
    const welcome = {
      content: `Hello! I'm ${UI.assistantName}. Ask me anything about your documents.`,
      timestamp: new Date(),
      sources: [],
    };
    setMessages([{ type: "bot", message: welcome }]);
  }, [UI.assistantName]);

  // Auto-scroll to latest message
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Compose and push a user message
  const pushUserMessage = (text) => {
    const msg = { content: text, timestamp: new Date() };
    setMessages((prev) => [...prev, { type: "user", message: msg }]);
  };

  // Push a bot message (answer or error)
  const pushBotMessage = (payload) => {
    const msg = {
      content: payload.content,
      timestamp: payload.timestamp || new Date(),
      sources: payload.sources || [],
    };
    setMessages((prev) => [...prev, { type: "bot", message: msg }]);
  };

  const sendMessage = async () => {
    const question = (inputValue || "").trim();
    if (!question || isLoading || !systemReady) return;

    // append user message
    pushUserMessage(question);
    setInputValue("");

    // show typing indicator
    setIsLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify({ query: question }),
      });

      if (!res.ok) {
        // try to surface server detail if present
        let detail = `HTTP ${res.status}`;
        try {
          const j = await res.json();
          if (j?.detail) detail = j.detail;
        } catch {}
        throw new Error(detail);
      }

      const data = await res.json();
      pushBotMessage({
        content: data.answer || "I couldn't generate an answer.",
        sources: Array.isArray(data.sources) ? data.sources : [],
      });
    } catch (err) {
      pushBotMessage({
        content: `Sorry, I hit an error: ${err?.message || err}. Please try again.`,
        sources: [],
      });
    } finally {
      setIsLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const placeholder = systemReady
    ? "Type your question here…"
    : "System initializing…";

  return (
    <div className="chat-interface">
      <div className="chat-messages" id="chat-messages">
        {messages.map((m, idx) => (
          <ChatMessage key={idx} type={m.type} message={m.message} />
        ))}

        {isLoading && (
          <div className="message bot loading-message">
            <div className="message-content">
              <div className="loading-dots">
                <span></span><span></span><span></span>
              </div>
            </div>
            <div className="message-meta">
              <span>{UI.assistantName}</span>
              <span>•</span>
              <span>{new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          id="chat-input"
          className="chat-input"
          placeholder={placeholder}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          disabled={!systemReady || isLoading}
        />
        <button
          id="send-button"
          className="send-button"
          onClick={sendMessage}
          disabled={!systemReady || isLoading || !inputValue.trim()}
          aria-label="Send"
          title="Send"
        >
          <i className="fas fa-paper-plane"></i>
        </button>
      </div>
    </div>
  );
};

window.ChatInterface = ChatInterface;