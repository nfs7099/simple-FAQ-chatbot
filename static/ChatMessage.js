/**
 * ChatMessage Component
 * Displays a single message in the chat interface
 */

const ChatMessage = ({ message, type }) => {
  const UI = window.UI_CONFIG || { assistantName: "Assistant" };
  const isUser = type === "user";
  const senderLabel = isUser ? "You" : UI.assistantName || "Assistant";

  const formatTime = (ts) => {
    const d = ts ? new Date(ts) : new Date();
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className={`message ${type}`}>
      <div className="message-content">
        {message.content}
      </div>

      <div className="message-meta">
        <span>{senderLabel}</span>
        <span>•</span>
        <span>{formatTime(message.timestamp)}</span>
      </div>

      {type === "bot" && Array.isArray(message.sources) && message.sources.length > 0 && (
        <SourceDisplay sources={message.sources} />
      )}
    </div>
  );
};
window.ChatMessage = ChatMessage;