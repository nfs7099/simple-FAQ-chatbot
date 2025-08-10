/**
 * ChatMessage Component
 * Displays a single message in the chat interface
 */

const ChatMessage = ({ message, type }) => {
    const formatTime = (timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <div className={`message ${type}`}>
            <div className="message-content">
                {message.content}
            </div>
            <div className="message-meta">
                <span>{type === 'user' ? 'You' : 'E-waste Assistant'}</span>
                <span>•</span>
                <span>{formatTime(message.timestamp)}</span>
            </div>
            {type === 'bot' && message.sources && message.sources.length > 0 && (
                <SourceDisplay sources={message.sources} />
            )}
        </div>
    );
};
