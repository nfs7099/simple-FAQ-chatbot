/**
 * ChatInterface Component
 * Main component for the chat functionality
 */

const ChatInterface = ({ systemReady }) => {
    console.log("ChatInterface component rendering, systemReady:", systemReady);
    const [messages, setMessages] = React.useState([]);
    const [inputValue, setInputValue] = React.useState('');
    const [isLoading, setIsLoading] = React.useState(false);
    const messagesEndRef = React.useRef(null);
    
    // Add welcome message when component mounts
    React.useEffect(() => {
        const welcomeMessage = {
            content: "Hello! I'm your E-waste FAQ assistant. Ask me any questions about electronic waste recycling, disposal, or environmental impact.",
            timestamp: new Date(),
            sources: []
        };
        setMessages([{ type: 'bot', message: welcomeMessage }]);
    }, []);
    
    // Scroll to bottom when messages change
    React.useEffect(() => {
        scrollToBottom();
    }, [messages]);
    
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };
    
    const handleInputChange = (e) => {
        setInputValue(e.target.value);
    };
    
    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };
    
    const handleSendMessage = async () => {
        if (inputValue.trim() === '' || isLoading || !systemReady) return;
        
        const userMessage = {
            content: inputValue.trim(),
            timestamp: new Date(),
        };
        
        // Add user message to chat
        setMessages(prevMessages => [
            ...prevMessages, 
            { type: 'user', message: userMessage }
        ]);
        
        // Clear input
        setInputValue('');
        
        // Show loading state
        setIsLoading(true);
        
        try {
            // Send request to API
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userMessage.content }),
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add bot response to chat
            const botMessage = {
                content: data.answer,
                timestamp: new Date(),
                sources: data.sources || []
            };
            
            setMessages(prevMessages => [
                ...prevMessages, 
                { type: 'bot', message: botMessage }
            ]);
            
        } catch (error) {
            console.error('Error querying API:', error);
            
            // Add error message
            const errorMessage = {
                content: `Sorry, I encountered an error: ${error.message}. Please try again later.`,
                timestamp: new Date(),
                sources: []
            };
            
            setMessages(prevMessages => [
                ...prevMessages, 
                { type: 'bot', message: errorMessage }
            ]);
        } finally {
            setIsLoading(false);
        }
    };
    
    console.log("Messages:", messages);
    return (
        <div className="chat-interface">
            <div style={{padding: '10px', backgroundColor: '#e0f7fa', marginBottom: '10px', borderRadius: '5px'}}>
                <h4>ChatInterface Debug</h4>
                <p>System Ready: {systemReady ? 'Yes' : 'No'}</p>
                <p>Message Count: {messages.length}</p>
                <p>Input Value: {inputValue}</p>
                <p>Is Loading: {isLoading ? 'Yes' : 'No'}</p>
            </div>
            <div className="chat-messages">
                {messages.map((item, index) => (
                    <ChatMessage 
                        key={index} 
                        message={item.message} 
                        type={item.type} 
                    />
                ))}
                
                {isLoading && (
                    <div className="message bot">
                        <div className="message-content">
                            <div className="loading-dots">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}
                
                <div ref={messagesEndRef} />
            </div>
            
            <div className="chat-input-container">
                <textarea
                    className="chat-input"
                    value={inputValue}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder={systemReady ? "Type your question here..." : "System initializing..."}
                    disabled={!systemReady || isLoading}
                    rows="1"
                />
                <button 
                    className="send-button" 
                    onClick={handleSendMessage}
                    disabled={!systemReady || isLoading || inputValue.trim() === ''}
                >
                    <i className="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    );
};
