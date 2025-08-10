/**
 * Main Application Script
 * Initializes the React application and handles system status
 */

// Root component
const App = () => {
    console.log("App component rendering");
    const [systemStatus, setSystemStatus] = React.useState({
        status: 'not_ready',
        pdf_count: 0,
        vector_db_initialized: false,
        llm_initialized: false,
        pdfs: []
    });
    const [isStatusModalOpen, setIsStatusModalOpen] = React.useState(false);
    const [isLoading, setIsLoading] = React.useState(true);
    
    // Fetch system status on component mount and periodically
    React.useEffect(() => {
        fetchSystemStatus();
        
        // Poll for status updates every 10 seconds
        const intervalId = setInterval(fetchSystemStatus, 10000);
        
        // Clean up interval on unmount
        return () => clearInterval(intervalId);
    }, []);
    
    const fetchSystemStatus = async () => {
        try {
            const response = await fetch('/api/status');
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const data = await response.json();
            setSystemStatus(data);
            setIsLoading(false);
            
            // Update status indicator
            updateStatusIndicator(data.status === 'ready');
            
        } catch (error) {
            console.error('Error fetching system status:', error);
            setIsLoading(false);
            
            // Update status indicator to show error
            updateStatusIndicator(false);
        }
    };
    
    const updateStatusIndicator = (isReady) => {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (statusDot && statusText) {
            // Remove existing classes
            statusDot.classList.remove('online', 'offline', 'loading');
            
            if (isLoading) {
                statusDot.classList.add('loading');
                statusText.textContent = 'Checking system status...';
            } else if (isReady) {
                statusDot.classList.add('online');
                statusText.textContent = 'System Ready';
            } else {
                statusDot.classList.add('offline');
                statusText.textContent = 'System Not Ready';
            }
        }
    };
    
    const handleStatusClick = () => {
        setIsStatusModalOpen(true);
    };
    
    const closeStatusModal = () => {
        setIsStatusModalOpen(false);
    };
    
    // Add click event listener to status indicator
    React.useEffect(() => {
        const statusIndicator = document.getElementById('status-indicator');
        
        if (statusIndicator) {
            statusIndicator.addEventListener('click', handleStatusClick);
            
            // Clean up event listener on unmount
            return () => {
                statusIndicator.removeEventListener('click', handleStatusClick);
            };
        }
    }, []);
    
    console.log("System status:", systemStatus);
    return (
        <React.Fragment>
            <div style={{padding: '20px', backgroundColor: '#f0f0f0', marginBottom: '10px', borderRadius: '5px'}}>
                <h3>Debug Info</h3>
                <p>Status: {systemStatus.status}</p>
                <p>PDF Count: {systemStatus.pdf_count}</p>
                <p>Vector DB: {systemStatus.vector_db_initialized ? 'Initialized' : 'Not Initialized'}</p>
                <p>LLM: {systemStatus.llm_initialized ? 'Initialized' : 'Not Initialized'}</p>
            </div>
            <ChatInterface systemReady={systemStatus.status === 'ready'} />
            <StatusModal 
                isOpen={isStatusModalOpen} 
                onClose={closeStatusModal} 
                status={systemStatus} 
            />
        </React.Fragment>
    );
};

// Render the app
ReactDOM.render(
    <App />,
    document.getElementById('chat-container')
);
