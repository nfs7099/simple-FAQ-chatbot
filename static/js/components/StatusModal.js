/**
 * StatusModal Component
 * Displays system status information
 */

const StatusModal = ({ isOpen, onClose, status }) => {
    if (!isOpen) return null;
    
    return (
        <div className="modal-backdrop">
            <div className="modal-content">
                <div className="modal-header">
                    <h2 className="modal-title">System Status</h2>
                    <button className="close-button" onClick={onClose}>×</button>
                </div>
                
                <div className="modal-body">
                    <div className="status-item">
                        <i className={`fas fa-server status-icon ${status.llm_initialized ? 'success' : 'error'}`}></i>
                        <div>
                            <div className="status-label">LLM Model</div>
                            <div>{status.llm_initialized ? 'Loaded and ready' : 'Not initialized'}</div>
                        </div>
                    </div>
                    
                    <div className="status-item">
                        <i className={`fas fa-database status-icon ${status.vector_db_initialized ? 'success' : 'error'}`}></i>
                        <div>
                            <div className="status-label">Vector Database</div>
                            <div>{status.vector_db_initialized ? 'Initialized' : 'Not initialized'}</div>
                        </div>
                    </div>
                    
                    <div className="status-item">
                        <i className={`fas fa-file-pdf status-icon ${status.pdf_count > 0 ? 'success' : 'warning'}`}></i>
                        <div>
                            <div className="status-label">PDF Documents</div>
                            <div>
                                {status.pdf_count > 0 
                                    ? `${status.pdf_count} document${status.pdf_count !== 1 ? 's' : ''} loaded` 
                                    : 'No documents found'}
                            </div>
                            {status.pdfs && status.pdfs.length > 0 && (
                                <div className="pdf-list">
                                    {status.pdfs.map((pdf, index) => (
                                        <div key={index} className="pdf-item">{pdf}</div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                    
                    <div className="status-item">
                        <i className={`fas fa-circle-check status-icon ${status.status === 'ready' ? 'success' : 'error'}`}></i>
                        <div>
                            <div className="status-label">Overall Status</div>
                            <div>{status.status === 'ready' ? 'System is ready' : 'System is not ready'}</div>
                        </div>
                    </div>
                </div>
                
                <div className="modal-footer">
                    <button className="modal-button primary-button" onClick={onClose}>Close</button>
                </div>
            </div>
        </div>
    );
};
