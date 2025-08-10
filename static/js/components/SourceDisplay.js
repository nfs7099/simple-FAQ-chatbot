/**
 * SourceDisplay Component
 * Displays the sources used for a bot response
 */

const SourceDisplay = ({ sources }) => {
    const [showSources, setShowSources] = React.useState(false);

    const toggleSources = () => {
        setShowSources(!showSources);
    };

    return (
        <div className="sources-container">
            <button className="sources-toggle" onClick={toggleSources}>
                <i className={`fas ${showSources ? 'fa-chevron-up' : 'fa-chevron-down'}`}></i>
                {showSources ? 'Hide Sources' : `Show Sources (${sources.length})`}
            </button>
            
            {showSources && (
                <div className="sources-list">
                    {sources.map((source, index) => (
                        <div key={index} className="source-item">
                            <div className="source-meta">
                                Source: {source.source} {source.page && `(Page ${source.page})`}
                            </div>
                            <div className="source-content">
                                {source.content.length > 200 
                                    ? `${source.content.substring(0, 200)}...` 
                                    : source.content}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
