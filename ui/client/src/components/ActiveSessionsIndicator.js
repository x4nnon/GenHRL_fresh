import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Activity, 
  ChevronDown, 
  ChevronUp, 
  Clock, 
  Zap, 
  Play,
  X
} from 'lucide-react';
import { useActiveSessions } from '../context/ActiveSessionContext';

const ActiveSessionsIndicator = () => {
  const navigate = useNavigate();
  const { 
    hasActiveSessions, 
    getActiveSessionsCount, 
    getAllActiveSessions,
    removeTaskGeneration,
    removeTrainingSession
  } = useActiveSessions();
  const [isExpanded, setIsExpanded] = useState(false);

  if (!hasActiveSessions()) {
    return null;
  }

  const activeSessions = getAllActiveSessions();
  const count = getActiveSessionsCount();

  const formatElapsedTime = (startTime) => {
    const elapsed = Math.floor((Date.now() - new Date(startTime)) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    
    if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    }
    return `${seconds}s`;
  };

  const handleSessionClick = (session) => {
    if (session.type === 'generation') {
      navigate('/create');
    } else if (session.type === 'training') {
      const ts = Date.now();
      navigate(`/tasks/${session.taskName}?robot=${session.robot}&training=${session.sessionId}&ts=${ts}`);
    }
  };

  const handleRemoveSession = (session, e) => {
    e.stopPropagation();
    
    if (session.type === 'generation') {
      if (window.confirm('Cancel task generation? This will stop the generation process.')) {
        removeTaskGeneration();
      }
    } else if (session.type === 'training') {
      if (window.confirm('Stop monitoring this training session? (Training will continue in background)')) {
        removeTrainingSession(session.sessionId);
      }
    }
  };

  return (
    <div className="border-t border-gray-200 pt-4">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-2 hover:bg-gray-100 rounded-lg transition-colors"
      >
        <div className="flex items-center space-x-2">
          <div className="p-1 bg-orange-100 text-orange-600 rounded">
            <Activity size={14} />
          </div>
          <span className="text-sm font-medium text-gray-900">
            Active Sessions ({count})
          </span>
          <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
        </div>
        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-2">
          {activeSessions.map((session, index) => (
            <div
              key={`${session.type}-${session.sessionId || session.taskName}`}
              className="bg-orange-50 border border-orange-200 rounded-lg p-3 cursor-pointer hover:bg-orange-100 transition-colors"
              onClick={() => handleSessionClick(session)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    {session.type === 'generation' ? (
                      <Zap size={12} className="text-orange-600 flex-shrink-0" />
                    ) : (
                      <Play size={12} className="text-orange-600 flex-shrink-0" />
                    )}
                    <span className="text-xs font-medium text-orange-900 truncate">
                      {session.type === 'generation' ? 'Generating' : 'Training'}
                    </span>
                  </div>
                  
                  <div className="text-xs text-orange-800 font-medium truncate mb-1">
                    {session.taskName}
                  </div>
                  
                  {session.robot && (
                    <div className="text-xs text-orange-700 mb-1">
                      Robot: {session.robot}
                    </div>
                  )}
                  
                  <div className="flex items-center space-x-1 text-xs text-orange-600">
                    <Clock size={10} />
                    <span>{formatElapsedTime(session.startTime)}</span>
                  </div>
                </div>
                
                <button
                  onClick={(e) => handleRemoveSession(session, e)}
                  className="p-1 text-orange-600 hover:text-red-600 hover:bg-red-100 rounded transition-colors"
                  title="Remove from active sessions"
                >
                  <X size={12} />
                </button>
              </div>
              
              <div className="mt-2 text-xs text-orange-700">
                Click to return to {session.type === 'generation' ? 'task generation' : 'training monitor'}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ActiveSessionsIndicator;