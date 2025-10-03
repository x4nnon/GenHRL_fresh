import React, { createContext, useContext, useState, useEffect } from 'react';

const ActiveSessionContext = createContext();

export const useActiveSessions = () => {
  const context = useContext(ActiveSessionContext);
  if (!context) {
    throw new Error('useActiveSessions must be used within an ActiveSessionProvider');
  }
  return context;
};

export const ActiveSessionProvider = ({ children }) => {
  const [activeSessions, setActiveSessions] = useState({
    taskGeneration: null,
    training: []
  });

  // Load active sessions from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('genhrl_active_sessions');
    if (saved) {
      try {
        const sessions = JSON.parse(saved);
        setActiveSessions(sessions);
      } catch (error) {
        console.error('Error loading active sessions:', error);
      }
    }
  }, []);

  // Save to localStorage whenever sessions change
  useEffect(() => {
    localStorage.setItem('genhrl_active_sessions', JSON.stringify(activeSessions));
  }, [activeSessions]);

  const addTaskGeneration = (sessionId, taskName, startTime) => {
    setActiveSessions(prev => ({
      ...prev,
      taskGeneration: {
        sessionId,
        taskName,
        startTime,
        type: 'generation'
      }
    }));
  };

  const removeTaskGeneration = () => {
    setActiveSessions(prev => ({
      ...prev,
      taskGeneration: null
    }));
  };

  const addTrainingSession = (sessionId, taskName, robot, startTime) => {
    setActiveSessions(prev => ({
      ...prev,
      training: [
        ...prev.training.filter(t => !(t.taskName === taskName && t.robot === robot)),
        {
          sessionId,
          taskName,
          robot,
          startTime,
          type: 'training'
        }
      ]
    }));
  };

  const removeTrainingSession = (sessionId) => {
    setActiveSessions(prev => ({
      ...prev,
      training: prev.training.filter(t => t.sessionId !== sessionId)
    }));
  };

  const getActiveSessionsCount = () => {
    const generationCount = activeSessions.taskGeneration ? 1 : 0;
    const trainingCount = activeSessions.training.length;
    return generationCount + trainingCount;
  };

  const hasActiveSessions = () => {
    return activeSessions.taskGeneration || activeSessions.training.length > 0;
  };

  const getAllActiveSessions = () => {
    const sessions = [];
    
    if (activeSessions.taskGeneration) {
      sessions.push(activeSessions.taskGeneration);
    }
    
    sessions.push(...activeSessions.training);
    
    return sessions;
  };

  const value = {
    activeSessions,
    addTaskGeneration,
    removeTaskGeneration,
    addTrainingSession,
    removeTrainingSession,
    getActiveSessionsCount,
    hasActiveSessions,
    getAllActiveSessions
  };

  return (
    <ActiveSessionContext.Provider value={value}>
      {children}
    </ActiveSessionContext.Provider>
  );
};