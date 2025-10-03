import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, AlertCircle, CheckCircle, Clock, Zap, X } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { useActiveSessions } from '../context/ActiveSessionContext';

const CreateTask = () => {
  const navigate = useNavigate();
  const terminalRef = useRef(null);
  const { addTaskGeneration, removeTaskGeneration } = useActiveSessions();
  const [formData, setFormData] = useState({
    taskName: '',
    taskDescription: '',
    robot: 'G1',
    maxHierarchyLevels: 3,
    apiKey: '',
    model: 'gemini-2.5-flash',
    backupModel: 'gemini-2.5-pro'
  });
  const [robots, setRobots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [progress, setProgress] = useState({
    isTracking: false,
    currentStage: 0,
    totalStages: 6,
    stage: '',
    message: '',
    details: '',
    debugMessages: [],
    isConnected: false,
    sessionId: null
  });

  useEffect(() => {
    fetchRobots();
    // Load API key from localStorage if available
    const savedApiKey = localStorage.getItem('genhrl_api_key');
    if (savedApiKey) {
      setFormData(prev => ({ ...prev, apiKey: savedApiKey }));
    }
  }, []);

  // No auto-scroll - users have manual control over terminal scrolling

  const fetchRobots = async () => {
    try {
      const response = await axios.get('/api/robots');
      setRobots(response.data);
    } catch (error) {
      console.error('Error fetching robots:', error);
      toast.error('Failed to fetch available robots');
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.taskName.trim()) {
      newErrors.taskName = 'Task name is required';
    } else if (!/^[a-zA-Z0-9_]+$/.test(formData.taskName)) {
      newErrors.taskName = 'Task name can only contain letters, numbers, and underscores';
    }
    
    if (!formData.taskDescription.trim()) {
      newErrors.taskDescription = 'Task description is required';
    } else if (formData.taskDescription.length < 20) {
      newErrors.taskDescription = 'Description should be at least 20 characters';
    }
    
    if (!formData.apiKey.trim()) {
      newErrors.apiKey = 'API key is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const startProgressTracking = (sessionId) => {
    console.log('[DEBUG] Starting SSE progress tracking for session:', sessionId);

    setProgress(prev => ({ ...prev, isTracking: true, debugMessages: [], isConnected: true, sessionId }));

    // Track active sessions for persistence
    addTaskGeneration(sessionId, formData.taskName, new Date().toISOString());

    // Establish EventSource connection
    let apiBase = '';
    if (window.location.port === '3000') {
      apiBase = 'http://localhost:5000';
    }
    const streamUrl = `${apiBase}/api/progress/${sessionId}`;
    console.log('[DEBUG] Connecting to progress SSE stream:', streamUrl);

    const eventSource = new EventSource(streamUrl);

    // Store reference for cleanup
    window.__genhrlProgressES = eventSource;

    eventSource.onopen = () => {
      console.log('[DEBUG] SSE connection opened for generation progress');
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
          setProgress(prev => ({
            ...prev,
            currentStage: parseInt(data.currentStage) || 0,
            totalStages: parseInt(data.totalStages) || 7,
            stage: data.stage,
            message: data.message,
            details: data.details || '',
            debugMessages: [...prev.debugMessages, {
              timestamp: new Date().toLocaleTimeString(),
              type: 'progress',
              message: `Stage ${data.currentStage}/${data.totalStages}: ${data.message}`
            }].slice(-200)
          }));
        } else if (data.type === 'terminal') {
          setProgress(prev => ({
            ...prev,
            debugMessages: [...prev.debugMessages, {
              timestamp: new Date(data.timestamp).toLocaleTimeString(),
              type: 'info',
              message: data.line
            }].slice(-1000)
          }));
        } else if (data.type === 'complete') {
          // Close connection
          eventSource.close();
          removeTaskGeneration();

          setProgress(prev => ({
            ...prev,
            isTracking: false,
            isConnected: false,
            sessionId: null,
            debugMessages: [...prev.debugMessages, {
              timestamp: new Date().toLocaleTimeString(),
              type: data.success ? 'success' : 'error',
              message: data.message || (data.success ? 'Task creation completed!' : 'Task creation failed')
            }]
          }));

          setLoading(false);

          if (data.success) {
            toast.success('Task created successfully!');
            navigate(`/tasks/${data.taskName}?robot=${formData.robot}`);
          } else {
            toast.error(data.error || 'Task creation failed');
          }
        }
      } catch (err) {
        console.error('[DEBUG] Error parsing SSE data:', err);
      }
    };

    eventSource.onerror = (err) => {
      console.error('[DEBUG] SSE error:', err);
      // Connection will auto-reconnect; optionally add handling here
    };

    // Cleanup
    return () => {
      console.log('[DEBUG] Closing SSE connection');
      eventSource.close();
    };
  };

  const cancelTaskCreation = async () => {
    if (!progress.sessionId) {
      toast.error('No active task creation to cancel');
      return;
    }

    try {
      console.log('[DEBUG] Cancelling task for session:', progress.sessionId);
      const response = await axios.post(`/api/tasks/cancel/${progress.sessionId}`);
      if (response.data.success) {
        toast.success('Task creation cancelled');
        setLoading(false);
        
        // Remove from active sessions
        removeTaskGeneration();
        
        // Note: The polling will detect the cancellation and stop automatically
        setProgress(prev => ({ 
          ...prev, 
          isTracking: false, 
          sessionId: null,
          debugMessages: [...prev.debugMessages, {
            timestamp: new Date().toLocaleTimeString(),
            type: 'info',
            message: 'Task creation cancelled by user'
          }]
        }));
      } else {
        toast.error(response.data.message || 'Failed to cancel task creation');
      }
    } catch (error) {
      console.error('Error cancelling task:', error);
      toast.error('Failed to cancel task creation');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);
    
    try {
      // Save API key to localStorage for future use
      localStorage.setItem('genhrl_api_key', formData.apiKey);
      
      console.log('[DEBUG] Sending task creation request:', formData);
      const response = await axios.post('/api/tasks', formData);
      console.log('[DEBUG] Task creation response:', response.data);
      
      if (response.data.success && response.data.sessionId) {
        console.log('[DEBUG] Task creation started, sessionId:', response.data.sessionId);
        // Start progress tracking (returns cleanup function)
        const cleanupProgressTracking = startProgressTracking(response.data.sessionId);
        
        // Clean up polling on component unmount or if user navigates away
        const cleanup = () => {
          cleanupProgressTracking();
          setProgress(prev => ({ ...prev, isTracking: false, sessionId: null }));
          setLoading(false);
        };
        
        // Store cleanup function for potential use
        window.addEventListener('beforeunload', cleanup);
        
        return () => {
          window.removeEventListener('beforeunload', cleanup);
          cleanup();
        };
      } else {
        throw new Error(response.data.error || 'Task creation failed');
      }
    } catch (error) {
      console.error('Error creating task:', error);
      const errorMessage = error.response?.data?.error || error.message || 'Failed to create task';
      const errorDetails = error.response?.data?.details;
      
      toast.error(errorMessage);
      
      if (errorDetails) {
        console.error('Error details:', errorDetails);
      }
      
      setLoading(false);
    }
  };

  const exampleDescriptions = [
    {
      name: "Box Stacking",
      description: "The robot should pick up three boxes of different sizes and stack them in order from largest to smallest, with the largest box on the bottom."
    },
    {
      name: "Ball Collection",
      description: "The robot should collect colored balls scattered around the environment and sort them by color into different containers."
    },
    {
      name: "Step Climbing",
      description: "The robot should navigate to a set of steps and climb them one by one, maintaining balance while reaching the top platform."
    }
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Create New Task</h1>
        <p className="text-gray-600 mt-2">
          Generate a hierarchical RL task from natural language description
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Form */}
        <div className="lg:col-span-2">
          <form onSubmit={handleSubmit} className="card space-y-6">
            {/* Task Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Task Name
              </label>
              <input
                type="text"
                name="taskName"
                value={formData.taskName}
                onChange={handleInputChange}
                className={`input-field ${errors.taskName ? 'border-red-500' : ''}`}
                placeholder="e.g., Pick_Up_Ball, Navigate_Maze"
                disabled={loading}
              />
              {errors.taskName && (
                <p className="text-red-500 text-sm mt-1 flex items-center">
                  <AlertCircle size={14} className="mr-1" />
                  {errors.taskName}
                </p>
              )}
              <p className="text-gray-500 text-sm mt-1">
                Use underscores instead of spaces. This will be used as the directory name.
              </p>
            </div>

            {/* Task Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Task Description
              </label>
              <textarea
                name="taskDescription"
                value={formData.taskDescription}
                onChange={handleInputChange}
                rows={6}
                className={`input-field ${errors.taskDescription ? 'border-red-500' : ''}`}
                placeholder="Describe the task in detail. Include objects, their properties, and the robot's objectives..."
                disabled={loading}
              />
              {errors.taskDescription && (
                <p className="text-red-500 text-sm mt-1 flex items-center">
                  <AlertCircle size={14} className="mr-1" />
                  {errors.taskDescription}
                </p>
              )}
              <p className="text-gray-500 text-sm mt-1">
                Provide a detailed description including objects, their properties, and the sequence of actions.
              </p>
            </div>

            {/* Robot Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Robot
              </label>
              <select
                name="robot"
                value={formData.robot}
                onChange={handleInputChange}
                className="input-field"
                disabled={loading}
              >
                {robots.map(robot => (
                  <option key={robot} value={robot}>{robot}</option>
                ))}
              </select>
            </div>

            {/* LLM Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Primary LLM Model
              </label>
              <select
                name="model"
                value={formData.model}
                onChange={handleInputChange}
                className="input-field"
                disabled={loading}
              >
                <option value="gemini-2.5-pro">Gemini 2.5 Pro (Google)</option>
                <option value="gemini-2.5-flash">Gemini 2.5 Flash (Google)</option>
                <option value="gemini-2.5-flash-lite-preview-06-17">Gemini Flash 2.5 Light (Google)</option>
                <option value="claude-4-sonnet-20240229">Claude 4 Sonnet (v20240229)</option>
                <option value="claude-3.5-sonnet-20240620">Claude 3.5 Sonnet (v20240620)</option>
              </select>
              <p className="text-gray-500 text-sm mt-1">
                Choose the language model used for task generation.
              </p>
            </div>

            {/* Backup LLM Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Backup LLM Model
              </label>
              <select
                name="backupModel"
                value={formData.backupModel}
                onChange={handleInputChange}
                className="input-field"
                disabled={loading}
              >
                <option value="gemini-2.5-pro">Gemini 2.5 Pro (Google)</option>
                <option value="gemini-2.5-flash">Gemini 2.5 Flash (Google)</option>
                <option value="gemini-2.5-flash-lite-preview-06-17">Gemini Flash 2.5 Light (Google)</option>
                <option value="claude-4-sonnet-20240229">Claude 4 Sonnet (v20240229)</option>
                <option value="claude-3.5-sonnet-20240620">Claude 3.5 Sonnet (v20240620)</option>
              </select>
              <p className="text-gray-500 text-sm mt-1">
                The backup model is used automatically if the primary model fails.
              </p>
            </div>

            {/* Hierarchy Levels */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hierarchy Levels
              </label>
              <select
                name="maxHierarchyLevels"
                value={formData.maxHierarchyLevels}
                onChange={handleInputChange}
                className="input-field"
                disabled={loading}
              >
                <option value={1}>Level 1 - Single task only</option>
                <option value={2}>Level 2 - Task → Skill decomposition</option>
                <option value={3}>Level 3 - Task → Skill → Sub-skill (full hierarchy)</option>
              </select>
              <p className="text-gray-500 text-sm mt-1">
                Higher levels create more complex skill hierarchies but take longer to generate.
              </p>
            </div>

            {/* API Key */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                API Key
              </label>
              <input
                type="password"
                name="apiKey"
                value={formData.apiKey}
                onChange={handleInputChange}
                className={`input-field ${errors.apiKey ? 'border-red-500' : ''}`}
                placeholder="Your Google Gemini or Anthropic Claude API key"
                disabled={loading}
              />
              {errors.apiKey && (
                <p className="text-red-500 text-sm mt-1 flex items-center">
                  <AlertCircle size={14} className="mr-1" />
                  {errors.apiKey}
                </p>
              )}
              <p className="text-gray-500 text-sm mt-1">
                Your API key will be saved locally for future use. Get one from Google AI Studio or Anthropic Console.
              </p>
            </div>

            {/* Progress Display */}
            {progress.isTracking && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-blue-900 flex items-center">
                    <Zap size={16} className="mr-2" />
                    Generating Task...
                    {progress.isConnected && (
                      <span className="ml-2 w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                    )}
                  </h4>
                  <span className="text-sm text-blue-700">
                    Step {progress.currentStage || 0} of {progress.totalStages || 7}
                  </span>
                </div>
                
                {/* Progress Bar */}
                <div className="w-full bg-blue-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${((progress.currentStage || 0) / (progress.totalStages || 7)) * 100}%` }}
                  ></div>
                </div>
                
                {/* Current Stage */}
                <div className="space-y-2">
                  <div className="flex items-center text-blue-800">
                    <Zap size={14} className="mr-2" />
                    <span className="font-medium">{progress.message}</span>
                  </div>
                  {progress.details && progress.stage !== 'debug' && (
                    <div className="ml-6">
                      {progress.details.split('\\n').map((line, index) => (
                        <div key={index} className="text-sm text-blue-700">
                          {line.startsWith('•') ? (
                            <div className="flex items-start">
                              <span className="text-blue-500 mr-2 mt-0.5">•</span>
                              <span>{line.substring(1).trim()}</span>
                            </div>
                          ) : (
                            <div>{line}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                
                {/* Stage Icons */}
                <div className="flex items-center justify-between text-xs">
                  {/* Generate dynamic step names based on current progress */}
                  {(() => {
                    const baseSteps = ['Init', 'Import', 'Setup', 'Objects', 'Decompose', 'Config'];
                    const totalSteps = progress.totalStages || 10;
                    const currentStep = progress.currentStage || 0;
                    
                    // Calculate number of skills (totalSteps = 6 base + 3*skills + 1 finalize)
                    const numSkills = Math.max(0, Math.floor((totalSteps - 7) / 3));
                    const finalizeStep = 6 + (numSkills * 3) + 1;
                    
                    // Generate all step names
                    let allSteps = [...baseSteps];
                    
                    // Add skill steps (3 per skill)
                    for (let i = 0; i < numSkills; i++) {
                      allSteps.push(`Skill${i + 1}R`); // Rewards
                      allSteps.push(`Skill${i + 1}S`); // Success
                      allSteps.push(`Skill${i + 1}C`); // Config
                    }
                    
                    // Add finalize step
                    if (totalSteps > 6) {
                      allSteps.push('Finalize');
                    }
                    
                    // Limit display to reasonable number of steps for UI
                    const maxDisplaySteps = 12;
                    let displaySteps = allSteps;
                    
                    if (allSteps.length > maxDisplaySteps) {
                      // Show first 6, current skill area, and last step
                      const skillStart = 6;
                      const currentSkill = Math.floor(Math.max(0, currentStep - 6) / 3);
                      const skillStepsStart = skillStart + (currentSkill * 3);
                      
                      displaySteps = [
                        ...baseSteps,
                        ...allSteps.slice(skillStepsStart, skillStepsStart + 3),
                        '...',
                        'Finalize'
                      ];
                    }
                    
                    return displaySteps.map((stage, index) => {
                      // For condensed view, map display index to actual step
                      let actualStep = index + 1;
                      if (stage === '...') {
                        return (
                          <div key={`ellipsis-${index}`} className="flex flex-col items-center space-y-1 text-gray-400">
                            <div className="w-6 h-6 flex items-center justify-center">
                              ...
                            </div>
                            <span>...</span>
                          </div>
                        );
                      } else if (stage === 'Finalize') {
                        actualStep = finalizeStep;
                      } else if (stage.startsWith('Skill') && allSteps.length > maxDisplaySteps) {
                        // Calculate actual step for skills in condensed view
                        const currentSkill = Math.floor(Math.max(0, currentStep - 6) / 3);
                        const skillStepsStart = 6 + (currentSkill * 3);
                        const skillSubIndex = index - 6; // Position within current skill group
                        actualStep = skillStepsStart + skillSubIndex + 1;
                      }
                      
                      return (
                        <div 
                          key={`${stage}-${index}`}
                          className={`flex flex-col items-center space-y-1 ${
                            actualStep <= currentStep ? 'text-blue-600' : 'text-gray-400'
                          }`}
                        >
                          <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                            actualStep <= currentStep ? 'border-blue-600 bg-blue-600 text-white' : 'border-gray-300'
                          } ${actualStep === currentStep ? 'animate-pulse' : ''}`}>
                            {actualStep < currentStep ? <CheckCircle size={12} /> : actualStep}
                          </div>
                          <span className="text-xs">{stage}</span>
                        </div>
                      );
                    });
                  })()}
                </div>
                
                {/* Terminal-like Debug Output */}
                {progress.debugMessages.length > 0 && (
                  <div className="mt-4">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="text-sm font-medium text-blue-900">Real-time Output</h5>
                      <div className="flex items-center space-x-2">
                        <button
                          type="button"
                          onClick={() => {
                            if (terminalRef.current) {
                              terminalRef.current.scrollTop = 0;
                            }
                          }}
                          className="text-xs text-blue-600 hover:text-blue-800 flex items-center space-x-1"
                          title="Scroll to top"
                        >
                          <span>↑</span>
                          <span>Top</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            if (terminalRef.current) {
                              terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
                            }
                          }}
                          className="text-xs text-blue-600 hover:text-blue-800 flex items-center space-x-1"
                          title="Scroll to bottom"
                        >
                          <span>↓</span>
                          <span>Bottom</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => setProgress(prev => ({ ...prev, debugMessages: [] }))}
                          className="text-xs text-blue-600 hover:text-blue-800"
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                    <div 
                      className="bg-gray-900 text-green-400 p-3 rounded-md font-mono text-xs max-h-48 overflow-y-auto custom-scrollbar dark-scrollbar" 
                      ref={terminalRef}
                      style={{ scrollBehavior: 'auto' }}
                    >
                      {progress.debugMessages.map((msg, index) => (
                        <div key={index} className={`mb-1 ${
                          msg.type === 'error' ? 'text-red-400' : 
                          msg.type === 'success' ? 'text-green-400' : 
                          msg.type === 'debug' ? 'text-yellow-400' :
                          msg.type === 'info' ? 'text-blue-400' : 'text-green-400'
                        }`}>
                          <span className="text-gray-500">[{msg.timestamp}]</span> {msg.message}
                        </div>
                      ))}
                      {progress.isTracking && (
                        <div className="text-gray-500 animate-pulse">
                          <span className="inline-block w-2 h-4 bg-green-400 ml-1"></span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Submit Button */}
            <div className="flex justify-end space-x-4">
              <button
                type="button"
                onClick={loading && progress.sessionId ? cancelTaskCreation : () => navigate('/tasks')}
                className={`btn-secondary flex items-center ${loading && progress.sessionId ? 'text-red-600 border-red-600 hover:bg-red-50' : ''}`}
                disabled={false}
              >
                {loading && progress.sessionId ? (
                  <>
                    <X size={16} className="mr-2" />
                    Cancel Task Creation
                  </>
                ) : (
                  'Cancel'
                )}
              </button>
              <button
                type="submit"
                className="btn-primary flex items-center"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Clock size={16} className="mr-2" />
                    {progress.isTracking ? progress.message || 'Processing...' : 'Starting...'}
                  </>
                ) : (
                  <>
                    <Play size={16} className="mr-2" />
                    Create Task
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Examples */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Example Tasks</h3>
            <div className="space-y-3">
              {exampleDescriptions.map((example, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-3">
                  <h4 className="font-medium text-gray-900 text-sm mb-1">{example.name}</h4>
                  <p className="text-xs text-gray-600 mb-2">{example.description}</p>
                  <button
                    type="button"
                    onClick={() => {
                      setFormData(prev => ({
                        ...prev,
                        taskName: example.name.replace(/ /g, '_'),
                        taskDescription: example.description
                      }));
                    }}
                    className="text-blue-600 hover:text-blue-700 text-xs"
                    disabled={loading}
                  >
                    Use this example
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Tips */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Tips</h3>
            <div className="space-y-3">
              <div className="flex items-start">
                <CheckCircle size={16} className="text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-600">
                  Be specific about object properties (size, weight, color)
                </p>
              </div>
              <div className="flex items-start">
                <CheckCircle size={16} className="text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-600">
                  Describe the sequence of actions clearly
                </p>
              </div>
              <div className="flex items-start">
                <CheckCircle size={16} className="text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-600">
                  Include spatial relationships between objects
                </p>
              </div>
              <div className="flex items-start">
                <CheckCircle size={16} className="text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-gray-600">
                  Mention success criteria explicitly
                </p>
              </div>
            </div>
          </div>

          {/* Status */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">GenHRL API</span>
                <span className="text-green-600 flex items-center">
                  <CheckCircle size={14} className="mr-1" />
                  Ready
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">IsaacLab</span>
                <span className="text-green-600 flex items-center">
                  <CheckCircle size={14} className="mr-1" />
                  Connected
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreateTask;