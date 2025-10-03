import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  X, 
  Play, 
  Pause, 
  Square, 
  Activity, 
  Clock, 
  Target, 
  CheckCircle, 
  AlertTriangle,
  Zap,
  Users,
  TrendingUp,
  Terminal,
  Copy,
  Download,
  Maximize2,
  Minimize2,
  ChevronDown,
  Trash2,
  FileVideo,
  RefreshCw,
  ExternalLink,
  BarChart3,
  Calendar
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const TrainingProgressModal = ({ 
  isOpen, 
  onClose, 
  taskName, 
  robot, 
  sessionId,
  onTrainingComplete 
}) => {
  const [progress, setProgress] = useState({
    stage: 'initializing',
    message: 'Initializing training...',
    details: {},
    timestamp: null
  });
  const [isComplete, setIsComplete] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [error, setError] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [currentSkillTime, setCurrentSkillTime] = useState(0);
  const [terminalOutput, setTerminalOutput] = useState([]);
  const [activeTab, setActiveTab] = useState('progress');
  const [isTerminalExpanded, setIsTerminalExpanded] = useState(false);
  // Auto-scroll removed - users have manual control
  const [showScrollIndicator, setShowScrollIndicator] = useState(false);
  const [currentSkillInfo, setCurrentSkillInfo] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState({});
  const [refreshing, setRefreshing] = useState(false);
  const [videos, setVideos] = useState([]);
  const [videosLoading, setVideosLoading] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [selectedSkill, setSelectedSkill] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('Reward/mean');
  const eventSourceRef = useRef(null);
  const startTimeRef = useRef(null);
  const skillStartTimeRef = useRef(null);
  const terminalRef = useRef(null);
  const videosIntervalRef = useRef(null);
  const metricsIntervalRef = useRef(null);
  
  useEffect(() => {
    if (isOpen && sessionId) {
      startTimeRef.current = Date.now();
      connectToProgressStream();
      
      // Start elapsed time counter
      const timer = setInterval(() => {
        if (startTimeRef.current) {
          setElapsedTime(Math.floor((Date.now() - startTimeRef.current) / 1000));
        }
        if (skillStartTimeRef.current) {
          setCurrentSkillTime(Math.floor((Date.now() - skillStartTimeRef.current) / 1000));
        }
      }, 1000);
      
      // Start polling for videos every 30 s while training is in progress
      if (!videosIntervalRef.current) {
        videosIntervalRef.current = setInterval(() => {
          fetchVideos();
        }, 30000);
      }
      
      // Poll metrics every 30 s
      if (!metricsIntervalRef.current) {
        metricsIntervalRef.current = setInterval(() => {
          fetchMetrics();
        }, 30000);
      }
      
      return () => {
        disconnectFromProgressStream();
        clearInterval(timer);
        if (videosIntervalRef.current) {
          clearInterval(videosIntervalRef.current);
          videosIntervalRef.current = null;
        }
        if (metricsIntervalRef.current) {
          clearInterval(metricsIntervalRef.current);
          metricsIntervalRef.current = null;
        }
      };
    }
  }, [isOpen, sessionId]);

  // No auto-scroll - users have manual control over terminal scrolling

  // Handle terminal scroll to detect when user scrolls up
  const handleTerminalScroll = (e) => {
    const terminal = e.target;
    const isNearBottom = terminal.scrollTop + terminal.clientHeight >= terminal.scrollHeight - 10;
    
    // Only show scroll indicator if user is not at bottom and there are new messages
    if (isNearBottom) {
      setShowScrollIndicator(false);
    } else {
      setShowScrollIndicator(true);
    }
    
    // Never auto-enable auto-scroll - let user control it manually
  };

  // Scroll to bottom manually
  const scrollToBottom = () => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
      setShowScrollIndicator(false);
    }
  };

  // Scroll to top manually
  const scrollToTop = () => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = 0;
    }
  };
  
  const connectToProgressStream = () => {
    if (!sessionId) {
      console.error('[TrainingProgress] No session ID provided');
      return;
    }
    
    // Build full SSE URL. In dev mode the React app runs on port 3000 while 
    // the Flask backend (SSE endpoint) is on 5000, and the CRA proxy does NOT
    // forward EventSource requests.  Use absolute URL when we detect that case.
    let apiBase = '';
    if (window.location.port === '3000') {
      apiBase = 'http://localhost:5000';
    }
    const streamUrl = `${apiBase}/api/tasks/${taskName}/training/progress/${sessionId}?robot=${robot}`;
    console.log(`[TrainingProgress] Connecting to training progress stream: ${streamUrl}`);
    
    const eventSource = new EventSource(streamUrl);
    eventSourceRef.current = eventSource;
    
    eventSource.onopen = () => {
      console.log('[TrainingProgress] SSE connection opened');
    };
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('[TrainingProgress] Received progress data:', data);
        
        if (data.type === 'progress') {
          console.log('[TrainingProgress] Setting progress with details:', data.details);
          setProgress({
            stage: data.stage,
            message: data.message,
            details: data.details || {},
            timestamp: data.timestamp
          });
          
          // Update skill information if available
          if (data.details && data.details.currentSkill) {
            console.log('[TrainingProgress] Updating skill info with step data:', {
              currentStep: data.details.currentStep,
              totalSteps: data.details.totalSteps
            });
            setCurrentSkillInfo({
              name: data.details.currentSkill,
              type: data.details.skillType || 'unknown',
              currentStep: data.details.currentStep || 0,
              totalSteps: data.details.totalSteps || 0,
              progressPercent: data.details.progressPercent || 0,
              estimatedTimeRemaining: data.details.estimatedTimeRemaining || null,
              skillIndex: data.details.skillIndex || 0,
              totalSkills: data.details.totalSkills || 0
            });
            
            // Update training metrics
            if (data.details.currentReward || data.details.episodeLength || data.details.successRate) {
              setTrainingMetrics({
                reward: data.details.currentReward || null,
                episodeLength: data.details.episodeLength || null,
                successRate: data.details.successRate || null
              });
            }
          }
          
          // Track skill start time for individual skill timing
          if (data.details?.currentSkill && data.details.currentSkill !== progress.details?.currentSkill) {
            skillStartTimeRef.current = Date.now();
            setCurrentSkillTime(0);
          }
        } else if (data.type === 'terminal') {
          // Handle terminal output
          console.log('[TrainingProgress] Received terminal output:', data.line);
          const timestamp = new Date(data.timestamp).toLocaleTimeString();
          setTerminalOutput(prev => {
            const newOutput = [...prev, {
              timestamp,
              line: data.line,
              type: getLineType(data.line),
              isDebug: isDebugOutput(data.line)
            }].slice(-1000); // Keep last 1000 lines
            console.log('[TrainingProgress] Terminal output updated, total lines:', newOutput.length);
            return newOutput;
          });
          
          // Auto-switch to terminal tab if there's an error
          if (data.line.includes('ERROR') || data.line.includes('FAILED') || data.line.includes('Exception')) {
            console.log('[TrainingProgress] Error detected, switching to terminal tab');
            setActiveTab('terminal');
          }
        } else if (data.type === 'complete') {
          setIsComplete(true);
          setIsSuccess(data.success);
          if (data.error) {
            setError(data.error);
          }
          
          if (data.success) {
            toast.success('Training completed successfully!');
          } else {
            toast.error('Training failed');
            // Auto-switch to terminal tab to show errors
            setActiveTab('terminal');
          }
          
          if (onTrainingComplete) {
            onTrainingComplete(data.success, data.message);
          }
        } else if (data.type === 'connected') {
          console.log('[TrainingProgress] Connected to training progress stream');
        } else if (data.type === 'heartbeat') {
          // Heartbeat - keep connection alive
        }
      } catch (err) {
        console.error('[TrainingProgress] Error parsing progress data:', err);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('[TrainingProgress] SSE error:', error);
      // Connection will auto-reconnect
    };
  };

  const getLineType = (line) => {
    if (line.includes('ERROR') || line.includes('Exception') || line.includes('Traceback')) {
      return 'error';
    } else if (line.includes('FAILED') || line.includes('failed')) {
      return 'warning';
    } else if (line.includes('SUCCESS') || line.includes('completed successfully') || line.includes('✅')) {
      return 'success';
    } else if (line.includes('===') || line.includes('Processing') || line.includes('Starting')) {
      return 'info';
    } else {
      return 'normal';
    }
  };

  // Helper function to determine if output is debug info
  const isDebugOutput = (line) => {
    const debugPatterns = [
      /^OUTLIER CLIPPING/,
      /^Reward \w+: mean=/,
      /GPU memory usage/,
      /Loading checkpoint/,
      /^\[INFO\]/,
      /^\[DEBUG\]/,
      /^\[Warning\]/,
      /^H\s+H\s+H\s+H/,  // Isaac Sim initialization pattern
      /simulation_app\.simulation_app/,
      /Kit\/Isaac-Sim/,
      /carb\.windowing/
    ];
    
    return debugPatterns.some(pattern => pattern.test(line));
  };

  // Filter terminal output by type
  const filteredTerminalOutput = terminalOutput.filter(entry => {
    if (activeTab === 'terminal') {
      return !entry.isDebug; // Show main training output
    } else if (activeTab === 'debug') {
      return entry.isDebug; // Show debug output
    }
    return true;
  });
  
  const disconnectFromProgressStream = () => {
    if (eventSourceRef.current) {
      console.log('[TrainingProgress] Disconnecting from training progress stream');
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };
  
  const handleCancelTraining = async () => {
    if (!window.confirm('Are you sure you want to cancel training? This will stop the current training process.')) {
      return;
    }
    
    try {
      await axios.post(`/api/tasks/${taskName}/training/cancel?robot=${robot}`);
      toast.success('Training cancelled');
      onClose();
    } catch (error) {
      console.error('Error cancelling training:', error);
      toast.error('Failed to cancel training');
    }
  };

  const handleCleanup = async () => {
    if (!window.confirm('Clean up Isaac Sim processes? This will force-close any remaining Isaac Sim instances to free up GPU memory.')) {
      return;
    }
    
    try {
      toast.info('Cleaning up Isaac Sim processes...');
      await axios.post('/api/cleanup');
      toast.success('Cleanup completed successfully');
    } catch (error) {
      console.error('Error during cleanup:', error);
      toast.error('Failed to perform cleanup');
    }
  };

  const handleCopyTerminal = async () => {
    const terminalText = terminalOutput.map(entry => `[${entry.timestamp}] ${entry.line}`).join('\n');
    try {
      await navigator.clipboard.writeText(terminalText);
      toast.success('Terminal output copied to clipboard');
    } catch (error) {
      toast.error('Failed to copy terminal output');
    }
  };

  const handleDownloadTerminal = () => {
    const terminalText = terminalOutput.map(entry => `[${entry.timestamp}] ${entry.line}`).join('\n');
    const blob = new Blob([terminalText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_log_${taskName}_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    toast.success('Terminal log downloaded');
  };
  
  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };
  
  const getStageIcon = (stage) => {
    switch (stage) {
      case 'initializing':
        return <Play size={16} className="text-blue-600" />;
      case 'planning':
        return <Target size={16} className="text-purple-600" />;
      case 'training':
        return <Activity size={16} className="text-green-600" />;
      case 'complete':
        return <CheckCircle size={16} className="text-green-600" />;
      case 'error':
        return <AlertTriangle size={16} className="text-red-600" />;
      default:
        return <Play size={16} className="text-gray-600" />;
    }
  };
  
  const getProgressPercentage = () => {
    const details = progress.details;
    if (details.totalSkills) {
      const idx = details.currentSkillIndex !== undefined ? details.currentSkillIndex : details.skillIndex;
      if (idx !== undefined) {
        return Math.round((idx / details.totalSkills) * 100);
      }
    }
    return 0;
  };

  const fetchVideos = async () => {
    try {
      setVideosLoading(true);
      
      // Only fetch videos for the current skill when training is active
      let url = `/api/tasks/${taskName}/videos?robot=${robot}`;
      if (progress.stage === 'training' && currentSkillInfo?.name) {
        url += `&currentSkill=${encodeURIComponent(currentSkillInfo.name)}`;
      }
      
      const res = await axios.get(url);
      const vidList = res.data.videos || [];
      setVideos(vidList);
      if (vidList.length > 0 && !selectedVideo) {
        setSelectedVideo(vidList[0]);
      }
    } catch (err) {
      console.error('[TrainingProgress] Failed to fetch videos', err);
    } finally {
      setVideosLoading(false);
    }
  };

  // Initial fetch when tab becomes visible
  useEffect(() => {
    if (activeTab === 'videos' && videos.length === 0) {
      fetchVideos();
    }
  }, [activeTab]);

  // Refresh videos when current skill changes during training
  useEffect(() => {
    if (activeTab === 'videos' && progress.stage === 'training' && currentSkillInfo?.name) {
      fetchVideos();
    }
  }, [currentSkillInfo?.name, activeTab, progress.stage]);

  const fetchMetrics = async () => {
    try {
      const res = await axios.get(`/api/tasks/${taskName}/metrics?robot=${robot}`);
      const data = res.data.metrics || {};
      setMetrics(data);
      const skillNames = Object.keys(data);
      if (skillNames.length > 0 && !selectedSkill) {
        setSelectedSkill(skillNames[0]);
      }
    } catch (err) {
      console.error('[TrainingProgress] Failed to fetch metrics', err);
    }
  };

  const getChartData = () => {
    if (!selectedSkill || !metrics[selectedSkill]) return [];
    const scalarData = metrics[selectedSkill].tensorboard_data?.scalars || {};
    const metricPoints = scalarData[selectedMetric] || [];
    return metricPoints.map(p => ({ step: p.step, value: p.value }));
  };

  const availableMetricTags = () => {
    if (!selectedSkill || !metrics[selectedSkill]) return [];
    return metrics[selectedSkill].tensorboard_data?.tags || [];
  };

  const tabs = [
    { id: 'progress', label: 'Progress', icon: <TrendingUp size={16} /> },
    { 
      id: 'terminal', 
      label: 'Training Output', 
      icon: <Terminal size={16} />,
      badge: terminalOutput.filter(entry => entry.type === 'error' && !entry.isDebug).length > 0 ? 'error' : null
    },
    { 
      id: 'debug', 
      label: 'Debug Output', 
      icon: <AlertTriangle size={16} />,
      badge: terminalOutput.filter(entry => entry.isDebug).length > 0 ? 'debug' : null
    },
    {
      id: 'videos',
      label: 'Videos',
      icon: <FileVideo size={16} />
    },
    {
      id: 'metrics',
      label: 'Metrics',
      icon: <BarChart3 size={16} />
    }
  ];
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className={`bg-white rounded-lg shadow-xl w-full transition-all duration-300 ${
        isTerminalExpanded ? 'max-w-7xl max-h-[95vh]' : 'max-w-4xl max-h-[90vh]'
      } overflow-hidden flex flex-col`}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 text-green-600 rounded-lg">
              <Activity size={20} />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Training: {taskName}
              </h2>
              <p className="text-sm text-gray-600">
                Robot: {robot} • Session: {sessionId?.slice(0, 8)}...
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="text-sm text-gray-600">
              Total Time: {formatTime(elapsedTime)}
            </div>
            
            {!isComplete && (
              <button
                onClick={handleCancelTraining}
                className="px-3 py-1 text-sm bg-red-100 text-red-600 rounded-md hover:bg-red-200 transition-colors"
              >
                Cancel Training
              </button>
            )}
            
            <button
              onClick={handleCleanup}
              className="px-3 py-1 text-sm bg-orange-100 text-orange-600 rounded-md hover:bg-orange-200 transition-colors flex items-center space-x-1"
              title="Clean up Isaac Sim processes to free GPU memory"
            >
              <Trash2 size={14} />
              <span>Cleanup</span>
            </button>
            
            <button
              onClick={onClose}
              className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 px-6">
          <nav className="flex space-x-8">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
                {tab.badge === 'error' && (
                  <span className="bg-red-100 text-red-600 text-xs px-2 py-1 rounded-full">
                    Errors
                  </span>
                )}
                {tab.badge === 'debug' && (
                  <span className="bg-blue-100 text-blue-600 text-xs px-2 py-1 rounded-full">
                    {terminalOutput.filter(entry => entry.isDebug).length}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'progress' && (
            <div className="p-6 h-full overflow-y-auto">
              {/* Status Overview */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <div className="p-2 bg-blue-100 rounded-lg mr-3">
                      {getStageIcon(progress.stage)}
                    </div>
                    <div>
                      <p className="text-sm text-blue-600">Current Stage</p>
                      <p className="font-semibold text-blue-900 capitalize">{progress.stage}</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <div className="p-2 bg-green-100 rounded-lg mr-3">
                      <Clock size={16} className="text-green-600" />
                    </div>
                    <div>
                      <p className="text-sm text-green-600">Elapsed Time</p>
                      <p className="font-semibold text-green-900">{formatTime(elapsedTime)}</p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="flex items-center">
                    <div className="p-2 bg-purple-100 rounded-lg mr-3">
                      <Target size={16} className="text-purple-600" />
                    </div>
                    <div>
                      <p className="text-sm text-purple-600">Step</p>
                      <p className="font-semibold text-purple-900">
                        {progress.details?.currentStep ?? 0} / {progress.details?.totalSteps ?? '--'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Current Status */}
              <div className="bg-gray-50 rounded-lg p-4 mb-6">
                <h3 className="font-medium text-gray-900 mb-2">Current Status</h3>
                <div className="flex items-start space-x-3">
                  <div className="mt-1">
                    {getStageIcon(progress.stage)}
                  </div>
                  <div className="flex-1">
                    <p className="text-gray-900 font-medium">{progress.message}</p>
                    {progress.details?.currentSkill && (
                      <p className="text-sm text-gray-600 mt-1">
                        Current Skill: {progress.details.currentSkill}
                      </p>
                    )}
                    {progress.timestamp && (
                      <p className="text-xs text-gray-500 mt-1">
                        Last Update: {new Date(progress.timestamp).toLocaleTimeString()}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Completion Status */}
              {isComplete && (
                <div className={`rounded-lg p-4 ${isSuccess ? 'bg-green-50' : 'bg-red-50'}`}>
                  <div className="flex items-center">
                    <div className="mr-3">
                      {isSuccess ? 
                        <CheckCircle size={20} className="text-green-600" /> : 
                        <AlertTriangle size={20} className="text-red-600" />
                      }
                    </div>
                    <div>
                      <h3 className={`font-semibold ${isSuccess ? 'text-green-900' : 'text-red-900'}`}>
                        {isSuccess ? 'Training Completed Successfully!' : 'Training Failed'}
                      </h3>
                      {error && (
                        <p className="text-red-700 mt-1 text-sm">{error}</p>
                      )}
                      {!isSuccess && (
                        <p className="text-red-700 mt-1 text-sm">
                          Check the Terminal Output tab for detailed error information.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'terminal' && (
            <div className="h-full flex flex-col">
              {/* Terminal Header */}
              <div className="flex items-center justify-between p-3 bg-gray-800 text-white">
                <div className="flex items-center space-x-2">
                  <Terminal size={16} />
                  <span className="text-sm font-medium">Training Output</span>
                  <span className="text-xs text-gray-400">
                    ({filteredTerminalOutput.length} lines)
                  </span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleCopyTerminal}
                    className="p-1 hover:bg-gray-700 rounded text-xs flex items-center space-x-1"
                    title="Copy terminal output"
                  >
                    <Copy size={12} />
                    <span>Copy</span>
                  </button>
                  <button
                    onClick={handleDownloadTerminal}
                    className="p-1 hover:bg-gray-700 rounded text-xs flex items-center space-x-1"
                    title="Download as file"
                  >
                    <Download size={12} />
                    <span>Download</span>
                  </button>
                  <button
                    onClick={scrollToTop}
                    className="p-1 hover:bg-gray-700 rounded text-xs flex items-center space-x-1"
                    title="Scroll to top"
                  >
                    <span>↑</span>
                    <span>Top</span>
                  </button>
                  <button
                    onClick={scrollToBottom}
                    className="p-1 hover:bg-gray-700 rounded text-xs flex items-center space-x-1"
                    title="Scroll to bottom"
                  >
                    <span>↓</span>
                    <span>Bottom</span>
                  </button>
                  <button
                    onClick={() => setIsTerminalExpanded(!isTerminalExpanded)}
                    className="p-1 hover:bg-gray-700 rounded text-xs flex items-center space-x-1"
                    title={isTerminalExpanded ? "Collapse" : "Expand"}
                  >
                    {isTerminalExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                    <span>{isTerminalExpanded ? "Collapse" : "Expand"}</span>
                  </button>
                </div>
              </div>
              
              {/* Terminal Content */}
              <div className="terminal-container flex-1 relative" style={{ minHeight: isTerminalExpanded ? '500px' : '300px' }}>
                <div 
                  ref={terminalRef}
                  className="terminal-content bg-gray-900 text-green-400 p-3 font-mono text-xs h-full overflow-y-auto custom-scrollbar dark-scrollbar"
                  onScroll={handleTerminalScroll}
                  style={{ 
                    scrollBehavior: 'auto',
                    maxHeight: '100%'
                  }}
                >
                  {filteredTerminalOutput.length === 0 ? (
                    <div className="text-gray-500 text-center py-8">
                      <Terminal size={24} className="mx-auto mb-2 opacity-50" />
                      <p>Waiting for training output...</p>
                      <p className="text-xs mt-2 text-gray-400">
                        Use scroll controls or mouse wheel to navigate
                      </p>
                    </div>
                  ) : (
                    <>
                      {filteredTerminalOutput.map((entry, index) => (
                        <div key={index} className={`mb-1 break-words ${
                          entry.type === 'error' ? 'text-red-400' : 
                          entry.type === 'warning' ? 'text-yellow-400' : 
                          entry.type === 'success' ? 'text-green-400' : 
                          entry.type === 'info' ? 'text-blue-400' : 'text-green-400'
                        }`}>
                          <span className="text-gray-500 select-none">[{entry.timestamp}]</span> 
                          <span className="ml-1">{entry.line}</span>
                        </div>
                      ))}
                      
                      {!isComplete && (
                        <div className="text-gray-500 animate-pulse flex items-center">
                          <span className="inline-block w-2 h-4 bg-green-400 ml-1 animate-pulse"></span>
                          <span className="ml-2 text-xs">Live output...</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
                
                {/* Scroll to bottom indicator */}
                {showScrollIndicator && (
                  <button
                    onClick={scrollToBottom}
                    className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded-full text-xs flex items-center space-x-1 shadow-lg transition-all duration-200 hover:scale-105"
                    title="Scroll to bottom"
                  >
                    <span>↓</span>
                    <span>Go to bottom</span>
                  </button>
                )}
              </div>
            </div>
          )}

          {activeTab === 'debug' && (
            <div className="h-full flex flex-col">
              {/* Debug Header */}
              <div className="flex items-center justify-between p-3 bg-blue-800 text-white">
                <div className="flex items-center space-x-2">
                  <AlertTriangle size={16} />
                  <span className="text-sm font-medium">Debug Information</span>
                  <span className="text-xs text-blue-200">
                    ({filteredTerminalOutput.length} lines)
                  </span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleCopyTerminal}
                    className="p-1 hover:bg-blue-700 rounded text-xs flex items-center space-x-1"
                    title="Copy debug output"
                  >
                    <Copy size={12} />
                    <span>Copy</span>
                  </button>
                  <button
                    onClick={scrollToTop}
                    className="p-1 hover:bg-blue-700 rounded text-xs flex items-center space-x-1"
                    title="Scroll to top"
                  >
                    <span>↑</span>
                    <span>Top</span>
                  </button>
                  <button
                    onClick={scrollToBottom}
                    className="p-1 hover:bg-blue-700 rounded text-xs flex items-center space-x-1"
                    title="Scroll to bottom"
                  >
                    <span>↓</span>
                    <span>Bottom</span>
                  </button>
                  <button
                    onClick={() => setIsTerminalExpanded(!isTerminalExpanded)}
                    className="p-1 hover:bg-blue-700 rounded text-xs flex items-center space-x-1"
                    title={isTerminalExpanded ? "Collapse" : "Expand"}
                  >
                    {isTerminalExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                    <span>{isTerminalExpanded ? "Collapse" : "Expand"}</span>
                  </button>
                </div>
              </div>
              
              {/* Debug Content */}
              <div className="terminal-container flex-1 relative" style={{ minHeight: isTerminalExpanded ? '500px' : '300px' }}>
                <div 
                  ref={terminalRef}
                  className="terminal-content bg-gray-900 text-blue-300 p-3 font-mono text-xs h-full overflow-y-auto custom-scrollbar dark-scrollbar"
                  onScroll={handleTerminalScroll}
                  style={{ 
                    scrollBehavior: 'auto',
                    maxHeight: '100%'
                  }}
                >
                  {filteredTerminalOutput.length === 0 ? (
                    <div className="text-gray-500 text-center py-8">
                      <AlertTriangle size={24} className="mx-auto mb-2 opacity-50" />
                      <p>No debug information available...</p>
                      <p className="text-xs mt-2 text-gray-400">
                        Debug output includes reward information, system messages, and diagnostic data
                      </p>
                    </div>
                  ) : (
                    <>
                      {filteredTerminalOutput.map((entry, index) => (
                        <div key={index} className={`mb-1 break-words ${
                          entry.line.includes('OUTLIER CLIPPING') ? 'text-yellow-300' : 
                          entry.line.includes('Reward') ? 'text-green-300' : 
                          entry.line.includes('GPU') ? 'text-purple-300' : 
                          entry.line.includes('WARNING') || entry.line.includes('Warning') ? 'text-orange-300' : 
                          'text-blue-300'
                        }`}>
                          <span className="text-gray-500 select-none">[{entry.timestamp}]</span> 
                          <span className="ml-1">{entry.line}</span>
                        </div>
                      ))}
                      
                      {!isComplete && (
                        <div className="text-gray-500 animate-pulse flex items-center">
                          <span className="inline-block w-2 h-4 bg-blue-400 ml-1 animate-pulse"></span>
                          <span className="ml-2 text-xs">Live debug output...</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
                
                {/* Scroll to bottom indicator */}
                {showScrollIndicator && (
                  <button
                    onClick={scrollToBottom}
                    className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded-full text-xs flex items-center space-x-1 shadow-lg transition-all duration-200 hover:scale-105"
                    title="Scroll to bottom"
                  >
                    <span>↓</span>
                    <span>Go to bottom</span>
                  </button>
                )}
              </div>
            </div>
          )}

          {activeTab === 'videos' && (
            <div className="flex h-full">
              {/* Video player */}
              <div className="flex-1 bg-black flex items-center justify-center">
                {selectedVideo ? (
                  <video
                    key={selectedVideo.relative_path}
                    src={selectedVideo.relative_path + `?robot=${robot}`}
                    controls
                    className="max-w-full max-h-full"
                  />
                ) : (
                  <p className="text-white">No video selected</p>
                )}
              </div>

              {/* Video list */}
              <div className="w-72 border-l border-gray-200 overflow-y-auto">
                <div className="p-4 border-b border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900">Videos ({videos.length})</span>
                    <button onClick={fetchVideos} title="Refresh" className="p-1 text-gray-600 hover:text-blue-600">
                      <RefreshCw size={14} className={videosLoading ? 'animate-spin' : ''} />
                    </button>
                  </div>
                  {progress.stage === 'training' && currentSkillInfo?.name && (
                    <div className="text-xs text-blue-600">
                      Current: {currentSkillInfo.name}
                    </div>
                  )}
                  {progress.stage !== 'training' && (
                    <div className="text-xs text-gray-500">
                      All skills
                    </div>
                  )}
                </div>
                {videosLoading && videos.length === 0 ? (
                  <div className="p-4 text-center text-gray-500">Loading...</div>
                ) : videos.length === 0 ? (
                  <div className="p-4 text-center text-gray-500">No videos yet</div>
                ) : (
                  videos.map((vid) => (
                    <div
                      key={vid.filename}
                      onClick={() => setSelectedVideo(vid)}
                      className={`p-3 border-b border-gray-200 cursor-pointer hover:bg-gray-100 ${
                        selectedVideo && selectedVideo.filename === vid.filename ? 'bg-blue-50' : ''
                      }`}
                    >
                      <p className="text-sm font-medium text-gray-900 truncate">{vid.skill_name}</p>
                      <p className="text-xs text-gray-500 truncate">{vid.filename}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {activeTab === 'metrics' && (
            <div className="flex h-full">
              {/* Skills sidebar */}
              <div className="w-64 border-r border-gray-200 overflow-y-auto">
                <div className="p-4 border-b border-gray-200 font-medium">Skills</div>
                {Object.keys(metrics).length === 0 ? (
                  <div className="p-4 text-gray-500">No metrics yet</div>
                ) : (
                  Object.keys(metrics).map(name => (
                    <div
                      key={name}
                      onClick={() => setSelectedSkill(name)}
                      className={`p-3 cursor-pointer border-b ${selectedSkill === name ? 'bg-blue-50' : ''}`}
                    >
                      {name}
                    </div>
                  ))
                )}
              </div>

              {/* Chart area */}
              <div className="flex-1 p-6">
                {(!selectedSkill || !metrics[selectedSkill]) ? (
                  <div className="h-full flex items-center justify-center text-gray-500">Select a skill</div>
                ) : (
                  <>
                    <div className="flex items-center space-x-4 mb-4">
                      <span className="text-sm">Metric:</span>
                      <select value={selectedMetric} onChange={e=>setSelectedMetric(e.target.value)} className="input-field w-64">
                        {availableMetricTags().map(tag=> (
                          <option key={tag} value={tag}>{tag}</option>
                        ))}
                      </select>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={getChartData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="step" label={{ value: 'Step', position:'insideBottom', offset:-5 }}/>
                        <YAxis label={{ value: selectedMetric, angle:-90, position:'insideLeft'}} />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="value" stroke="#3b82f6" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 bg-gray-50">
          <div className="flex justify-between items-center text-sm text-gray-600">
            <div className="flex items-center space-x-4">
              <span>📊 Training metrics powered by TensorBoard</span>
              {activeTab === 'videos' && progress.stage === 'training' && currentSkillInfo?.name && (
                <span>🎬 Showing videos for current skill: {currentSkillInfo.name}</span>
              )}
              {activeTab === 'videos' && progress.stage !== 'training' && (
                <span>🎬 Showing videos from all skills</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingProgressModal;