import React, { useState, useEffect } from 'react';
import { 
  X, 
  Play, 
  Settings, 
  Clock, 
  Target, 
  Users, 
  Shuffle, 
  RefreshCw, 
  SkipForward,
  Monitor,
  Video,
  HelpCircle
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const TrainingModal = ({ 
  isOpen, 
  onClose, 
  taskName, 
  robot, 
  onTrainingStart 
}) => {
  const [config, setConfig] = useState({
    maxTime: 180,
    minSuccessStates: 50,
    numEnvs: 4096,
    seed: 42,
    newRun: false,
    skipComplete: true,
    headless: true,
    video: false,
    videoInterval: 2000,
    videoLength: 200,
    minTrainingStepsPrimitive: 30000,
    minTrainingStepsComposite: 50000,
    enforceMinSteps: true,
    checkpointInterval: 1000
  });
  
  const [loading, setLoading] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  
  useEffect(() => {
    if (isOpen && taskName && robot) {
      fetchTrainingStatus();
    }
  }, [isOpen, taskName, robot]);
  
  const fetchTrainingStatus = async () => {
    try {
      const response = await axios.get(`/api/tasks/${taskName}/training/status?robot=${robot}`);
      setTrainingStatus(response.data);
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };
  
  const handleConfigChange = (field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  const handleStartTraining = async () => {
    try {
      setLoading(true);
      
      const response = await axios.post(
        `/api/tasks/${taskName}/training/start?robot=${robot}`,
        config
      );
      
      if (response.data.success) {
        toast.success('Training started successfully!');
        onTrainingStart(response.data.sessionId);
        onClose();
      } else {
        toast.error(response.data.error || 'Failed to start training');
      }
    } catch (error) {
      console.error('Error starting training:', error);
      toast.error(error.response?.data?.error || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };
  
  const getSkillStatusSummary = () => {
    if (!trainingStatus?.skillStatus) return null;
    
    const skills = Object.values(trainingStatus.skillStatus);
    const completed = skills.filter(s => s.status === 'completed').length;
    const pending = skills.filter(s => s.status === 'pending').length;
    
    return { total: skills.length, completed, pending };
  };
  
  const skillSummary = getSkillStatusSummary();
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 text-green-600 rounded-lg">
              <Play size={20} />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Training Configuration
              </h2>
              <p className="text-sm text-gray-600">
                Task: {taskName} • Robot: {robot}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>
        
        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Status Overview */}
          {skillSummary && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-medium text-blue-900 mb-2">Current Training Status</h3>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{skillSummary.total}</div>
                  <div className="text-blue-700">Total Skills</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{skillSummary.completed}</div>
                  <div className="text-green-700">Completed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">{skillSummary.pending}</div>
                  <div className="text-orange-700">Pending</div>
                </div>
              </div>
            </div>
          )}
          
          {/* Configuration Sections */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Training Parameters */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <Settings size={16} className="mr-2" />
                Training Parameters
              </h3>
              
              <div className="space-y-4">
                {/* Max Time */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Clock size={14} className="mr-1" />
                    Max Time per Skill (minutes)
                  </label>
                  <input
                    type="number"
                    value={config.maxTime}
                    onChange={(e) => handleConfigChange('maxTime', parseInt(e.target.value))}
                    className="input-field"
                    min="30"
                    max="1440"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Maximum training time per skill. Default: 180 minutes
                  </p>
                </div>
                
                {/* Min Success States */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Target size={14} className="mr-1" />
                    Min Success States
                  </label>
                  <input
                    type="number"
                    value={config.minSuccessStates}
                    onChange={(e) => handleConfigChange('minSuccessStates', parseInt(e.target.value))}
                    className="input-field"
                    min="10"
                    max="1000"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Minimum success states required to complete a skill. Default: 50
                  </p>
                </div>
                
                {/* Min Training Steps (Primitive) */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Shuffle size={14} className="mr-1" />
                    Min Steps (Primitive)
                  </label>
                  <input
                    type="number"
                    value={config.minTrainingStepsPrimitive}
                    onChange={(e) => handleConfigChange('minTrainingStepsPrimitive', parseInt(e.target.value))}
                    className="input-field"
                    min="0"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Early-exit primitive skills after this many steps (default 30000)
                  </p>
                </div>
                
                {/* Min Training Steps (Composite) */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Shuffle size={14} className="mr-1" />
                    Min Steps (Composite)
                  </label>
                  <input
                    type="number"
                    value={config.minTrainingStepsComposite}
                    onChange={(e) => handleConfigChange('minTrainingStepsComposite', parseInt(e.target.value))}
                    className="input-field"
                    min="0"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Early-exit composite skills after this many steps (default 50000)
                  </p>
                </div>
                
                {/* Enforce Min Steps */}
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.enforceMinSteps}
                    onChange={(e) => handleConfigChange('enforceMinSteps', e.target.checked)}
                    className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                  />
                  <label className="text-sm text-gray-700">
                    Enforce min steps before early-exit
                  </label>
                  <HelpCircle size={14} className="text-gray-400 ml-1" title="If unchecked, training will stop as soon as enough success-state files are produced, regardless of steps." />
                </div>
                
                {/* Num Environments */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Users size={14} className="mr-1" />
                    Parallel Environments
                  </label>
                  <select
                    value={config.numEnvs}
                    onChange={(e) => handleConfigChange('numEnvs', parseInt(e.target.value))}
                    className="input-field"
                  >
                    <option value={1024}>1,024 (Fast)</option>
                    <option value={2048}>2,048 (Balanced)</option>
                    <option value={4096}>4,096 (Recommended)</option>
                    <option value={8192}>8,192 (High Performance)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    More environments = faster training but higher GPU memory usage
                  </p>
                </div>
                
                {/* Seed */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Shuffle size={14} className="mr-1" />
                    Random Seed
                  </label>
                  <input
                    type="number"
                    value={config.seed}
                    onChange={(e) => handleConfigChange('seed', parseInt(e.target.value))}
                    className="input-field"
                    min="0"
                    max="999999"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Seed for reproducible training. Default: 42
                  </p>
                </div>

                {/* Checkpoint Interval */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center">
                    <Clock size={14} className="mr-1" />
                    Checkpoint Interval (steps)
                  </label>
                  <input
                    type="number"
                    value={config.checkpointInterval}
                    onChange={(e) => handleConfigChange('checkpointInterval', parseInt(e.target.value))}
                    className="input-field"
                    min="100"
                    max="100000"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Save agent checkpoints and check success states every N steps. Default: 1000
                  </p>
                </div>
              </div>
            </div>
            
            {/* Training Options */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <Monitor size={16} className="mr-2" />
                Training Options
              </h3>
              
              <div className="space-y-4">
                {/* New Run */}
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <RefreshCw size={16} className="mr-2 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">New Run</div>
                      <div className="text-sm text-gray-600">Clean all previous training data</div>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.newRun}
                      onChange={(e) => handleConfigChange('newRun', e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                {/* Skip Complete */}
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <SkipForward size={16} className="mr-2 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Skip Complete</div>
                      <div className="text-sm text-gray-600">Skip skills with sufficient success states</div>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.skipComplete}
                      onChange={(e) => handleConfigChange('skipComplete', e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                {/* Headless */}
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <Monitor size={16} className="mr-2 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Headless Mode</div>
                      <div className="text-sm text-gray-600">Run without visual interface (faster)</div>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.headless}
                      onChange={(e) => handleConfigChange('headless', e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                {/* Video Recording */}
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <Video size={16} className="mr-2 text-gray-600" />
                    <div>
                      <div className="font-medium text-gray-900">Record Videos</div>
                      <div className="text-sm text-gray-600">Record training videos (slower)</div>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.video}
                      onChange={(e) => handleConfigChange('video', e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                {/* Video Settings */}
                {config.video && (
                  <div className="ml-6 space-y-3 pl-4 border-l-2 border-gray-200">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Video Interval (steps)
                      </label>
                      <input
                        type="number"
                        value={config.videoInterval}
                        onChange={(e) => handleConfigChange('videoInterval', parseInt(e.target.value))}
                        className="input-field"
                        min="100"
                        max="10000"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Video Length (steps)
                      </label>
                      <input
                        type="number"
                        value={config.videoLength}
                        onChange={(e) => handleConfigChange('videoLength', parseInt(e.target.value))}
                        className="input-field"
                        min="50"
                        max="1000"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Help Text */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-start">
              <HelpCircle size={16} className="text-yellow-600 mr-2 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-yellow-800">
                <p className="font-medium mb-1">Training Process:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>Skills are trained sequentially in dependency order</li>
                  <li>Success states are transferred between skills for curriculum learning</li>
                  <li>Training automatically stops when success criteria are met</li>
                  <li>You can monitor progress in real-time and cancel if needed</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 bg-gray-50">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-600">
              💡 Estimated training time: {Math.round(config.maxTime * (skillSummary?.pending || 1) / 60)} hours
            </div>
            <div className="flex space-x-3">
              <button
                onClick={onClose}
                className="btn-secondary"
                disabled={loading}
              >
                Cancel
              </button>
              <button
                onClick={handleStartTraining}
                className="btn-primary flex items-center"
                disabled={loading || trainingStatus?.isTraining}
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Starting...
                  </>
                ) : (
                  <>
                    <Play size={16} className="mr-2" />
                    Start Training
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingModal;