import React, { useState, useEffect, useCallback } from 'react';
import { X, Code, Target, Settings, Download, RefreshCw } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import CodeEditor from './CodeEditor';

const SkillCodeModal = ({ 
  isOpen, 
  onClose, 
  taskName, 
  skillName, 
  robot, 
  initialSkillData = null 
}) => {
  const [skillData, setSkillData] = useState(initialSkillData);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('rewards');
  const [refreshing, setRefreshing] = useState(false);

  const fetchSkillData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await axios.get(
        `/api/tasks/${taskName}/skills/${skillName}?robot=${robot}`
      );
      setSkillData(response.data);
    } catch (error) {
      console.error('Error fetching skill data:', error);
      toast.error('Failed to fetch skill code');
    } finally {
      setLoading(false);
    }
  }, [taskName, skillName, robot]);

  // Fetch complete skill data when modal opens
  useEffect(() => {
    if (isOpen && taskName && skillName) {
      fetchSkillData();
    }
  }, [isOpen, taskName, skillName, fetchSkillData]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchSkillData();
    setRefreshing(false);
    toast.success('Skill code refreshed');
  };

  const handleSaveCode = async (codeType, newCode) => {
    try {
      const updateData = { [codeType]: newCode };
      
      await axios.put(
        `/api/tasks/${taskName}/skills/${skillName}/code?robot=${robot}`,
        updateData
      );

      // Update local state
      setSkillData(prev => ({
        ...prev,
        [codeType]: newCode
      }));

      return Promise.resolve();
    } catch (error) {
      console.error(`Error saving ${codeType}:`, error);
      
      // Extract error message from response
      const errorMessage = error.response?.data?.error || 
                          error.response?.data?.details || 
                          error.message || 
                          'Unknown error';
      
      throw new Error(errorMessage);
    }
  };

  const handleDownload = (codeType) => {
    if (!skillData || !skillData[codeType]) {
      toast.error(`No ${codeType} code available`);
      return;
    }

    const filename = codeType === 'rewards' 
      ? 'TaskRewardsCfg.py' 
      : codeType === 'success' 
        ? 'SuccessTerminationCfg.py'
        : `${skillName.toLowerCase()}_cfg.py`;

    const blob = new Blob([skillData[codeType]], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    toast.success(`Downloaded ${filename}`);
  };

  if (!isOpen) return null;

  const tabs = [
    { 
      id: 'rewards', 
      label: 'Reward Functions', 
      icon: <Target size={16} />,
      description: 'Define reward functions that guide the agent\'s learning'
    },
    { 
      id: 'success', 
      label: 'Success Criteria', 
      icon: <Code size={16} />,
      description: 'Define when the skill is considered successfully completed'
    },
    { 
      id: 'config', 
      label: 'Skill Config', 
      icon: <Settings size={16} />,
      description: 'Main skill configuration and environment settings'
    }
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl h-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
              <Code size={20} />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                {skillName} - Code Editor
              </h2>
              <p className="text-sm text-gray-600">
                Task: {taskName} • Robot: {robot}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              title="Refresh code"
            >
              <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
            </button>
            
            <button
              onClick={() => handleDownload(activeTab)}
              className="p-2 text-gray-600 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors"
              title="Download current file"
            >
              <Download size={16} />
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
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Description */}
        <div className="px-6 py-3 bg-blue-50 border-b border-blue-100">
          <p className="text-sm text-blue-800">
            {tabs.find(tab => tab.id === activeTab)?.description}
          </p>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-4 text-gray-600">Loading skill code...</p>
              </div>
            </div>
          ) : skillData ? (
            <div className="h-full p-6">
              {activeTab === 'rewards' && (
                <CodeEditor
                  title="Reward Functions (TaskRewardsCfg.py)"
                  code={skillData.rewards || '# No reward functions found'}
                  onSave={(code) => handleSaveCode('rewards', code)}
                  height="100%"
                  className="h-full"
                />
              )}
              
              {activeTab === 'success' && (
                <CodeEditor
                  title="Success Criteria (SuccessTerminationCfg.py)"
                  code={skillData.success || '# No success criteria found'}
                  onSave={(code) => handleSaveCode('success', code)}
                  height="100%"
                  className="h-full"
                />
              )}
              
              {activeTab === 'config' && (
                <CodeEditor
                  title={`Skill Configuration (${skillName.toLowerCase()}_cfg.py)`}
                  code={skillData.config || '# No configuration found'}
                  onSave={(code) => handleSaveCode('config', code)}
                  height="100%"
                  className="h-full"
                />
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-500">
                <Code size={48} className="mx-auto mb-4 opacity-50" />
                <p>No skill data available</p>
                <button
                  onClick={fetchSkillData}
                  className="mt-4 btn-primary"
                >
                  Retry Loading
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 bg-gray-50">
          <div className="flex justify-between items-center text-sm text-gray-600">
            <div className="flex items-center space-x-4">
              <span>📁 File: {
                activeTab === 'rewards' ? 'TaskRewardsCfg.py' :
                activeTab === 'success' ? 'SuccessTerminationCfg.py' :
                `${skillName.toLowerCase()}_cfg.py`
              }</span>
              {skillData && skillData[activeTab] && (
                <span>
                  {skillData[activeTab].split('\n').length} lines
                </span>
              )}
            </div>
            <div className="flex items-center space-x-2">
              <span>💡 Tip: Use Ctrl+S (Cmd+S) to save quickly</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SkillCodeModal;