import React, { useState, useEffect } from 'react';
import { 
  X, 
  TrendingUp, 
  BarChart3, 
  Activity, 
  Calendar, 
  RefreshCw,
  Download,
  Maximize2,
  Minimize2,
  ExternalLink,
  Zap,
  Target,
  Clock
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import axios from 'axios';
import toast from 'react-hot-toast';

const TrainingMetricsModal = ({ 
  isOpen, 
  onClose, 
  taskName, 
  robot 
}) => {
  const [metrics, setMetrics] = useState({});
  const [loading, setLoading] = useState(false);
  const [selectedSkill, setSelectedSkill] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('Reward/mean');
  const [isExpanded, setIsExpanded] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (isOpen && taskName && robot) {
      fetchMetrics();
    }
  }, [isOpen, taskName, robot]);

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`/api/tasks/${taskName}/metrics?robot=${robot}`);
      setMetrics(response.data.metrics || {});
      
      // Auto-select the first skill if available
      const skillNames = Object.keys(response.data.metrics || {});
      if (skillNames.length > 0) {
        setSelectedSkill(skillNames[0]);
      }
    } catch (error) {
      console.error('Error fetching metrics:', error);
      toast.error('Failed to fetch training metrics');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchMetrics();
    setRefreshing(false);
    toast.success('Metrics refreshed');
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const getChartData = () => {
    if (!selectedSkill || !metrics[selectedSkill] || !metrics[selectedSkill].tensorboard_data || !metrics[selectedSkill].tensorboard_data.scalars) {
      return [];
    }

    const scalarData = metrics[selectedSkill].tensorboard_data.scalars;
    const metricData = scalarData[selectedMetric];
    
    if (!metricData) return [];

    return metricData.map(point => ({
      step: point.step,
      value: point.value,
      time: new Date(point.wall_time * 1000).toLocaleTimeString()
    }));
  };

  const getAvailableMetrics = () => {
    if (!selectedSkill || !metrics[selectedSkill] || !metrics[selectedSkill].tensorboard_data || !metrics[selectedSkill].tensorboard_data.tags) {
      return [];
    }

    return metrics[selectedSkill].tensorboard_data.tags;
  };

  const getMetricColor = (metricName) => {
    const colors = {
      'Reward/mean': '#8884d8',
      'Reward/max': '#82ca9d',
      'Reward/min': '#ffc658',
      'Loss/policy': '#ff7300',
      'Loss/value': '#8dd1e1',
      'Learning_rate': '#d084d0',
      'Episode/length_mean': '#82ca9d',
      'Episode/return_mean': '#8884d8',
      'Episode/success_rate': '#00ff00'
    };
    return colors[metricName] || '#8884d8';
  };

  const getLatestValue = (metricName) => {
    if (!selectedSkill || !metrics[selectedSkill] || !metrics[selectedSkill].tensorboard_data || !metrics[selectedSkill].tensorboard_data.scalars) {
      return 'N/A';
    }

    const scalarData = metrics[selectedSkill].tensorboard_data.scalars;
    const metricData = scalarData[metricName];
    
    if (!metricData || metricData.length === 0) return 'N/A';
    
    const latestValue = metricData[metricData.length - 1].value;
    return typeof latestValue === 'number' ? latestValue.toFixed(4) : latestValue;
  };

  const skillNames = Object.keys(metrics);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className={`bg-white rounded-lg shadow-xl transition-all duration-300 ${
        isExpanded ? 'w-full max-w-7xl h-full max-h-[95vh]' : 'w-full max-w-6xl max-h-[90vh]'
      } overflow-hidden flex flex-col`}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
              <TrendingUp size={20} />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Training Metrics
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
              title="Refresh metrics"
            >
              <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
            </button>
            
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              title={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
            
            <button
              onClick={onClose}
              className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-4 text-gray-600">Loading training metrics...</p>
              </div>
            </div>
          ) : skillNames.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-500">
                <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium">No Training Metrics Found</p>
                <p className="mt-2">
                  No training has been performed for this task yet. 
                  Start training to see metrics and logs here.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex h-full">
              {/* Skills Sidebar */}
              <div className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col">
                <div className="p-4 border-b border-gray-200">
                  <h3 className="font-medium text-gray-900">Skills ({skillNames.length})</h3>
                </div>
                
                <div className="flex-1 overflow-y-auto">
                  {skillNames.map((skillName) => {
                    const skillData = metrics[skillName];
                    return (
                      <div
                        key={skillName}
                        className={`p-4 border-b border-gray-200 cursor-pointer hover:bg-gray-100 transition-colors ${
                          selectedSkill === skillName ? 'bg-blue-50 border-blue-200' : ''
                        }`}
                        onClick={() => setSelectedSkill(skillName)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <h4 className="font-medium text-gray-900 truncate">
                              {skillName}
                            </h4>
                            <div className="flex items-center space-x-2 mt-1">
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                                skillData.skill_type === 'primitive' 
                                  ? 'bg-green-100 text-green-800' 
                                  : 'bg-purple-100 text-purple-800'
                              }`}>
                                {skillData.skill_type}
                              </span>
                            </div>
                            <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                              <span className="flex items-center">
                                <Calendar size={10} className="mr-1" />
                                {formatDate(skillData.created_at)}
                              </span>
                            </div>
                            <div className="flex items-center space-x-2 mt-1 text-xs">
                              {skillData.has_checkpoints && (
                                <span className="text-green-600">✓ Checkpoints</span>
                              )}
                              {skillData.has_videos && (
                                <span className="text-blue-600">✓ Videos</span>
                              )}
                              {skillData.tensorboard_data?.has_data && (
                                <span className="text-purple-600">✓ Metrics</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Main Content */}
              <div className="flex-1 flex flex-col">
                {selectedSkill && metrics[selectedSkill] ? (
                  <>
                    {/* Skill Header */}
                    <div className="p-6 border-b border-gray-200 bg-white">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-gray-900">{selectedSkill}</h3>
                          <p className="text-sm text-gray-600 mt-1">
                            Experiment: {metrics[selectedSkill].experiment_name}
                          </p>
                        </div>
                        <div className="flex items-center space-x-4">
                          {/* Quick Stats */}
                          <div className="grid grid-cols-3 gap-4 text-center">
                            <div className="bg-green-50 p-3 rounded-lg">
                              <div className="text-lg font-bold text-green-600">
                                {getLatestValue('Reward/mean')}
                              </div>
                              <div className="text-xs text-gray-600">Latest Reward</div>
                            </div>
                            <div className="bg-blue-50 p-3 rounded-lg">
                              <div className="text-lg font-bold text-blue-600">
                                {getLatestValue('Episode/length_mean')}
                              </div>
                              <div className="text-xs text-gray-600">Episode Length</div>
                            </div>
                            <div className="bg-purple-50 p-3 rounded-lg">
                              <div className="text-lg font-bold text-purple-600">
                                {getLatestValue('Episode/success_rate') !== 'N/A' ? 
                                  (parseFloat(getLatestValue('Episode/success_rate')) * 100).toFixed(1) + '%' :
                                  'N/A'
                                }
                              </div>
                              <div className="text-xs text-gray-600">Success Rate</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Metrics Selection */}
                    <div className="p-4 border-b border-gray-200 bg-gray-50">
                      <div className="flex items-center space-x-4">
                        <label className="text-sm font-medium text-gray-700">Metric:</label>
                        <select
                          value={selectedMetric}
                          onChange={(e) => setSelectedMetric(e.target.value)}
                          className="input-field w-auto"
                        >
                          {getAvailableMetrics().map(metric => (
                            <option key={metric} value={metric}>{metric}</option>
                          ))}
                        </select>
                        <div className="flex items-center space-x-2 text-sm text-gray-600">
                          <span>Latest Value:</span>
                          <span className="font-medium text-blue-600">{getLatestValue(selectedMetric)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Chart */}
                    <div className="flex-1 p-6">
                      {metrics[selectedSkill].tensorboard_data?.has_data ? (
                        <div className="h-full">
                          <div className="flex items-center justify-between mb-4">
                            <h4 className="text-lg font-medium text-gray-900">{selectedMetric}</h4>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                              <Activity size={14} />
                              <span>{getChartData().length} data points</span>
                            </div>
                          </div>
                          
                          <ResponsiveContainer width="100%" height="80%">
                            <LineChart data={getChartData()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis 
                                dataKey="step" 
                                label={{ value: 'Training Steps', position: 'insideBottom', offset: -10 }}
                              />
                              <YAxis 
                                label={{ value: selectedMetric, angle: -90, position: 'insideLeft' }}
                              />
                              <Tooltip 
                                labelFormatter={(value) => `Step: ${value}`}
                                formatter={(value, name) => [value.toFixed(4), selectedMetric]}
                              />
                              <Legend />
                              <Line
                                type="monotone"
                                dataKey="value"
                                stroke={getMetricColor(selectedMetric)}
                                strokeWidth={2}
                                dot={false}
                                name={selectedMetric}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <div className="text-center text-gray-500">
                            <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
                            <p className="text-lg font-medium">No Metrics Data</p>
                            <p className="mt-2">
                              {metrics[selectedSkill].tensorboard_data?.message || 
                               'No TensorBoard logs found for this skill.'}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center text-gray-500">
                      <Target size={48} className="mx-auto mb-4 opacity-50" />
                      <p>Select a skill to view metrics</p>
                    </div>
                  </div>
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
              {selectedSkill && metrics[selectedSkill]?.tensorboard_data?.event_file && (
                <span className="text-xs">
                  Source: {metrics[selectedSkill].tensorboard_data.event_file.split('/').pop()}
                </span>
              )}
            </div>
            <div className="flex items-center space-x-4">
              {skillNames.length > 0 && (
                <span>
                  {skillNames.length} skill{skillNames.length !== 1 ? 's' : ''} with metrics
                </span>
              )}
              <button
                onClick={onClose}
                className="btn-secondary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingMetricsModal; 