import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link, useLocation } from 'react-router-dom';
import { 
  ChevronRight, 
  ChevronDown, 
  Play, 
  Eye, 
  Code,
  Box,
  BrainCircuit,
  BarChart3,
  FileVideo,
  Settings,
  ArrowLeft,
  Trash2,
  Edit,
  Target,
  CheckCircle
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import TrainingModal from '../components/TrainingModal';
import TrainingProgressModal from '../components/TrainingProgressModal';
import SkillCodeModal from '../components/SkillCodeModal';
import TrainingMetricsModal from '../components/TrainingMetricsModal';
import VideoViewerModal from '../components/VideoViewerModal';

const MINIMUM_VOLUME_THRESHOLD = 0.001; // e.g., 1 cubic centimeter

const TaskDetail = () => {
  const { taskName } = useParams();
  const [task, setTask] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedSkills, setExpandedSkills] = useState({});
  const [isTrainingModalOpen, setIsTrainingModalOpen] = useState(false);
  const [isTrainingProgressOpen, setIsTrainingProgressOpen] = useState(false);
  const [trainingSessionId, setTrainingSessionId] = useState(null);
  const [isMetricsModalOpen, setIsMetricsModalOpen] = useState(false);
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false);
  const [objectConfig, setObjectConfig] = useState(null);
  const [showObjectConfig, setShowObjectConfig] = useState(false);
  // Add state for editing skill
  const [editingSkill, setEditingSkill] = useState(null);
  const [isSkillCodeModalOpen, setIsSkillCodeModalOpen] = useState(false);

  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const robot = queryParams.get('robot') || 'G1'; // Default to G1 if not provided

  useEffect(() => {
    const fetchTaskDetails = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`/api/tasks/${taskName}?robot=${robot}`);
        setTask(response.data);
      } catch (error) {
        console.error('Error fetching task details:', error);
        toast.error('Failed to load task');
      } finally {
        setLoading(false);
      }
    };

    const fetchObjectConfig = async () => {
      try {
        const response = await axios.get(`/api/tasks/${taskName}/object_config?robot=${robot}`);
        setObjectConfig(response.data);
      } catch (error) {
        console.error('Error fetching object config:', error);
        // Do not show a toast error if the file is just not found
        if (error.response?.status !== 404) {
          toast.error('Failed to load object configuration');
        }
      }
    };

    if (taskName && robot) {
      fetchTaskDetails();
      fetchObjectConfig();
    }
  }, [taskName, robot]);

  const toggleSkillExpansion = (skillName) => {
    setExpandedSkills(prev => ({ ...prev, [skillName]: !prev[skillName] }));
  };

  const handleTrainingStart = (sessionId) => {
    setTrainingSessionId(sessionId);
    setIsTrainingProgressOpen(true);
    setIsTrainingModalOpen(false);
  };
  
  const handleTrainingComplete = () => {
    // Maybe refresh some data here in the future
  };

  const closeTrainingProgress = () => {
    setIsTrainingProgressOpen(false);
    setTrainingSessionId(null);
  };
  
  const calculateVolume = (type, params) => {
    const PI = Math.PI;
    switch(type) {
      case 'box': return params.size.x * params.size.y * params.size.z;
      case 'sphere': return (4/3) * PI * Math.pow(params.radius, 3);
      case 'cylinder': return PI * Math.pow(params.radius, 2) * params.height;
      case 'capsule': return PI * Math.pow(params.radius, 2) * params.height + (4/3) * PI * Math.pow(params.radius, 3);
      default: return 0;
    }
  };

  const countMeaningfulObjects = (objectConfig) => {
    if (!objectConfig) return 0;
    
    let count = 0;
    const { num_boxes, num_spheres, num_cylinders, num_capsules } = objectConfig;
    
    if (num_boxes > 0 && objectConfig.size_boxes) {
      for (let i = 0; i < num_boxes; i++) {
        const size = { x: objectConfig.size_boxes[i*3], y: objectConfig.size_boxes[i*3+1], z: objectConfig.size_boxes[i*3+2] };
        if (calculateVolume('box', { size }) > MINIMUM_VOLUME_THRESHOLD) count++;
      }
    }
    
    if (num_spheres > 0 && objectConfig.radius_spheres) {
      for (let i = 0; i < num_spheres; i++) {
        if (calculateVolume('sphere', { radius: objectConfig.radius_spheres[i] }) > MINIMUM_VOLUME_THRESHOLD) count++;
      }
    }
    
    if (num_cylinders > 0 && objectConfig.radius_cylinders && objectConfig.height_cylinders) {
      for (let i = 0; i < num_cylinders; i++) {
        const radius = objectConfig.radius_cylinders[i] || 0;
        const height = objectConfig.height_cylinders[i] || 0;
        const volume = calculateVolume('cylinder', { radius, height });
        if (volume > MINIMUM_VOLUME_THRESHOLD) count++;
      }
    }
    
    if (num_capsules > 0 && objectConfig.radius_capsules && objectConfig.height_capsules) {
      for (let i = 0; i < num_capsules; i++) {
        const radius = objectConfig.radius_capsules[i] || 0;
        const height = objectConfig.height_capsules[i] || 0;
        const volume = calculateVolume('capsule', { radius, height });
        if (volume > MINIMUM_VOLUME_THRESHOLD) count++;
      }
    }
    
    return count;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!task) {
    return (
      <div className="text-center py-8">
        <h1 className="text-xl font-semibold text-gray-900">Task not found</h1>
        <Link to="/tasks" className="text-blue-600 hover:text-blue-700 mt-4 inline-block">
          Back to Tasks
        </Link>
      </div>
    );
  }

  const renderHierarchy = (node, level = 0) => {
    if (!node) return null;
    
    const isExpanded = expandedSkills[node.name] ?? true;
    const hasChildren = node.children && node.children.length > 0;
    
    return (
      <div key={node.name} className={`ml-${level * 4}`}>
        <div className="flex items-center py-2">
          {hasChildren && (
            <button 
              onClick={() => toggleSkillExpansion(node.name)}
              className="mr-2 p-1"
            >
              {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </button>
          )}
          <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-lg text-sm">
            {node.name}
          </div>
          <span className="ml-2 text-sm text-gray-600">{node.description}</span>
          {/* Edit button for each skill */}
          <button
            className="ml-2 btn-secondary flex items-center px-2 py-1 text-xs"
            onClick={() => {
              setEditingSkill(node.name);
              setIsSkillCodeModalOpen(true);
            }}
          >
            <Edit size={12} className="mr-1" /> Edit
          </button>
        </div>
        {hasChildren && isExpanded && (
          <div className="ml-4">
            {node.children.map(child => renderHierarchy(child, level + 1))}
          </div>
        )}
      </div>
    );
  };
  
  return (
    <div className="space-y-6">
      <Link to="/tasks" className="flex items-center text-blue-600 hover:text-blue-700">
        <ArrowLeft size={16} className="mr-2" />
        Back to All Tasks
      </Link>
      
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">{task.name}</h1>
          <p className="text-gray-600 mt-1 max-w-2xl">{task.description}</p>
        </div>
        <div className="flex items-center space-x-2">
          <button className="btn-secondary flex items-center">
            <Edit size={16} className="mr-2" /> Edit
          </button>
          <button 
            onClick={() => setIsTrainingModalOpen(true)}
            className="btn-primary flex items-center"
          >
            <Play size={16} className="mr-2" />
            Train Task
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Task Details</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="font-medium text-gray-600">Robot</span>
              <span className="text-gray-800">{task.robot}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium text-gray-600">Created At</span>
              <span className="text-gray-800">{new Date(task.createdAt).toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium text-gray-600">Total Skills</span>
              <span className="text-gray-800">{task.totalSkills}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium text-gray-600">Hierarchy Levels</span>
              <span className="text-gray-800">{task.hierarchyLevels}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium text-gray-600">Meaningful Objects</span>
              <span className="text-gray-800">{countMeaningfulObjects(task.objectConfig)}</span>
            </div>
          </div>
        </div>

        {objectConfig && (
          <div className="card">
            <button
              onClick={() => setShowObjectConfig(!showObjectConfig)}
              className="w-full flex justify-between items-center text-left"
            >
              <h2 className="text-xl font-semibold text-gray-900">Object Configuration</h2>
              {showObjectConfig ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
            </button>
            {showObjectConfig && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg max-h-96 overflow-auto custom-scrollbar">
                <pre className="text-sm">{JSON.stringify(objectConfig, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
        
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Training Tools</h2>
          <div className="space-y-4">
            <button 
              onClick={() => setIsMetricsModalOpen(true)}
              className="w-full btn-secondary flex items-center justify-start"
            >
              <BarChart3 size={16} className="mr-2"/>
              View Training Metrics
            </button>
            <button 
              onClick={() => setIsVideoModalOpen(true)}
              className="w-full btn-secondary flex items-center justify-start"
            >
              <FileVideo size={16} className="mr-2"/>
              View Training Videos
            </button>
          </div>
        </div>
      </div>

      {/* Skill Hierarchy */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Skill Hierarchy</h2>
        {renderHierarchy(task.skillsHierarchy)}
      </div>

      {/* Modals */}
      <TrainingModal 
        isOpen={isTrainingModalOpen}
        onClose={() => setIsTrainingModalOpen(false)}
        taskName={taskName}
        robot={robot}
        onTrainingStart={handleTrainingStart}
      />
      <TrainingProgressModal 
        isOpen={isTrainingProgressOpen}
        onClose={closeTrainingProgress}
        taskName={taskName}
        robot={robot}
        sessionId={trainingSessionId}
        onTrainingComplete={handleTrainingComplete}
      />
      <TrainingMetricsModal
        isOpen={isMetricsModalOpen}
        onClose={() => setIsMetricsModalOpen(false)}
        taskName={taskName}
        robot={robot}
      />
      <VideoViewerModal
        isOpen={isVideoModalOpen}
        onClose={() => setIsVideoModalOpen(false)}
        taskName={taskName}
        robot={robot}
      />
      {/* SkillCodeModal for editing skill code */}
      <SkillCodeModal
        isOpen={isSkillCodeModalOpen}
        onClose={() => setIsSkillCodeModalOpen(false)}
        taskName={taskName}
        skillName={editingSkill}
        robot={robot}
      />
    </div>
  );
};

export default TaskDetail;