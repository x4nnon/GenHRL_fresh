import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { 
  Search, 
  Filter, 
  Calendar, 
  Bot, 
  Trash2,
  Eye,
  Play,
  AlertCircle
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import TrainingModal from '../components/TrainingModal';
import TrainingProgressModal from '../components/TrainingProgressModal';

const TaskList = () => {
  const [tasks, setTasks] = useState([]);
  const [filteredTasks, setFilteredTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRobot, setSelectedRobot] = useState('all');
  const [sortBy, setSortBy] = useState('newest');
  const [robots, setRobots] = useState([]);
  const [isTrainingModalOpen, setIsTrainingModalOpen] = useState(false);
  const [isTrainingProgressOpen, setIsTrainingProgressOpen] = useState(false);
  const [trainingSessionId, setTrainingSessionId] = useState(null);
  const [selectedTask, setSelectedTask] = useState(null);

  useEffect(() => {
    fetchTasks();
    fetchRobots();
  }, []);

  const fetchTasks = async () => {
    try {
      setLoading(true);
      // Fetch tasks for all robots
      const allRobots = ['G1', 'H1', 'Franka', 'UR10'];
      const taskPromises = allRobots.map(robot => 
        axios.get(`/api/tasks?robot=${robot}`).catch(() => ({ data: [] }))
      );
      
      const responses = await Promise.all(taskPromises);
      const allTasks = responses.flatMap(response => response.data);
      
      setTasks(allTasks);
    } catch (error) {
      console.error('Error fetching tasks:', error);
      toast.error('Failed to fetch tasks');
    } finally {
      setLoading(false);
    }
  };

  const fetchRobots = async () => {
    try {
      const response = await axios.get('/api/robots');
      setRobots(response.data);
    } catch (error) {
      console.error('Error fetching robots:', error);
    }
  };

  const filterAndSortTasks = useCallback(() => {
    let filtered = tasks;

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(task =>
        task.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        task.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Filter by robot
    if (selectedRobot !== 'all') {
      filtered = filtered.filter(task => task.robot === selectedRobot);
    }

    // Sort tasks
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'newest':
          return new Date(b.createdAt) - new Date(a.createdAt);
        case 'oldest':
          return new Date(a.createdAt) - new Date(b.createdAt);
        case 'name':
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });

    setFilteredTasks(filtered);
  }, [tasks, searchTerm, selectedRobot, sortBy]);

  useEffect(() => {
    filterAndSortTasks();
  }, [filterAndSortTasks]);

  const deleteTask = async (taskName, robot) => {
    if (!window.confirm(`Are you sure you want to delete task "${taskName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      await axios.delete(`/api/tasks/${taskName}?robot=${robot}`);
      toast.success('Task deleted successfully');
      fetchTasks(); // Refresh the list
    } catch (error) {
      console.error('Error deleting task:', error);
      toast.error('Failed to delete task');
    }
  };

  const handleStartTraining = (task) => {
    setSelectedTask(task);
    setIsTrainingModalOpen(true);
  };

  const handleTrainingStart = (sessionId) => {
    setTrainingSessionId(sessionId);
    setIsTrainingModalOpen(false);
    setIsTrainingProgressOpen(true);
  };

  const handleTrainingComplete = (success, message) => {
    // Refresh the task list to get updated training status
    fetchTasks();
  };

  const closeTrainingProgress = () => {
    setIsTrainingProgressOpen(false);
    setTrainingSessionId(null);
    setSelectedTask(null);
  };

  const getHierarchyInfo = (hierarchy) => {
    if (!hierarchy) return { levels: 'Unknown', skills: 0 };
    
    const countSkills = (node) => {
      if (!node.children || node.children.length === 0) return 1;
      return node.children.reduce((count, child) => count + countSkills(child), 0);
    };
    
    const getDepth = (node) => {
      if (!node.children || node.children.length === 0) return 1;
      return 1 + Math.max(...node.children.map(child => getDepth(child)));
    };
    
    if (Array.isArray(hierarchy)) {
      return { levels: '2', skills: hierarchy.length };
    } else if (hierarchy.children) {
      return { levels: getDepth(hierarchy).toString(), skills: countSkills(hierarchy) };
    } else {
      return { levels: '1', skills: 1 };
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">All Tasks</h1>
          <p className="text-gray-600 mt-1">
            Manage and explore your generated hierarchical RL tasks
          </p>
        </div>
        <Link to="/create" className="btn-primary">
          Create New Task
        </Link>
      </div>

      {/* Filters and Search */}
      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
            <input
              type="text"
              placeholder="Search tasks..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input-field pl-10"
            />
          </div>

          {/* Robot Filter */}
          <div>
            <select
              value={selectedRobot}
              onChange={(e) => setSelectedRobot(e.target.value)}
              className="input-field"
            >
              <option value="all">All Robots</option>
              {robots.map(robot => (
                <option key={robot} value={robot}>{robot}</option>
              ))}
            </select>
          </div>

          {/* Sort */}
          <div>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="input-field"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="name">Name (A-Z)</option>
            </select>
          </div>

          {/* Results Count */}
          <div className="flex items-center text-sm text-gray-600">
            <Filter size={16} className="mr-2" />
            {filteredTasks.length} of {tasks.length} tasks
          </div>
        </div>
      </div>

      {/* Tasks Grid */}
      {filteredTasks.length === 0 ? (
        <div className="card text-center py-12">
          <AlertCircle className="mx-auto text-gray-400 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {tasks.length === 0 ? 'No tasks found' : 'No tasks match your filters'}
          </h3>
          <p className="text-gray-600 mb-4">
            {tasks.length === 0 
              ? 'Create your first hierarchical RL task to get started'
              : 'Try adjusting your search terms or filters'
            }
          </p>
          {tasks.length === 0 && (
            <Link to="/create" className="btn-primary">
              Create Your First Task
            </Link>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredTasks.map((task) => {
            const hierarchyInfo = getHierarchyInfo(task.skillsHierarchy);
            
            return (
              <div key={`${task.robot}-${task.name}`} className="card hover:shadow-lg transition-shadow">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="font-semibold text-gray-900 text-lg">{task.name}</h3>
                  <div className="flex space-x-2">
                    <Link
                      to={`/tasks/${task.name}?robot=${task.robot}`}
                      className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                      title="View Details"
                    >
                      <Eye size={16} />
                    </Link>
                    <button
                      onClick={() => deleteTask(task.name, task.robot)}
                      className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Delete Task"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>

                <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                  {task.description}
                </p>

                <div className="space-y-3">
                  {/* Task Stats */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center text-gray-600">
                      <Bot size={14} className="mr-2" />
                      {task.robot}
                    </div>
                    <div className="flex items-center text-gray-600">
                      <Calendar size={14} className="mr-2" />
                      {new Date(task.createdAt).toLocaleDateString()}
                    </div>
                  </div>

                  {/* Hierarchy Info */}
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Hierarchy Levels:</span>
                        <span className="ml-2 font-medium">{hierarchyInfo.levels}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Skills:</span>
                        <span className="ml-2 font-medium">{hierarchyInfo.skills}</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex space-x-2 pt-2">
                    <Link
                      to={`/tasks/${task.name}?robot=${task.robot}`}
                      className="flex-1 btn-primary text-center flex items-center justify-center"
                    >
                      <Eye size={14} className="mr-2" />
                      View Details
                    </Link>
                    <button
                      onClick={() => handleStartTraining(task)}
                      className="btn-secondary flex items-center"
                      title="Start Training"
                    >
                      <Play size={14} />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Training Configuration Modal */}
      {selectedTask && (
        <TrainingModal
          isOpen={isTrainingModalOpen}
          onClose={() => setIsTrainingModalOpen(false)}
          taskName={selectedTask.name}
          robot={selectedTask.robot}
          onTrainingStart={handleTrainingStart}
        />
      )}

      {/* Training Progress Modal */}
      {selectedTask && (
        <TrainingProgressModal
          isOpen={isTrainingProgressOpen}
          onClose={closeTrainingProgress}
          taskName={selectedTask.name}
          robot={selectedTask.robot}
          sessionId={trainingSessionId}
          onTrainingComplete={handleTrainingComplete}
        />
      )}
    </div>
  );
};

export default TaskList;