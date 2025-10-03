import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Calendar, Bot, ChevronRight, Activity, AlertCircle } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalTasks: 0,
    recentTasks: 0,
    activeTraining: 0
  });

  useEffect(() => {
    fetchTasks();
  }, []);

  const fetchTasks = async () => {
    try {
      const response = await axios.get('/api/tasks');
      const tasksData = response.data;
      setTasks(tasksData);
      
      // Calculate stats
      const now = new Date();
      const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const recentTasks = tasksData.filter(task => 
        new Date(task.createdAt) > oneDayAgo
      ).length;
      
      setStats({
        totalTasks: tasksData.length,
        recentTasks,
        activeTraining: 0 // Will be updated when training integration is added
      });
    } catch (error) {
      console.error('Error fetching tasks:', error);
      toast.error('Failed to fetch tasks');
    } finally {
      setLoading(false);
    }
  };

  const getHierarchyInfo = (hierarchy) => {
    if (!hierarchy) return 'Unknown';
    
    const countSkills = (node) => {
      if (!node.children || node.children.length === 0) return 1;
      return node.children.reduce((count, child) => count + countSkills(child), 0);
    };
    
    if (Array.isArray(hierarchy)) {
      return `${hierarchy.length} skills`;
    } else if (hierarchy.children) {
      return `${countSkills(hierarchy)} skills`;
    } else {
      return '1 skill';
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
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Overview of your GenHRL tasks and training progress
          </p>
        </div>
        <Link to="/create" className="btn-primary flex items-center">
          <Plus size={18} className="mr-2" />
          Create New Task
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 bg-blue-100 rounded-lg">
              <Bot className="text-blue-600" size={24} />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-600">Total Tasks</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalTasks}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="p-3 bg-green-100 rounded-lg">
              <Calendar className="text-green-600" size={24} />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-600">Created Today</p>
              <p className="text-2xl font-bold text-gray-900">{stats.recentTasks}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="p-3 bg-orange-100 rounded-lg">
              <Activity className="text-orange-600" size={24} />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-600">Training Active</p>
              <p className="text-2xl font-bold text-gray-900">{stats.activeTraining}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Tasks */}
      <div className="card">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold text-gray-900">Recent Tasks</h2>
          <Link to="/tasks" className="text-blue-600 hover:text-blue-700 flex items-center">
            View All
            <ChevronRight size={16} className="ml-1" />
          </Link>
        </div>

        {tasks.length === 0 ? (
          <div className="text-center py-8">
            <AlertCircle className="mx-auto text-gray-400 mb-4" size={48} />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No tasks yet</h3>
            <p className="text-gray-600 mb-4">
              Create your first hierarchical RL task to get started
            </p>
            <Link to="/create" className="btn-primary">
              Create Task
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {tasks.slice(0, 5).map((task) => (
              <Link
                key={task.name}
                to={`/tasks/${task.name}`}
                className="block p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:shadow-md transition-all"
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900 mb-1">{task.name}</h3>
                    <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                      {task.description}
                    </p>
                    <div className="flex items-center text-xs text-gray-500 space-x-4">
                      <span>Robot: {task.robot}</span>
                      <span>Skills: {getHierarchyInfo(task.skillsHierarchy)}</span>
                      <span>Created: {new Date(task.createdAt).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <ChevronRight size={16} className="text-gray-400 ml-4 flex-shrink-0" />
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Quick Actions</h3>
          <div className="space-y-2">
            <Link to="/create" className="block w-full text-left btn-secondary">
              Create New Task
            </Link>
            <Link to="/tasks" className="block w-full text-left btn-secondary">
              Browse All Tasks
            </Link>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">System Info</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">GenHRL Version:</span>
              <span className="font-medium">v2.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">IsaacLab Status:</span>
              <span className="text-green-600 font-medium">Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">API Status:</span>
              <span className="text-green-600 font-medium">Ready</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;