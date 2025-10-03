import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Plus, 
  List, 
  Bot, 
  Settings,
  Activity,
  ServerOff
} from 'lucide-react';
import ActiveSessionsIndicator from './ActiveSessionsIndicator';
import axios from 'axios';
import toast from 'react-hot-toast';

const navItems = [
  { path: '/', label: 'Dashboard', icon: Home },
  { path: '/create', label: 'Create Task', icon: Plus },
  { path: '/tasks', label: 'All Tasks', icon: List },
];

const Sidebar = () => {
  const location = useLocation();

  const handleRestartServer = async () => {
    if (window.confirm('Are you sure you want to restart the server? This will terminate all active processes.')) {
      try {
        toast.loading('Restarting server...', { id: 'restart-toast' });
        await axios.post('/api/server/restart');
        // The server will restart, and the page will become unavailable.
        // The user will need to refresh the page after a few moments.
        // We can show a toast message that will persist until the user refreshes.
        toast.success('Server is restarting. Please refresh the page in a moment.', { id: 'restart-toast', duration: 10000 });
      } catch (error) {
        toast.error('Failed to restart server.', { id: 'restart-toast' });
      }
    }
  };

  return (
    <div className="sidebar">
      <div className="mb-8">
        <h1 className="text-xl font-bold text-gray-800 flex items-center">
          <Bot className="mr-2 text-blue-600" size={24} />
          GenHRL UI
        </h1>
        <p className="text-sm text-gray-600 mt-1">
          Hierarchical RL Task Manager
        </p>
      </div>

      <nav className="space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center px-3 py-2 rounded-lg transition-colors ${
                isActive 
                  ? 'bg-blue-100 text-blue-700 font-medium' 
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Icon size={18} className="mr-3" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-8 pt-8 border-t border-gray-200">
        <div className="text-xs text-gray-500 mb-2">SYSTEM STATUS</div>
        <div className="flex items-center text-sm text-green-600">
          <Activity size={16} className="mr-2" />
          Connected
        </div>
      </div>

      {/* Active Sessions Indicator */}
      <div className="mt-4">
        <ActiveSessionsIndicator />
      </div>

      <div className="absolute bottom-4 left-4 right-4">
        <div className="text-xs text-gray-500 mb-2">GenHRL v2.0</div>
        <button 
          onClick={handleRestartServer}
          className="w-full text-left text-xs text-gray-600 hover:text-red-600 flex items-center mb-2"
        >
          <ServerOff size={14} className="mr-2" />
          Restart Server
        </button>
        <button className="w-full text-left text-xs text-gray-600 hover:text-gray-800 flex items-center">
          <Settings size={14} className="mr-2" />
          Settings
        </button>
      </div>
    </div>
  );
};

export default Sidebar;