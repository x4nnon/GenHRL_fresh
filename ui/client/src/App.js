import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { ActiveSessionProvider } from './context/ActiveSessionContext';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import CreateTask from './pages/CreateTask';
import TaskDetail from './pages/TaskDetail';
import TaskList from './pages/TaskList';

function App() {
  return (
    <ActiveSessionProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/create" element={<CreateTask />} />
              <Route path="/tasks" element={<TaskList />} />
              <Route path="/tasks/:taskName" element={<TaskDetail />} />
            </Routes>
          </main>
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </div>
      </Router>
    </ActiveSessionProvider>
  );
}

export default App;