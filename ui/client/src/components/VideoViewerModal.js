import React, { useState, useEffect } from 'react';
import { 
  X, 
  Play, 
  Download, 
  Calendar, 
  FileVideo, 
  Maximize2, 
  Minimize2,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX
} from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

const VideoViewerModal = ({ 
  isOpen, 
  onClose, 
  taskName, 
  robot 
}) => {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (isOpen && taskName && robot) {
      fetchVideos();
    }
  }, [isOpen, taskName, robot]);

  const fetchVideos = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`/api/tasks/${taskName}/videos?robot=${robot}`);
      setVideos(response.data.videos || []);
      
      // Auto-select the first video if available
      if (response.data.videos && response.data.videos.length > 0) {
        setSelectedVideo(response.data.videos[0]);
      }
    } catch (error) {
      console.error('Error fetching videos:', error);
      toast.error('Failed to fetch training videos');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
    setCurrentTime(0);
    setIsPlaying(false);
  };

  const handleDownloadVideo = (video) => {
    const link = document.createElement('a');
    link.href = video.relative_path + `?robot=${robot}`;
    link.download = video.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success('Video download started');
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className={`bg-white rounded-lg shadow-xl transition-all duration-300 ${
        isExpanded ? 'w-full max-w-7xl h-full max-h-[95vh]' : 'w-full max-w-5xl max-h-[90vh]'
      } overflow-hidden flex flex-col`}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-100 text-purple-600 rounded-lg">
              <FileVideo size={20} />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Training Videos
              </h2>
              <p className="text-sm text-gray-600">
                Task: {taskName} • Robot: {robot}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
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
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto"></div>
                <p className="mt-4 text-gray-600">Loading training videos...</p>
              </div>
            </div>
          ) : videos.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-500">
                <FileVideo size={48} className="mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium">No Training Videos Found</p>
                <p className="mt-2">
                  No videos have been recorded for this task yet. 
                  Enable video recording in training configuration to see videos here.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex h-full">
              {/* Video Player */}
              <div className="flex-1 flex flex-col bg-black">
                {selectedVideo ? (
                  <>
                    {/* Video Element */}
                    <div className="flex-1 flex items-center justify-center relative">
                      <video
                        key={selectedVideo.relative_path}
                        className="max-w-full max-h-full"
                        controls
                        muted={isMuted}
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                        onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                        onLoadedMetadata={(e) => setDuration(e.target.duration)}
                        src={selectedVideo.relative_path + `?robot=${robot}`}
                      >
                        Your browser does not support the video tag.
                      </video>
                    </div>

                    {/* Video Info */}
                    <div className="bg-gray-900 text-white p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-medium">{selectedVideo.skill_name}</h3>
                          <p className="text-sm text-gray-300">{selectedVideo.filename}</p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-gray-300">
                            {formatTime(currentTime)} / {formatTime(duration)}
                          </span>
                          <button
                            onClick={() => handleDownloadVideo(selectedVideo)}
                            className="p-2 text-gray-300 hover:text-white hover:bg-gray-700 rounded"
                            title="Download Video"
                          >
                            <Download size={16} />
                          </button>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-white">
                    <div className="text-center">
                      <Play size={48} className="mx-auto mb-4 opacity-50" />
                      <p>Select a video to play</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Video List */}
              <div className="w-80 bg-gray-50 border-l border-gray-200 flex flex-col">
                <div className="p-4 border-b border-gray-200">
                  <h3 className="font-medium text-gray-900">
                    Available Videos ({videos.length})
                  </h3>
                </div>
                
                <div className="flex-1 overflow-y-auto">
                  {videos.map((video, index) => (
                    <div
                      key={`${video.skill_name}-${video.filename}`}
                      className={`p-4 border-b border-gray-200 cursor-pointer hover:bg-gray-100 transition-colors ${
                        selectedVideo?.filename === video.filename ? 'bg-purple-50 border-purple-200' : ''
                      }`}
                      onClick={() => handleVideoSelect(video)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium text-gray-900 truncate">
                            {video.skill_name}
                          </h4>
                          <p className="text-sm text-gray-600 truncate mt-1">
                            {video.filename}
                          </p>
                          <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                            <span className="flex items-center">
                              <Calendar size={12} className="mr-1" />
                              {formatDate(video.created_at)}
                            </span>
                            <span>{formatFileSize(video.size)}</span>
                          </div>
                          <p className="text-xs text-gray-500 mt-1">
                            Experiment: {video.experiment}
                          </p>
                        </div>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadVideo(video);
                          }}
                          className="p-1 text-gray-400 hover:text-gray-600 ml-2 flex-shrink-0"
                          title="Download"
                        >
                          <Download size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 px-6 py-4 bg-gray-50">
          <div className="flex justify-between items-center text-sm text-gray-600">
            <div>
              💡 Tip: Videos are recorded during training when video recording is enabled
            </div>
            <div className="flex items-center space-x-4">
              {videos.length > 0 && (
                <span>
                  Total: {videos.length} video{videos.length !== 1 ? 's' : ''}
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

export default VideoViewerModal; 