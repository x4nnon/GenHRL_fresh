import React, { useState, useRef } from 'react';
import Editor from '@monaco-editor/react';
import { Save, Copy, RotateCcw, Check, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

const CodeEditor = ({ 
  title, 
  code, 
  onSave, 
  readOnly = false, 
  language = 'python',
  height = '400px',
  className = '' 
}) => {
  const [currentCode, setCurrentCode] = useState(code || '');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const editorRef = useRef(null);

  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor;
    
    // Configure Python language features
    monaco.languages.setLanguageConfiguration('python', {
      // Enhanced Python language configuration
      comments: {
        lineComment: '#',
        blockComment: ['"""', '"""']
      },
      brackets: [
        ['{', '}'],
        ['[', ']'],
        ['(', ')']
      ],
      autoClosingPairs: [
        { open: '{', close: '}' },
        { open: '[', close: ']' },
        { open: '(', close: ')' },
        { open: '"', close: '"', notIn: ['string'] },
        { open: "'", close: "'", notIn: ['string', 'comment'] }
      ],
      surroundingPairs: [
        { open: '{', close: '}' },
        { open: '[', close: ']' },
        { open: '(', close: ')' },
        { open: '"', close: '"' },
        { open: "'", close: "'" }
      ],
      indentationRules: {
        increaseIndentPattern: /^\s*(class|def|if|elif|else|for|while|with|try|except|finally|async def).*:$/,
        decreaseIndentPattern: /^\s*(else|elif|except|finally):/
      }
    });

    // Set editor options
    editor.updateOptions({
      fontSize: 13,
      minimap: { enabled: isExpanded },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      tabSize: 4,
      insertSpaces: true,
      wordWrap: 'on',
      lineNumbers: 'on',
      folding: true,
      bracketPairColorization: { enabled: true }
    });
  };

  const handleCodeChange = (value) => {
    setCurrentCode(value || '');
    setHasUnsavedChanges(value !== code);
  };

  const handleSave = async () => {
    if (!onSave || !hasUnsavedChanges) return;

    setIsSaving(true);
    try {
      await onSave(currentCode);
      setHasUnsavedChanges(false);
      toast.success(`${title} saved successfully!`);
    } catch (error) {
      console.error('Save error:', error);
      toast.error(`Failed to save ${title}: ${error.message || 'Unknown error'}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setCurrentCode(code || '');
    setHasUnsavedChanges(false);
    toast.success('Changes reset');
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentCode);
      toast.success('Code copied to clipboard');
    } catch (error) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = currentCode;
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      try {
        document.execCommand('copy');
        toast.success('Code copied to clipboard');
      } catch (fallbackError) {
        toast.error('Failed to copy code');
      }
      document.body.removeChild(textArea);
    }
  };

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
    setTimeout(() => {
      if (editorRef.current) {
        editorRef.current.layout();
      }
    }, 100);
  };

  return (
    <div className={`border border-gray-200 rounded-lg ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-gray-50 border-b border-gray-200 rounded-t-lg">
        <div className="flex items-center space-x-2">
          <h4 className="font-medium text-gray-900">{title}</h4>
          {hasUnsavedChanges && (
            <span className="flex items-center text-xs text-orange-600">
              <AlertCircle size={12} className="mr-1" />
              Unsaved changes
            </span>
          )}
          {!readOnly && !hasUnsavedChanges && currentCode === code && (
            <span className="flex items-center text-xs text-green-600">
              <Check size={12} className="mr-1" />
              Saved
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={toggleExpanded}
            className="text-xs text-gray-600 hover:text-gray-800 px-2 py-1 rounded"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
          
          <button
            onClick={handleCopy}
            className="p-1 text-gray-600 hover:text-gray-800 hover:bg-gray-200 rounded"
            title="Copy code"
          >
            <Copy size={14} />
          </button>

          {!readOnly && hasUnsavedChanges && (
            <button
              onClick={handleReset}
              className="p-1 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded"
              title="Reset changes"
            >
              <RotateCcw size={14} />
            </button>
          )}

          {!readOnly && (
            <button
              onClick={handleSave}
              disabled={!hasUnsavedChanges || isSaving}
              className={`flex items-center px-2 py-1 text-xs rounded ${
                hasUnsavedChanges && !isSaving
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-200 text-gray-500 cursor-not-allowed'
              }`}
              title="Save changes"
            >
              <Save size={12} className="mr-1" />
              {isSaving ? 'Saving...' : 'Save'}
            </button>
          )}
        </div>
      </div>

      {/* Editor */}
      <div className="relative">
        <Editor
          height={isExpanded ? '600px' : height}
          language={language}
          value={currentCode}
          onChange={handleCodeChange}
          onMount={handleEditorDidMount}
          options={{
            readOnly: readOnly,
            theme: 'light',
            automaticLayout: true,
            scrollBeyondLastLine: false,
            minimap: { enabled: isExpanded },
            fontSize: 13,
            tabSize: 4,
            insertSpaces: true,
            wordWrap: 'on',
            lineNumbers: 'on',
            folding: true,
            bracketPairColorization: { enabled: true },
            padding: { top: 10, bottom: 10 }
          }}
          loading={
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            </div>
          }
        />
      </div>

      {/* Footer with info */}
      <div className="px-3 py-2 bg-gray-50 border-t border-gray-200 rounded-b-lg text-xs text-gray-600">
        <div className="flex justify-between items-center">
          <span>
            {currentCode.split('\n').length} lines, {currentCode.length} characters
          </span>
          {!readOnly && (
            <span>
              Use Ctrl+S (Cmd+S on Mac) to save quickly
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default CodeEditor;