'use client';

import { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { ArrowLeft, Play, Save, Download, List, Target, Zap, FileText } from 'lucide-react';

// Entry components
import { EntryMethodSelector } from './entry/EntryMethodSelector';
import { YouTubeInputPanel } from './entry/YouTubeInputPanel';

// Interactive components removed - using single panel approach

// Shared components
import { ScriptAnalysisDisplay } from './shared/ScriptAnalysisDisplay';
import { SimpleWordCounter } from './shared/SimpleWordCounter';
import { ScriptPanel } from './shared/ScriptPanel';
import { DevModeToggle } from './DevModeToggle';

// Types
import { SavedScript } from '../../types/settings';

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  command?: string;
  status?: 'sending' | 'sent' | 'error';
}

export interface ScriptSession {
  id: string;
  entryMethod: 'youtube' | null;
  sourceUrl?: string;
  currentScript: string;
  messages: ChatMessage[];
  wordCount: number;
  isFinalized: boolean;
}

interface ScriptBuilderProps {
  onFinalize?: (session: ScriptSession) => void;
  onBack?: () => void;
  className?: string;
}

export function ScriptBuilder({ onFinalize, onBack, className = '' }: ScriptBuilderProps) {
  const [currentStep, setCurrentStep] = useState<'entry' | 'youtube' | 'building'>('entry');
  const [session, setSession] = useState<ScriptSession>({
    id: '',
    entryMethod: null,
    currentScript: '',
    messages: [],
    wordCount: 0,
    isFinalized: false
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Skip mode state
  const [skipMode, setSkipMode] = useState(false);
  const [availableScripts, setAvailableScripts] = useState<SavedScript[]>([]);
  const [selectedScript, setSelectedScript] = useState<SavedScript | null>(null);
  const [scriptsLoading, setScriptsLoading] = useState(false);

  useEffect(() => {
    // Initialize session when component mounts
    initializeSession();
  }, []);

  useEffect(() => {
    // Load available scripts when skip mode is enabled
    if (skipMode) {
      loadAvailableScripts();
    }
  }, [skipMode]);

  const initializeSession = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/script/initialize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) throw new Error('Failed to initialize session');
      
      const data = await response.json();
      setSession(prev => ({ ...prev, id: data.session_id }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize session');
    } finally {
      setIsLoading(false);
    }
  };

  const loadAvailableScripts = async () => {
    try {
      setScriptsLoading(true);
      setError(null);
      
      const response = await fetch('http://127.0.0.1:8000/api/scripts/list');
      
      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }
      
      const data = await response.json();
      setAvailableScripts(data.scripts || []);
      
      if (data.scripts && data.scripts.length > 0) {
        console.log(`[SUCCESS] Found ${data.scripts.length} saved scripts for skip mode`);
      }
      
    } catch (err) {
      console.error('Failed to load scripts:', err);
      
      // No fallback scripts - skip mode requires real saved scripts
      setAvailableScripts([]);
      
      // Set a user-friendly error message
      if (err instanceof Error && err.message.includes('fetch')) {
        console.warn('Backend server not available. Complete a script in normal mode first.');
      } else {
        console.warn('Failed to load saved scripts from backend. Complete a script in normal mode first.');
      }
    } finally {
      setScriptsLoading(false);
    }
  };

  const handleToggleSkipMode = (enabled: boolean) => {
    setSkipMode(enabled);
    if (!enabled) {
      setSelectedScript(null);
      setAvailableScripts([]);
    }
  };

  const handleSelectScript = (script: SavedScript) => {
    setSelectedScript(script);
  };

  const handleLoadScript = async () => {
    if (!selectedScript) return;

    try {
      setIsLoading(true);
      setError(null);

      // Try to load from backend first
      let scriptData;
      try {
        const formData = new FormData();
        formData.append('script_id', selectedScript.id);

        const response = await fetch('http://127.0.0.1:8000/api/scripts/load', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) throw new Error('Backend not available');

        const data = await response.json();
        scriptData = data.script;
      } catch (backendErr) {
        // No fallback - skip mode requires backend to load real scripts
        throw new Error('Backend server required to load saved scripts. Please start the backend server and try again.');
      }
      
      // Create a session with the loaded script
      const loadedSession: ScriptSession = {
        id: scriptData.id,
        entryMethod: 'youtube',
        sourceUrl: scriptData.source_url,
        currentScript: scriptData.script_text || '',
        messages: [
          {
            id: Date.now().toString(),
            type: 'system',
            content: `[SUCCESS] Loaded script: ${scriptData.title}`,
            timestamp: new Date()
          },
          {
            id: (Date.now() + 1).toString(),
            type: 'system',
            content: `[STATS] Script contains ${scriptData.word_count.toLocaleString()} words and is ready for video processing.`,
            timestamp: new Date()
          },
          {
            id: (Date.now() + 2).toString(),
            type: 'assistant',
            content: `🚀 Skip mode activated! Your script is loaded and ready. Click "Start Video Editing" to proceed to the video processing phase.`,
            timestamp: new Date()
          }
        ],
        wordCount: scriptData.word_count,
        isFinalized: true
      };

      setSession(loadedSession);
      setCurrentStep('building'); // Show the script in the building interface
      
      console.log(`[SUCCESS] Script loaded successfully: ${scriptData.title} (${scriptData.word_count} words)`);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load script');
    } finally {
      setIsLoading(false);
    }
  };

  const handleMethodSelected = async (method: 'youtube') => {
    try {
      setIsLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('session_id', session.id);
      formData.append('entry_method', method);
      
      const response = await fetch('http://127.0.0.1:8000/api/script/set-entry-method', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Failed to set entry method');
      
      setSession(prev => ({ ...prev, entryMethod: method }));
      setCurrentStep(method);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set entry method');
    } finally {
      setIsLoading(false);
    }
  };

  const handleYouTubeSubmit = async (url: string) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('session_id', session.id);
      formData.append('youtube_url', url);
      formData.append('use_default_prompt', 'true');
      
      const response = await fetch('http://127.0.0.1:8000/api/script/youtube/extract', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Failed to extract YouTube content');
      
            const data = await response.json();
      
      // Update session with URL and automatically generate script
      setSession(prev => ({
        ...prev,
        sourceUrl: url,
        messages: [] // Clear messages since we're not using chat interface
      }));

      // Generate full script automatically after extraction
      try {
        const scriptFormData = new FormData();
        scriptFormData.append('session_id', session.id);

        const scriptResponse = await fetch('http://127.0.0.1:8000/api/script/generate-full-script', {
          method: 'POST',
          body: scriptFormData
        });

        if (!scriptResponse.ok) throw new Error('Failed to generate script');

        const scriptData = await scriptResponse.json();

        setSession(prev => ({
          ...prev,
          currentScript: scriptData.script || '',
          wordCount: scriptData.word_count || 0
        }));

      } catch (scriptErr) {
        console.error('Script auto-generation failed:', scriptErr);
        // Script generation failed, but user can manually generate later
        setError('Script auto-generation failed. Use the "Generate Full Script" button to try again.');
      }

      setCurrentStep('building');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract YouTube content');
    } finally {
      setIsLoading(false);
    }
  };

  // Chat message handling removed - using direct script generation approach

  const handleGenerateFullScript = async () => {
    if (!session.id) return;

    try {
      setIsLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append('session_id', session.id);

      const response = await fetch('http://127.0.0.1:8000/api/script/generate-full-script', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Failed to generate script');

      const data = await response.json();

      setSession(prev => ({
        ...prev,
        currentScript: data.script || '',
        wordCount: data.word_count || 0
      }));

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate script');
    } finally {
      setIsLoading(false);
    }
  };

  const handleScriptUpdate = (newScript: string) => {
    setSession(prev => ({
      ...prev,
      currentScript: newScript,
      wordCount: newScript.split(' ').length
    }));
  };

  const handleSaveDraft = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('session_id', session.id);
      formData.append('title', `Draft - ${new Date().toLocaleDateString()}`);
      
      const response = await fetch('http://127.0.0.1:8000/api/scripts/save', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Failed to save draft');
      
      const data = await response.json();
      console.log('Draft saved:', data.title);
      
      // You might want to show a toast notification here
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save draft');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFinalize = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('session_id', session.id);
      
      const response = await fetch('http://127.0.0.1:8000/api/script/finalize', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Failed to finalize script');
      
      const data = await response.json();
      
      const finalizedSession: ScriptSession = {
        ...session,
        currentScript: data.final_script || session.currentScript,
        wordCount: data.word_count || session.wordCount,
        isFinalized: true
      };
      
      setSession(finalizedSession);
      
      // Auto-save the script for future use in skip mode
      try {
        const saveFormData = new FormData();
        saveFormData.append('session_id', session.id);
        
        await fetch('http://127.0.0.1:8000/api/scripts/save', {
          method: 'POST',
          body: saveFormData
        });
        
        console.log('Script auto-saved for skip mode');
      } catch (saveErr) {
        console.warn('Failed to auto-save script:', saveErr);
        // Don't fail the finalization if auto-save fails
      }
      
      if (onFinalize) {
        onFinalize(finalizedSession);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to finalize script');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBack = () => {
    if (currentStep === 'building') {
      setCurrentStep('entry');
    } else if (currentStep === 'youtube') {
      setCurrentStep('entry');
    } else if (onBack) {
      onBack();
    }
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'entry':
        return (
          <EntryMethodSelector
            onMethodSelected={handleMethodSelected}
            isLoading={isLoading}
          />
        );
      
      case 'youtube':
        return (
          <YouTubeInputPanel
            onUrlSubmitted={handleYouTubeSubmit}
            onBack={handleBack}
            isLoading={isLoading}
            error={error || undefined}
          />
        );
      
      case 'building':
        return (
          <div className="space-y-6">
            {/* Skip Mode Banner */}
            {skipMode && session.isFinalized && (
              <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950 dark:to-blue-950 border border-green-200 dark:border-green-800 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <Zap className="h-5 w-5 text-green-600" />
                    <span className="font-medium text-green-800 dark:text-green-200">Skip Mode Active</span>
                  </div>
                  <div className="flex-1 text-sm text-green-700 dark:text-green-300">
                    Script loaded successfully - {session.wordCount.toLocaleString()} words - Ready for video processing
                  </div>
                  <Button
                    onClick={handleFinalize}
                    size="sm"
                    className="bg-green-600 hover:bg-green-700 text-white"
                  >
                    <Play className="h-4 w-4 mr-1" />
                    Process Video
                  </Button>
                </div>
              </div>
            )}

            {/* Single Column Layout - Script Panel Only */}
            <div className="max-w-5xl mx-auto">
              <div className="space-y-6">
                {/* Script Panel - Centered with proper height */}
                <ScriptPanel
                  script={session.currentScript}
                  wordCount={session.wordCount}
                  bulletPoints={[]}
                  className="h-[600px] min-h-[500px] max-h-[75vh]"
                  sessionId={session.id}
                  onScriptUpdate={handleScriptUpdate}
                  isModificationMode={!skipMode}
                />
                
                {/* Progress Tracker */}
                <div className="w-full">
                  <SimpleWordCounter
                    currentWords={session.wordCount}
                    targetWords={2000}
                  />
                </div>
                
                {/* Action Buttons - Centered under script panel */}
                <div className="flex justify-center">
                  <div className="flex flex-col items-center space-y-3 w-full max-w-md">
                    {!session.currentScript && (
                      <Button
                        onClick={handleGenerateFullScript}
                        disabled={isLoading}
                        className="w-full flex items-center gap-2 bg-blue-600 hover:bg-blue-700"
                        size="lg"
                      >
                        <FileText className="h-4 w-4" />
                        Generate Full Script
                      </Button>
                    )}
                    
                    <Button
                      onClick={handleFinalize}
                      disabled={isLoading || session.wordCount < 500}
                      className="w-full flex items-center gap-2"
                      size="lg"
                    >
                      <Play className="h-4 w-4" />
                      Start Video Editing
                    </Button>
                    
                    <div className="grid grid-cols-2 gap-2 w-full">
                      <Button
                        variant="outline"
                        disabled={isLoading}
                        className="flex items-center gap-2"
                        onClick={handleSaveDraft}
                      >
                        <Save className="h-4 w-4" />
                        Save Draft
                      </Button>
                      <Button
                        variant="outline"
                        disabled={isLoading}
                        className="flex items-center gap-2"
                      >
                        <Download className="h-4 w-4" />
                        Export
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>



            {/* Script analysis removed - using full script approach */}
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className={`w-full max-w-7xl mx-auto p-6 ${className}`}>
      {/* Development Mode Toggle */}
      <div className="mb-6">
        <DevModeToggle
          skipMode={skipMode}
          onToggleSkipMode={handleToggleSkipMode}
          availableScripts={availableScripts}
          selectedScript={selectedScript}
          onSelectScript={handleSelectScript}
          onLoadScript={handleLoadScript}
          isLoading={isLoading || scriptsLoading}
        />
      </div>

      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        {currentStep !== 'entry' && (
          <Button
            variant="ghost"
            onClick={handleBack}
            disabled={isLoading}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        )}
        
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-foreground">
            {skipMode ? 'Skip Script Phase' : ''}
            {!skipMode && currentStep === 'entry' && 'YouTube Script Builder'}
            {!skipMode && currentStep === 'youtube' && 'YouTube Video Input'}
            {!skipMode && currentStep === 'building' && 'Interactive Script Builder'}
          </h1>
          
          {session.id && (
            <p className="text-sm text-muted-foreground mt-1">
              Session: {session.id.slice(0, 8)}
              {session.entryMethod && ` - ${session.entryMethod.charAt(0).toUpperCase() + session.entryMethod.slice(1)} Mode`}
              {session.wordCount > 0 && ` - ${session.wordCount} words`}
              {skipMode && ` - Skip Mode Active`}
            </p>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Main Content */}
      {renderCurrentStep()}
    </div>
  );
} 