'use client';

import { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { ArrowLeft, Play, Save, Download, List, Target, Zap } from 'lucide-react';

// Entry components
import { EntryMethodSelector } from './entry/EntryMethodSelector';
import { YouTubeInputPanel } from './entry/YouTubeInputPanel';

// Interactive components
import { ChatInterface } from './interactive/ChatInterface';

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

export interface BulletPoint {
  id: string;
  title: string;
  description: string;
  target_length: number;
  importance: string;
  order: number;
  key_points: string[];
  emotional_tone: string;
  engagement_strategy: string;
}

export interface ScriptSession {
  id: string;
  entryMethod: 'youtube' | null;
  sourceUrl?: string;
  currentScript: string;
  bulletPoints: BulletPoint[];
  messages: ChatMessage[];
  wordCount: number;
  targetWordCount: number;
  sectionsCompleted: number;
  totalSections: number;
  analysis?: any;
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
    bulletPoints: [],
    messages: [],
    wordCount: 0,
    targetWordCount: 2000,
    sectionsCompleted: 0,
    totalSections: 5,
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
        console.log(`✅ Found ${data.scripts.length} saved scripts for skip mode`);
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
        bulletPoints: scriptData.bullet_points || [],
        messages: [
          {
            id: Date.now().toString(),
            type: 'system',
            content: `✅ Loaded script: ${scriptData.title}`,
            timestamp: new Date()
          },
          {
            id: (Date.now() + 1).toString(),
            type: 'system',
            content: `📊 Script contains ${scriptData.word_count.toLocaleString()} words and is ready for video processing.`,
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
        targetWordCount: scriptData.word_count,
        sectionsCompleted: scriptData.bullet_points?.length || 0,
        totalSections: scriptData.bullet_points?.length || 0,
        isFinalized: true
      };

      setSession(loadedSession);
      setCurrentStep('building'); // Show the script in the building interface
      
      console.log(`✅ Script loaded successfully: ${scriptData.title} (${scriptData.word_count} words)`);
      
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
      
      // Add system message for transcript extraction
      const systemMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'system',
        content: `Successfully extracted content from YouTube video: ${data.video_title}. Transcript length: ${data.transcript_length} characters.`,
        timestamp: new Date()
      };

      setSession(prev => ({
        ...prev,
        sourceUrl: url,
        messages: [systemMessage]
      }));

      // Automatically generate bullet points after transcript extraction
      const bulletFormData = new FormData();
      bulletFormData.append('session_id', session.id);
      
      const bulletResponse = await fetch('http://127.0.0.1:8000/api/script/generate-bullet-points', {
        method: 'POST',
        body: bulletFormData
      });
      
      if (!bulletResponse.ok) throw new Error('Failed to generate bullet points');
      
      const bulletData = await bulletResponse.json();
      
      // Add AI message for bullet points generation
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `Generated ${bulletData.bullet_points?.length || 0} bullet points for your script. You can now start building sections interactively by typing commands like "/generate section 1" or asking me to refine specific sections.`,
        timestamp: new Date()
      };

      setSession(prev => ({
        ...prev,
        bulletPoints: bulletData.bullet_points || [],
        messages: [...prev.messages, aiMessage],
        totalSections: bulletData.bullet_points?.length || 5
      }));

      setCurrentStep('building');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract YouTube content');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (message: string) => {
    if (!message.trim() || isLoading) return;

    // Add user message immediately
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date(),
      status: 'sending'
    };

    setSession(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage]
    }));

    try {
      setIsLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append('session_id', session.id);
      formData.append('message', message);
      formData.append('message_type', 'user');

      const response = await fetch('http://127.0.0.1:8000/api/script/chat', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Failed to send message');

      const data = await response.json();

      // Add AI response message
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant' as const,
        content: data.ai_response?.content || 'Response received',
        timestamp: new Date()
      };

      // Update session with response data
      setSession(prev => ({
        ...prev,
        messages: prev.messages.map(msg => {
          if (msg.id === userMessage.id) {
            return { ...msg, status: 'sent' as const };
          }
          return msg;
        }).concat([aiMessage]),
        // Update script content from backend
        currentScript: data.script_data?.current_script || prev.currentScript,
        wordCount: data.script_data?.word_count || prev.wordCount,
        sectionsCompleted: data.script_data?.sections_completed || prev.sectionsCompleted,
        totalSections: data.script_data?.total_sections || prev.totalSections
      }));

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      
      // Mark user message as error
      setSession(prev => ({
        ...prev,
        messages: prev.messages.map(msg => {
          if (msg.id === userMessage.id) {
            return { ...msg, status: 'error' as const };
          }
          return msg;
        })
      }));
    } finally {
      setIsLoading(false);
    }
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
                    Script loaded successfully • {session.wordCount.toLocaleString()} words • Ready for video processing
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

            {/* Main 2-Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[600px]">
              {/* Left Column - Script Panel */}
              <div className="flex flex-col space-y-4">
                <ScriptPanel
                  script={session.currentScript}
                  wordCount={session.wordCount}
                  bulletPoints={session.bulletPoints}
                  className="flex-1"
                />
                
                {/* Progress Tracker - Full width under left column */}
                <SimpleWordCounter
                  currentWords={session.wordCount}
                  targetWords={session.targetWordCount}
                />
              </div>
              
              {/* Right Column - Chat Interface + Action Buttons */}
              <div className="flex flex-col space-y-4">
                <div className="flex-1">
                  <ChatInterface
                    sessionId={session.id}
                    messages={session.messages}
                    onSendMessage={skipMode && session.isFinalized ? () => {} : handleSendMessage}
                    isLoading={isLoading}
                    wordCount={session.wordCount}
                    targetWordCount={session.targetWordCount}
                  />
                </div>
                
                {/* Action Buttons - Under right column only */}
                <div className="space-y-3">
                  <Button
                    onClick={handleFinalize}
                    disabled={isLoading || session.wordCount < 500}
                    className="w-full flex items-center gap-2"
                    size="lg"
                  >
                    <Play className="h-4 w-4" />
                    Start Video Editing
                  </Button>
                  
                  <div className="grid grid-cols-2 gap-2">
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



            {session.analysis && (
              <ScriptAnalysisDisplay
                analysis={session.analysis}
                onRefineSection={(sectionName) => {
                  handleSendMessage(`/refine ${sectionName} Please improve this section`);
                }}
                onImplementSuggestion={(suggestionId) => {
                  handleSendMessage(`Please implement suggestion ${suggestionId}`);
                }}
                isLoading={isLoading}
              />
            )}
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
              {session.entryMethod && ` • ${session.entryMethod.charAt(0).toUpperCase() + session.entryMethod.slice(1)} Mode`}
              {session.wordCount > 0 && ` • ${session.wordCount} words`}
              {skipMode && ` • Skip Mode Active`}
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