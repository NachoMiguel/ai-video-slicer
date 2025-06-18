'use client';

import { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { ArrowLeft, Play, Save, Download } from 'lucide-react';

// Entry components
import { EntryMethodSelector } from './entry/EntryMethodSelector';
import { YouTubeInputPanel } from './entry/YouTubeInputPanel';

// Interactive components
import { ChatInterface } from './interactive/ChatInterface';

// Shared components
import { ScriptAnalysisDisplay } from './shared/ScriptAnalysisDisplay';
import { WordCountTracker } from './shared/WordCountTracker';

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
  bulletPoints: string[];
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

  useEffect(() => {
    // Initialize session when component mounts
    initializeSession();
  }, []);

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
      setSession(prev => ({
        ...prev,
        sourceUrl: url,
        bulletPoints: data.bulletPoints || [],
        analysis: data.analysis
      }));
      
      // Add system message
      const systemMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'system',
        content: `Successfully extracted content from YouTube video: ${data.video_title}. Transcript length: ${data.transcript_length} characters.`,
        timestamp: new Date()
      };
      
      setSession(prev => ({
        ...prev,
        messages: [systemMessage]
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

      // Update messages with response
      setSession(prev => ({
        ...prev,
        messages: prev.messages.map(msg => {
          if (msg.id === userMessage.id) {
            return { ...msg, status: 'sent' as const };
          }
          return msg;
        }).concat([{
          id: (Date.now() + 1).toString(),
          type: 'assistant' as const,
          content: data.response || 'Response received',
          timestamp: new Date()
        }]),
        currentScript: data.currentScript || prev.currentScript,
        wordCount: data.wordCount || prev.wordCount,
        bulletPoints: data.bulletPoints || prev.bulletPoints
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
      
      const finalizedSession = { ...session, isFinalized: true };
      setSession(finalizedSession);
      
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
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Chat Interface */}
            <div className="lg:col-span-2">
              <ChatInterface
                sessionId={session.id}
                messages={session.messages}
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                wordCount={session.wordCount}
                targetWordCount={session.targetWordCount}
              />
            </div>
            
            {/* Right Column - Analysis & Progress */}
            <div className="space-y-6">
              <WordCountTracker
                currentWords={session.wordCount}
                targetWords={session.targetWordCount}
                sectionsCompleted={session.sectionsCompleted}
                totalSections={session.totalSections}
              />
              
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
              
              {/* Action Buttons */}
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
        );
      
      default:
        return null;
    }
  };

  return (
    <div className={`w-full max-w-7xl mx-auto p-6 ${className}`}>
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
            {currentStep === 'entry' && 'YouTube Script Builder'}
            {currentStep === 'youtube' && 'YouTube Video Input'}
            {currentStep === 'building' && 'Interactive Script Builder'}
          </h1>
          
          {session.id && (
            <p className="text-sm text-muted-foreground mt-1">
              Session: {session.id.slice(0, 8)}
              {session.entryMethod && ` • ${session.entryMethod.charAt(0).toUpperCase() + session.entryMethod.slice(1)} Mode`}
              {session.wordCount > 0 && ` • ${session.wordCount} words`}
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