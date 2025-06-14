'use client';

import { useState, ChangeEvent } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';

interface YouTubeScriptPanelProps {
  onScriptGenerated: (script: string) => void;
  isProcessing: boolean;
}

export function YouTubeScriptPanel({ onScriptGenerated, isProcessing }: YouTubeScriptPanelProps) {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [useDefaultPrompt, setUseDefaultPrompt] = useState(true);
  const [customPrompt, setCustomPrompt] = useState('');
  const [generatedScript, setGeneratedScript] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>('');

  const handleGenerateScript = async () => {
    if (!youtubeUrl) return;

    setIsGenerating(true);
    setError(null);
    setGeneratedScript(null);
    
    try {
      setProgress('🔗 Connecting to YouTube...');
      
      const formData = new FormData();
      formData.append('youtube_url', youtubeUrl);
      formData.append('use_default_prompt', String(useDefaultPrompt));
      if (!useDefaultPrompt && customPrompt) {
        formData.append('custom_prompt', customPrompt);
      }

      setProgress('📥 Downloading audio...');
      
      const response = await fetch('http://127.0.0.1:8000/api/generate-script', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        let errorMessage = errorData.detail || 'Failed to generate script';
        
        // Provide more specific error messages
        if (response.status === 400) {
          if (errorMessage.includes('Could not download audio')) {
            errorMessage = 'Unable to download audio from this YouTube video. Please check the URL and try again.';
          } else if (errorMessage.includes('No prompt provided')) {
            errorMessage = 'Please provide a custom prompt or enable the default prompt.';
          } else if (errorMessage.includes('OPENAI_API_KEY')) {
            errorMessage = 'OpenAI API key is missing. Please check your backend configuration.';
          }
        } else if (response.status === 500) {
          if (errorMessage.includes('OpenAI client not available')) {
            errorMessage = 'OpenAI service is not available. Please check your API key configuration.';
          }
        }
        
        throw new Error(errorMessage);
      }

      setProgress('🎙️ Transcribing audio...');
      
      const data = await response.json();
      
      setProgress('🤖 Generating script...');
      
      if (!data.script) {
        throw new Error('No script was generated');
      }
      
      setProgress('✅ Script generated successfully!');
      setGeneratedScript(data.script);
      onScriptGenerated(data.script);
      
      // Clear progress after a short delay
      setTimeout(() => setProgress(''), 2000);
      
    } catch (error) {
      console.error('Error generating script:', error);
      const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred';
      setError(errorMessage);
      setProgress('');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <label htmlFor="youtube-url" className="block text-sm font-medium mb-2">YouTube URL</label>
          <input
            id="youtube-url"
            type="url"
            placeholder="https://www.youtube.com/watch?v=..."
            value={youtubeUrl}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setYoutubeUrl(e.target.value)}
            disabled={isGenerating || isProcessing}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="flex items-center space-x-2">
          <input
            id="default-prompt"
            type="checkbox"
            checked={useDefaultPrompt}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setUseDefaultPrompt(e.target.checked)}
            disabled={isGenerating || isProcessing}
            className="rounded"
          />
          <label htmlFor="default-prompt" className="text-sm font-medium">Use default prompt</label>
        </div>

        {!useDefaultPrompt && (
          <div>
            <label htmlFor="custom-prompt" className="block text-sm font-medium mb-2">Custom Prompt</label>
            <textarea
              id="custom-prompt"
              placeholder="Enter your custom prompt for script generation..."
              value={customPrompt}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setCustomPrompt(e.target.value)}
              disabled={isGenerating || isProcessing}
              rows={4}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        )}

        <Button
          onClick={handleGenerateScript}
          disabled={!youtubeUrl || isGenerating || isProcessing || (!useDefaultPrompt && !customPrompt)}
          className="w-full"
        >
          {isGenerating ? (
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Processing...</span>
            </div>
          ) : (
            'Transcribe & Rewrite'
          )}
        </Button>

        {/* Progress indicator */}
        {progress && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
            <div className="flex items-center space-x-2">
              <div className="animate-pulse h-2 w-2 bg-blue-500 rounded-full"></div>
              <span className="text-sm text-blue-700">{progress}</span>
            </div>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center space-x-2">
              <span className="text-red-500">⚠️</span>
              <span className="text-sm text-red-700">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-xs text-red-600 hover:text-red-800 underline"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>

      {generatedScript && (
        <div className="space-y-2">
          <label className="block text-sm font-medium mb-2">Generated Script</label>
          <div className="p-4 border rounded-lg bg-gray-50 max-h-96 overflow-y-auto">
            <pre className="whitespace-pre-wrap text-sm">{generatedScript}</pre>
          </div>
        </div>
      )}
    </div>
  );
} 