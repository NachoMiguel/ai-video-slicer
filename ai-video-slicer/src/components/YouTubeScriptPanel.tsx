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

  const handleGenerateScript = async () => {
    if (!youtubeUrl) return;

    setIsGenerating(true);
    try {
      const formData = new FormData();
      formData.append('youtube_url', youtubeUrl);
      formData.append('use_default_prompt', String(useDefaultPrompt));
      if (!useDefaultPrompt && customPrompt) {
        formData.append('custom_prompt', customPrompt);
      }

      const response = await fetch('http://127.0.0.1:8000/api/generate-script', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to generate script');
      }

      const data = await response.json();
      setGeneratedScript(data.script);
      onScriptGenerated(data.script);
    } catch (error) {
      console.error('Error generating script:', error);
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
          {isGenerating ? 'Transcribing & Rewriting...' : 'Transcribe & Rewrite'}
        </Button>
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