'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Youtube, ExternalLink, AlertCircle, CheckCircle, Clock, Eye } from 'lucide-react';

interface YouTubeInputPanelProps {
  onUrlSubmitted: (url: string) => void;
  onBack: () => void;
  isLoading?: boolean;
  error?: string;
}

interface VideoInfo {
  title: string;
  channel: string;
  duration: string;
  views: string;
  thumbnail: string;
}

export function YouTubeInputPanel({ onUrlSubmitted, onBack, isLoading = false, error }: YouTubeInputPanelProps) {
  const [url, setUrl] = useState('');
  const [isValidUrl, setIsValidUrl] = useState<boolean | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [extractionStep, setExtractionStep] = useState<'input' | 'preview' | 'extracting'>('input');

  const validateYouTubeUrl = (url: string): boolean => {
    const patterns = [
      /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]{11}/,
      /^https?:\/\/(www\.)?youtu\.be\/[\w-]{11}/,
      /^https?:\/\/(www\.)?youtube\.com\/embed\/[\w-]{11}/
    ];
    return patterns.some(pattern => pattern.test(url));
  };

  const handleUrlChange = async (value: string) => {
    setUrl(value);
    const isValid = value ? validateYouTubeUrl(value) : null;
    setIsValidUrl(isValid);
    
    // Fetch real video info for valid URLs
    if (isValid) {
      try {
        const formData = new FormData();
        formData.append('youtube_url', value);
        
        const response = await fetch('http://127.0.0.1:8000/api/youtube/info', {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          const data = await response.json();
          setVideoInfo({
            title: data.title || "Video Title",
            channel: data.channel || "YouTube Channel",
            duration: data.duration || "Unknown Duration",
            views: data.views || "Unknown Views",
            thumbnail: data.thumbnail || "https://via.placeholder.com/320x180"
          });
        } else {
          // Fallback to basic info if API fails
          setVideoInfo({
            title: "Video information will be extracted during processing",
            channel: "YouTube",
            duration: "Unknown",
            views: "Unknown",
            thumbnail: "https://via.placeholder.com/320x180"
          });
        }
      } catch (error) {
        console.warn('Failed to fetch video info:', error);
        // Fallback to basic info if API fails
        setVideoInfo({
          title: "Video information will be extracted during processing",
          channel: "YouTube", 
          duration: "Unknown",
          views: "Unknown",
          thumbnail: "https://via.placeholder.com/320x180"
        });
      }
    } else {
      setVideoInfo(null);
    }
  };

  const handleSubmit = () => {
    if (isValidUrl && url) {
      setExtractionStep('extracting');
      onUrlSubmitted(url);
    }
  };

  const handlePreview = () => {
    if (isValidUrl) {
      setExtractionStep('preview');
    }
  };

  const renderUrlInput = () => (
    <div className="space-y-4">
      <div className="space-y-2">
        <label htmlFor="youtube-url" className="text-sm font-medium text-foreground">
          YouTube Video URL
        </label>
        <div className="relative">
          <input
            id="youtube-url"
            type="url"
            value={url}
            onChange={(e) => handleUrlChange(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className={`w-full px-4 py-3 pr-12 rounded-lg border transition-colors ${
              isValidUrl === false 
                ? 'border-red-300 focus:border-red-500 focus:ring-red-500' 
                : isValidUrl === true
                ? 'border-green-300 focus:border-green-500 focus:ring-green-500'
                : 'border-gray-300 focus:border-primary focus:ring-primary'
            } focus:outline-none focus:ring-2 focus:ring-opacity-50`}
            disabled={isLoading}
          />
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            {isValidUrl === true && (
              <CheckCircle className="h-5 w-5 text-green-500" />
            )}
            {isValidUrl === false && (
              <AlertCircle className="h-5 w-5 text-red-500" />
            )}
          </div>
        </div>
        
        {/* URL validation feedback */}
        {isValidUrl === false && (
          <div className="flex items-center gap-2 text-sm text-red-600">
            <AlertCircle className="h-4 w-4" />
            <span>Please enter a valid YouTube URL</span>
          </div>
        )}
        
        {isValidUrl === true && (
          <div className="flex items-center gap-2 text-sm text-green-600">
            <CheckCircle className="h-4 w-4" />
            <span>Valid YouTube URL detected</span>
          </div>
        )}
      </div>

      {/* Video Info Preview */}
      {videoInfo && isValidUrl && (
        <Card className="border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800">
          <CardContent className="p-4">
            <div className="flex gap-4">
              <img 
                src={videoInfo.thumbnail} 
                alt="Video thumbnail"
                className="w-24 h-14 rounded object-cover flex-shrink-0"
              />
              <div className="flex-1 min-w-0">
                <h4 className="font-medium text-foreground truncate">{videoInfo.title}</h4>
                <p className="text-sm text-muted-foreground">{videoInfo.channel}</p>
                <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {videoInfo.duration}
                  </div>
                  <div className="flex items-center gap-1">
                    <Eye className="h-3 w-3" />
                    {videoInfo.views}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3 pt-2">
        <Button
          variant="outline"
          onClick={onBack}
          disabled={isLoading}
          className="flex-1"
        >
          Back to Methods
        </Button>
        
        {isValidUrl && videoInfo && (
          <Button
            variant="secondary"
            onClick={handlePreview}
            disabled={isLoading}
            className="flex items-center gap-2"
          >
            <Eye className="h-4 w-4" />
            Preview
          </Button>
        )}
        
        <Button
          onClick={handleSubmit}
          disabled={!isValidUrl || isLoading}
          className="flex-1 flex items-center gap-2"
        >
          <Youtube className="h-4 w-4" />
          Extract & Start Building
        </Button>
      </div>
    </div>
  );

  const renderPreview = () => (
    <div className="space-y-6">
      {/* Video Preview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Youtube className="h-5 w-5" />
              Video Preview
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => window.open(url, '_blank')}
              className="flex items-center gap-1"
            >
              <ExternalLink className="h-3 w-3" />
              Open on YouTube
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {videoInfo && (
            <div className="space-y-4">
              <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
                <img 
                  src={videoInfo.thumbnail} 
                  alt="Video thumbnail"
                  className="max-w-full max-h-full rounded"
                />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-foreground">{videoInfo.title}</h3>
                <p className="text-muted-foreground">{videoInfo.channel}</p>
                <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {videoInfo.duration}
                  </div>
                  <div className="flex items-center gap-1">
                    <Eye className="h-4 w-4" />
                    {videoInfo.views}
                  </div>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* What will happen next */}
      <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
        <CardHeader>
          <CardTitle className="text-blue-900 dark:text-blue-100">What happens next?</CardTitle>
        </CardHeader>
        <CardContent className="text-blue-800 dark:text-blue-200">
          <ol className="list-decimal list-inside space-y-2">
            <li>We'll extract the video's transcript using AI</li>
                              <li>Generate a complete, flowing script from the content</li>
            <li>Open the interactive script builder</li>
                          <li>You can highlight text to modify, adjust length, and customize the output</li>
            <li>When ready, proceed to video editing</li>
          </ol>
          <p className="mt-3 text-sm">
            Estimated time: 5-10 minutes depending on video length and your customizations.
          </p>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button
          variant="outline"
          onClick={() => setExtractionStep('input')}
          disabled={isLoading}
          className="flex-1"
        >
          Back to URL
        </Button>
        <Button
          onClick={handleSubmit}
          disabled={isLoading}
          className="flex-1 flex items-center gap-2"
        >
          <Youtube className="h-4 w-4" />
          Start Extraction
        </Button>
      </div>
    </div>
  );

  const renderExtracting = () => (
    <div className="text-center space-y-6">
      <div className="flex justify-center">
        <div className="relative">
          <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
            <Youtube className="h-10 w-10 text-primary" />
          </div>
          <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
        </div>
      </div>
      
      <div className="space-y-2">
        <h3 className="text-xl font-semibold">Extracting Video Content</h3>
        <p className="text-muted-foreground">
          We're analyzing your video and preparing the transcript...
        </p>
      </div>
      
      {error && (
        <Card className="border-red-200 bg-red-50 dark:bg-red-950 dark:border-red-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <AlertCircle className="h-4 w-4" />
              <span className="font-medium">Extraction Error</span>
            </div>
            <p className="text-red-600 dark:text-red-400 mt-1">{error}</p>
            <Button 
              variant="outline" 
              size="sm" 
              className="mt-3"
              onClick={() => setExtractionStep('input')}
            >
              Try Again
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      <div className="text-center space-y-4 mb-8">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Youtube className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-foreground">YouTube Video Input</h2>
        </div>
        <p className="text-muted-foreground">
          Paste a YouTube URL to extract and analyze the video content for script generation
        </p>
      </div>

      <Card>
        <CardContent className="p-6">
          {extractionStep === 'input' && renderUrlInput()}
          {extractionStep === 'preview' && renderPreview()}
          {extractionStep === 'extracting' && renderExtracting()}
        </CardContent>
      </Card>
    </div>
  );
} 