'use client';

import { useState, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Upload, FileText, AlertCircle, CheckCircle, Eye, X } from 'lucide-react';

interface ScriptUploadPanelProps {
  onScriptUploaded: (content: string, fileName: string) => void;
  onBack: () => void;
  isLoading?: boolean;
  error?: string;
}

interface ScriptAnalysis {
  wordCount: number;
  estimatedReadingTime: string;
  sections: number;
  quality: 'good' | 'fair' | 'needs-work';
  suggestions: string[];
}

export function ScriptUploadPanel({ onScriptUploaded, onBack, isLoading = false, error }: ScriptUploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [scriptContent, setScriptContent] = useState<string>('');
  const [analysis, setAnalysis] = useState<ScriptAnalysis | null>(null);
  const [step, setStep] = useState<'upload' | 'preview' | 'processing'>('upload');
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    const allowedTypes = ['text/plain', 'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!allowedTypes.includes(file.type) && !file.name.endsWith('.txt')) {
      return false;
    }
    
    if (file.size > maxSize) {
      return false;
    }
    
    return true;
  };

  const analyzeScript = (content: string): ScriptAnalysis => {
    const wordCount = content.split(/\s+/).filter(word => word.length > 0).length;
    const readingWPM = 180;
    const estimatedMinutes = Math.ceil(wordCount / readingWPM);
    const sections = content.split(/\n\s*\n/).filter(section => section.trim().length > 0).length;
    
    let quality: 'good' | 'fair' | 'needs-work' = 'good';
    const suggestions: string[] = [];
    
    if (wordCount < 500) {
      quality = 'needs-work';
      suggestions.push('Script is quite short - consider expanding sections');
    } else if (wordCount < 1000) {
      quality = 'fair';
      suggestions.push('Script could benefit from more detailed content');
    }
    
    if (sections < 3) {
      quality = quality === 'good' ? 'fair' : 'needs-work';
      suggestions.push('Consider breaking into more distinct sections');
    }
    
    if (content.toLowerCase().includes('um') || content.toLowerCase().includes('uh')) {
      suggestions.push('Remove filler words like "um" and "uh"');
    }
    
    return {
      wordCount,
      estimatedReadingTime: `${estimatedMinutes} min`,
      sections,
      quality,
      suggestions
    };
  };

  const processFile = async (file: File) => {
    try {
      const content = await file.text();
      setScriptContent(content);
      setAnalysis(analyzeScript(content));
      setStep('preview');
    } catch (err) {
      console.error('Error reading file:', err);
    }
  };

  const handleFileSelect = useCallback((file: File) => {
    if (!validateFile(file)) {
      return;
    }
    
    setUploadedFile(file);
    processFile(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleSubmit = () => {
    if (scriptContent && uploadedFile) {
      setStep('processing');
      onScriptUploaded(scriptContent, uploadedFile.name);
    }
  };

  const resetUpload = () => {
    setUploadedFile(null);
    setScriptContent('');
    setAnalysis(null);
    setStep('upload');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const renderUploadArea = () => (
    <div className="space-y-6">
      {/* Drag and Drop Area */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging 
            ? 'border-primary bg-primary/5' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.pdf,.doc,.docx"
          onChange={handleFileInputChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isLoading}
        />
        
        <div className="space-y-4">
          <div className="flex justify-center">
            <div className={`p-4 rounded-full ${
              isDragging ? 'bg-primary text-primary-foreground' : 'bg-gray-100 text-gray-600'
            }`}>
              <Upload className="h-8 w-8" />
            </div>
          </div>
          
          <div>
            <p className="text-lg font-medium text-foreground">
              {isDragging ? 'Drop your script here' : 'Upload your script'}
            </p>
            <p className="text-muted-foreground mt-1">
              Drag and drop or click to select
            </p>
          </div>
          
          <div className="text-sm text-muted-foreground">
            Supported formats: TXT, PDF, DOC, DOCX (max 10MB)
          </div>
        </div>
      </div>

      {/* Manual Text Input Option */}
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <div className="flex-1 border-t border-gray-300"></div>
          <span className="text-sm text-muted-foreground">or paste text directly</span>
          <div className="flex-1 border-t border-gray-300"></div>
        </div>
        
        <div className="space-y-2">
          <label htmlFor="script-text" className="text-sm font-medium text-foreground">
            Script Content
          </label>
          <textarea
            id="script-text"
            value={scriptContent}
            onChange={(e) => {
              setScriptContent(e.target.value);
              if (e.target.value.trim()) {
                setAnalysis(analyzeScript(e.target.value));
                setStep('preview');
              }
            }}
            placeholder="Paste your script content here..."
            className="w-full h-32 px-4 py-3 rounded-lg border border-gray-300 focus:border-primary focus:ring-primary focus:outline-none focus:ring-2 focus:ring-opacity-50 resize-none"
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Back Button */}
      <div className="flex justify-start">
        <Button
          variant="outline"
          onClick={onBack}
          disabled={isLoading}
        >
          Back to Methods
        </Button>
      </div>
    </div>
  );

  const renderPreview = () => (
    <div className="space-y-6">
      {/* File Info */}
      <Card className="border-green-200 bg-green-50 dark:bg-green-950 dark:border-green-800">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-green-600" />
              <div>
                <p className="font-medium text-foreground">
                  {uploadedFile?.name || 'Pasted Content'}
                </p>
                <p className="text-sm text-muted-foreground">
                  {uploadedFile ? `${(uploadedFile.size / 1024).toFixed(1)} KB` : 'Direct input'}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={resetUpload}
              className="text-red-600 hover:text-red-700"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Script Analysis */}
      {analysis && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Script Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-foreground">{analysis.wordCount}</div>
                <div className="text-sm text-muted-foreground">Words</div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-foreground">{analysis.estimatedReadingTime}</div>
                <div className="text-sm text-muted-foreground">Reading Time</div>
              </div>
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-foreground">{analysis.sections}</div>
                <div className="text-sm text-muted-foreground">Sections</div>
              </div>
            </div>

            {/* Quality Assessment */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Quality Assessment:</span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  analysis.quality === 'good' 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                    : analysis.quality === 'fair'
                    ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                    : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                }`}>
                  {analysis.quality === 'good' ? 'Good' : analysis.quality === 'fair' ? 'Fair' : 'Needs Work'}
                </span>
              </div>
              
              {analysis.suggestions.length > 0 && (
                <div className="space-y-1">
                  <p className="text-sm font-medium text-muted-foreground">Suggestions:</p>
                  <ul className="text-sm space-y-1">
                    {analysis.suggestions.map((suggestion, index) => (
                      <li key={index} className="flex items-start gap-2 text-muted-foreground">
                        <span className="text-yellow-500 mt-1">•</span>
                        {suggestion}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Content Preview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Content Preview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 max-h-60 overflow-y-auto">
            <pre className="text-sm whitespace-pre-wrap text-foreground">
              {scriptContent.substring(0, 500)}
              {scriptContent.length > 500 && '...'}
            </pre>
          </div>
        </CardContent>
      </Card>

      {/* What happens next */}
      <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
        <CardHeader>
          <CardTitle className="text-blue-900 dark:text-blue-100">What happens next?</CardTitle>
        </CardHeader>
        <CardContent className="text-blue-800 dark:text-blue-200">
          <ol className="list-decimal list-inside space-y-2">
            <li>AI will analyze your script structure and content</li>
            <li>Generate improvement suggestions and optimization recommendations</li>
            <li>Open the interactive script builder</li>
            <li>You can refine sections, improve readability, and enhance engagement</li>
            <li>When ready, proceed to video editing</li>
          </ol>
          <p className="mt-3 text-sm">
            Estimated time: 2-5 minutes depending on script length and your refinements.
          </p>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button
          variant="outline"
          onClick={resetUpload}
          disabled={isLoading}
          className="flex-1"
        >
          Upload Different File
        </Button>
        <Button
          onClick={handleSubmit}
          disabled={isLoading}
          className="flex-1 flex items-center gap-2"
        >
          <FileText className="h-4 w-4" />
          Start Analysis
        </Button>
      </div>
    </div>
  );

  const renderProcessing = () => (
    <div className="text-center space-y-6">
      <div className="flex justify-center">
        <div className="relative">
          <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
            <FileText className="h-10 w-10 text-primary" />
          </div>
          <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
        </div>
      </div>
      
      <div className="space-y-2">
        <h3 className="text-xl font-semibold">Analyzing Your Script</h3>
        <p className="text-muted-foreground">
          We're analyzing your content and preparing optimization suggestions...
        </p>
      </div>
      
      {error && (
        <Card className="border-red-200 bg-red-50 dark:bg-red-950 dark:border-red-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <AlertCircle className="h-4 w-4" />
              <span className="font-medium">Analysis Error</span>
            </div>
            <p className="text-red-600 dark:text-red-400 mt-1">{error}</p>
            <Button 
              variant="outline" 
              size="sm" 
              className="mt-3"
              onClick={() => setStep('preview')}
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
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold text-foreground">
          Upload Your Script
        </h2>
        <p className="text-muted-foreground">
          Upload an existing script for AI-powered analysis and refinement
        </p>
      </div>

      <Card>
        <CardContent className="p-6">
          {step === 'upload' && renderUploadArea()}
          {step === 'preview' && renderPreview()}
          {step === 'processing' && renderProcessing()}
        </CardContent>
      </Card>
    </div>
  );
} 