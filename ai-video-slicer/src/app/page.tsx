'use client'

import { useState } from 'react'
import { UploadZone } from '../components/UploadZone'
import { ProcessingStatus } from '../components/ProcessingStatus'
import { PromptInput } from '../components/PromptInput'
import { ScriptPanel } from '../components/ScriptPanel'
import { VideoPreview } from '../components/VideoPreview'
import { YouTubeScriptPanel } from '../components/YouTubeScriptPanel'
import { ThemeToggle } from '../components/theme-toggle'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';

export default function Home() {
  const [videos, setVideos] = useState<File[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [showPrompt, setShowPrompt] = useState(false)
  const [processId, setProcessId] = useState<string | null>(null)
  const [generatedScript, setGeneratedScript] = useState<string | null>(null)
  const [result, setResult] = useState<{
    script: string;
    videoUrl: string;
  } | null>(null)

  const handleFilesAccepted = (files: File[]) => {
    setVideos(files)
  }

  const handleScriptGenerated = (script: string) => {
    setGeneratedScript(script)
  }

  const handleGenerate = async (prompt: string) => {
    if (videos.length < 2) return

    setIsProcessing(true)
    setProcessId(null)
    setResult(null)

    try {
      const formData = new FormData()
      videos.forEach(video => formData.append('videos', video))
      formData.append('prompt', prompt)
      formData.append('use_base_prompt', String(prompt === 'Use base prompt'))

      const response = await fetch('http://127.0.0.1:8000/api/process', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Failed to process videos')
      }

      const data = await response.json()
      setProcessId(data.process_id)

      // Create blob URLs for the video and script
      const videoBlob = new Blob([data.data], { type: 'video/mp4' })
      const videoUrl = URL.createObjectURL(videoBlob)
      
      setResult({
        script: data.script,
        videoUrl
      })
    } catch (error) {
      console.error('Error:', error)
      setIsProcessing(false)
    }
  }

  const handleDownloadScript = () => {
    if (!result?.script) return

    const blob = new Blob([result.script], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'generated-script.txt'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleDownloadVideo = () => {
    if (!result?.videoUrl) return

    const a = document.createElement('a')
    a.href = result.videoUrl
    a.download = 'generated-video.mp4'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-6 py-16">
        <div className="flex justify-end mb-4">
          <ThemeToggle />
        </div>
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-primary via-accent-purple to-primary bg-clip-text text-transparent mb-4">
            AI Video Slicer & Recomposer
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Transform your videos with AI-powered editing. Upload multiple videos and let our intelligent system create compelling content.
          </p>
        </div>

      {!result ? (
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Panel - Script Generation */}
            <Card className="border border-border shadow-lg bg-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl font-semibold text-card-foreground">Script Generation</CardTitle>
              </CardHeader>
              <CardContent>
                <YouTubeScriptPanel
                  onScriptGenerated={handleScriptGenerated}
                  isProcessing={isProcessing}
                />
              </CardContent>
            </Card>

            {/* Right Panel - Video Input */}
            <Card className="border border-border shadow-lg bg-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl font-semibold text-card-foreground">Video Input</CardTitle>
              </CardHeader>
              <CardContent>
                {!generatedScript ? (
                  <div className="text-center py-8 text-slate-500">
                    <p>Generate a script first to unlock video upload</p>
                  </div>
                ) : (
                  <>
                    <UploadZone onFilesAccepted={handleFilesAccepted} />
                    
                    {videos.length > 0 && (
                      <div className="mt-6 space-y-3">
                        <h3 className="font-semibold text-slate-800">Uploaded Videos</h3>
                        {videos.map((video, index) => (
                          <div
                            key={index}
                            className="flex items-center justify-between p-4 rounded-xl bg-slate-50/80 border border-slate-200/50 hover:bg-slate-100/80 transition-all duration-200"
                          >
                            <div>
                              <p className="font-semibold text-slate-800">{video.name}</p>
                              <p className="text-sm text-slate-500">
                                {(video.size / (1024 * 1024)).toFixed(2)} MB
                              </p>
                            </div>
                            <Button
                              variant="destructive"
                              size="sm"
                              onClick={() => setVideos(prev => prev.filter((_, i) => i !== index))}
                            >
                              Remove
                            </Button>
                          </div>
                        ))}

                        {videos.length >= 2 && (
                          <Button
                            className="w-full mt-6 h-12 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
                            disabled={isProcessing}
                            onClick={() => handleGenerate(generatedScript || '')}
                          >
                            Generate Edited Video
                          </Button>
                        )}
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {processId && (
            <div className="mt-8">
              <ProcessingStatus processId={processId} />
            </div>
          )}
        </div>
      ) : (
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card className="border border-border shadow-lg bg-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl font-semibold text-card-foreground">Generated Script</CardTitle>
              </CardHeader>
              <CardContent>
                <ScriptPanel
                  script={result.script}
                  onDownload={handleDownloadScript}
                />
              </CardContent>
            </Card>
            <Card className="border border-border shadow-lg bg-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl font-semibold text-card-foreground">Video Preview</CardTitle>
              </CardHeader>
              <CardContent>
                <VideoPreview
                  videoUrl={result.videoUrl}
                  onDownload={handleDownloadVideo}
                />
              </CardContent>
            </Card>
          </div>
        </div>
      )}
      </div>
    </div>
  );
} 