'use client'

import { useState } from 'react'
import { useKeyboardShortcuts, createCommonShortcuts } from '../hooks/useKeyboardShortcuts'
import { UploadZone } from '../components/UploadZone'
import { ProcessingStatus } from '../components/ProcessingStatus'
import { VideoPreview } from '../components/VideoPreview'
import { ScriptBuilder, type ScriptSession } from '../components/script-builder'
import { WorkflowProgress } from '../components/WorkflowProgress'
import { SettingsPanel } from '../components/settings'
import { ThemeToggle } from '../components/theme-toggle'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { ArrowLeft, Sparkles, Video, FileText, Settings, Youtube } from 'lucide-react'

type WorkflowStep = 'welcome' | 'script-building' | 'video-upload' | 'processing' | 'results' | 'settings'

export default function Home() {
  // Main workflow state
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('welcome')
  
  // Keyboard shortcuts
  const shortcuts = createCommonShortcuts({
    settings: () => setCurrentStep('settings'),
    escape: () => {
      if (currentStep === 'settings') {
        setCurrentStep('welcome')
      }
    },
    newItem: () => {
      if (currentStep === 'results' || currentStep === 'processing') {
        setCurrentStep('welcome')
        setFinalizedSession(null)
        setVideos([])
        setResult(null)
        setProcessId(null)
        setIsProcessing(false)
      }
    }
  })

  useKeyboardShortcuts({ shortcuts })
  
  // Script building state
  const [finalizedSession, setFinalizedSession] = useState<ScriptSession | null>(null)
  
  // Video processing state
  const [videos, setVideos] = useState<File[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [processId, setProcessId] = useState<string | null>(null)
  const [result, setResult] = useState<{
    script: string;
    videoUrl: string;
    assemblyType?: string;
    metadata?: any;
    stats?: any;
  } | null>(null)
  const [useAdvancedAssembly, setUseAdvancedAssembly] = useState(true)

  const handleFilesAccepted = (files: File[]) => {
    setVideos(files)
  }

  const handleScriptFinalized = (session: ScriptSession) => {
    setFinalizedSession(session)
    setCurrentStep('video-upload')
  }

  const handleGenerate = async () => {
    if (videos.length < 2 || !finalizedSession) return

    setIsProcessing(true)
    setProcessId(null)
    setResult(null)
    setCurrentStep('processing')

    try {
      const formData = new FormData()
      videos.forEach(video => formData.append('videos', video))
      formData.append('prompt', finalizedSession.currentScript)
      formData.append('use_base_prompt', 'false')
      formData.append('use_advanced_assembly', String(useAdvancedAssembly))

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
        script: data.script || finalizedSession.currentScript,
        videoUrl,
        assemblyType: data.assembly_type,
        metadata: data.metadata,
        stats: data.stats
      })
      
      setCurrentStep('results')
    } catch (error) {
      console.error('Error:', error)
      setIsProcessing(false)
      // Stay on processing step to show error
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

  const handleBackToScriptBuilder = () => {
    setCurrentStep('script-building')
  }

  const handleStartOver = () => {
    setCurrentStep('welcome')
    setFinalizedSession(null)
    setVideos([])
    setResult(null)
    setProcessId(null)
    setIsProcessing(false)
  }

  const renderWelcome = () => (
    <div className="max-w-4xl mx-auto text-center space-y-8">
      <div className="space-y-4">
        <h1 className="text-5xl font-bold bg-gradient-to-r from-primary via-accent-purple to-primary bg-clip-text text-transparent">
          AI Video Slicer & Recomposer
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Transform YouTube videos with AI-powered editing. Interactive mode: Build your script interactively from any YouTube URL, then let our intelligent system create compelling content.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
        <Card className="relative overflow-hidden border-2 hover:border-primary/50 transition-all duration-300 hover:scale-105">
          <CardContent className="p-6 text-center">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
              <Youtube className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">1. YouTube Script Builder</h3>
            <p className="text-muted-foreground text-sm">
              Paste any YouTube URL and let AI extract, analyze, and transform content into optimized scripts.
            </p>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden border-2 hover:border-primary/50 transition-all duration-300 hover:scale-105">
          <CardContent className="p-6 text-center">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-green-100 dark:bg-green-900 flex items-center justify-center">
              <Video className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">2. Upload Videos</h3>
            <p className="text-muted-foreground text-sm">
              Upload your source videos. Our AI will analyze and match scenes to your script.
            </p>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden border-2 hover:border-primary/50 transition-all duration-300 hover:scale-105">
          <CardContent className="p-6 text-center">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-purple-100 dark:bg-purple-900 flex items-center justify-center">
              <Sparkles className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 className="text-lg font-semibold mb-2">3. AI Assembly</h3>
            <p className="text-muted-foreground text-sm">
              Watch as AI intelligently assembles your videos based on your script and content analysis.
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="pt-8">
        <Button
          onClick={() => setCurrentStep('script-building')}
          size="lg"
          className="px-8 py-4 text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
        >
          Start Building Your Script
        </Button>
      </div>
    </div>
  )

  const renderScriptBuilding = () => (
    <ScriptBuilder
      onFinalize={handleScriptFinalized}
      onBack={() => setCurrentStep('welcome')}
      className="animate-in slide-in-from-right-4 duration-500"
    />
  )

  const renderVideoUpload = () => (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header with back button */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          onClick={handleBackToScriptBuilder}
          className="flex items-center gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Script
        </Button>
        <div>
          <h2 className="text-2xl font-bold text-foreground">Upload Your Videos</h2>
          <p className="text-muted-foreground">Your script is ready! Now upload the videos you want to edit.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Panel - Script Summary */}
        <Card className="border border-border shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Finalized Script
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-foreground">{finalizedSession?.wordCount || 0}</div>
                <div className="text-sm text-muted-foreground">Words</div>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-foreground">{Math.ceil((finalizedSession?.wordCount || 0) / 250)}</div>
                <div className="text-sm text-muted-foreground">Est. Minutes</div>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 max-h-64 overflow-y-auto">
              <pre className="text-sm whitespace-pre-wrap text-foreground">
                {finalizedSession?.currentScript?.substring(0, 500) || 'No script content'}
                {(finalizedSession?.currentScript?.length || 0) > 500 && '...'}
              </pre>
            </div>
            
            <Button
              variant="outline"
              onClick={handleBackToScriptBuilder}
              className="w-full"
            >
              Modify Script
            </Button>
          </CardContent>
        </Card>

        {/* Right Panel - Video Upload */}
        <Card className="border border-border shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="h-5 w-5" />
              Video Upload
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <UploadZone onFilesAccepted={handleFilesAccepted} />
            
            {videos.length > 0 && (
              <div className="space-y-3">
                <h3 className="font-semibold text-foreground">Uploaded Videos ({videos.length})</h3>
                {videos.map((video, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border"
                  >
                    <div>
                      <p className="font-medium text-foreground">{video.name}</p>
                      <p className="text-sm text-muted-foreground">
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
                  <div className="space-y-4 pt-4 border-t">
                    <div className="flex items-center justify-between p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border">
                      <div>
                        <h4 className="font-semibold text-foreground">Assembly Mode</h4>
                        <p className="text-sm text-muted-foreground">
                          {useAdvancedAssembly 
                            ? "AI-powered intelligent scene matching and assembly" 
                            : "Simple video concatenation"}
                        </p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={useAdvancedAssembly}
                          onChange={(e) => setUseAdvancedAssembly(e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                        <span className="ml-3 text-sm font-medium text-foreground">
                          {useAdvancedAssembly ? 'Advanced' : 'Simple'}
                        </span>
                      </label>
                    </div>
                    
                    <Button
                      onClick={handleGenerate}
                      disabled={isProcessing}
                      className="w-full h-12 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
                    >
                      <Sparkles className="h-4 w-4 mr-2" />
                      Generate AI-Edited Video
                    </Button>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )

  const renderProcessing = () => (
    <div className="max-w-4xl mx-auto">
      <Card className="border border-border shadow-lg">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Processing Your Video</CardTitle>
          <p className="text-muted-foreground">
            Our AI is analyzing your script and assembling your videos. This may take a few minutes.
          </p>
        </CardHeader>
        <CardContent>
          {processId ? (
            <ProcessingStatus processId={processId} />
          ) : (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">Initializing video processing...</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )

  const renderResults = () => (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold text-foreground">Your Video is Ready! 🎉</h2>
        <p className="text-muted-foreground">
          AI has successfully assembled your video based on your script. Review and download below.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Script Panel */}
        <Card className="border border-border shadow-lg">
          <CardHeader>
            <CardTitle className="text-xl font-semibold">Final Script</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 max-h-64 overflow-y-auto">
                <pre className="text-sm whitespace-pre-wrap text-foreground">
                  {result?.script || 'No script available'}
                </pre>
              </div>
              <Button onClick={handleDownloadScript} className="w-full">
                Download Script
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Video Panel */}
        <Card className="border border-border shadow-lg">
          <CardHeader>
            <CardTitle className="text-xl font-semibold">Generated Video</CardTitle>
            {result?.assemblyType && (
              <div className="flex items-center gap-2 mt-2">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  result.assemblyType === 'advanced' 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                    : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                }`}>
                  {result.assemblyType === 'advanced' ? '🤖 Advanced Assembly' : '🔧 Simple Assembly'}
                </span>
              </div>
            )}
          </CardHeader>
          <CardContent>
            {result?.videoUrl && (
              <VideoPreview
                videoUrl={result.videoUrl}
                onDownload={handleDownloadVideo}
              />
            )}
            
            {result?.stats && result.assemblyType === 'advanced' && (
              <div className="mt-4 p-4 rounded-lg bg-gray-50 dark:bg-gray-800 border">
                <h4 className="font-semibold text-foreground mb-2">Assembly Statistics</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>Characters Found: <span className="font-medium">{result.stats.characters_found}</span></div>
                  <div>Face Registry: <span className="font-medium">{result.stats.face_registry_entities}</span></div>
                  <div>Video Scenes: <span className="font-medium">{result.stats.video_scenes_matched}</span></div>
                  <div>Assembly Plan: <span className="font-medium">{result.stats.assembly_plan_length}</span></div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Actions */}
      <div className="flex justify-center space-x-4">
        <Button
          variant="outline"
          onClick={handleStartOver}
          className="px-6"
        >
          Create Another Video
        </Button>
        <Button
          onClick={handleBackToScriptBuilder}
          className="px-6"
        >
          Refine Script & Regenerate
        </Button>
      </div>
    </div>
  )



  const renderSettings = () => (
    <SettingsPanel 
      onClose={() => setCurrentStep('welcome')}
      className="animate-in slide-in-from-right-4 duration-500"
    />
  )

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-6 py-16">
        {/* Header Actions */}
        <div className="flex items-center justify-end gap-3 mb-6">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setCurrentStep('settings')}
            className="flex items-center gap-2"
          >
            <Settings className="h-4 w-4" />
            Settings
          </Button>
          <ThemeToggle />
        </div>

        {/* Workflow Progress */}
        <WorkflowProgress currentStep={currentStep} className="mb-8" />

        {/* Main Content */}
        {currentStep === 'welcome' && renderWelcome()}
        {currentStep === 'script-building' && renderScriptBuilding()}
        {currentStep === 'video-upload' && renderVideoUpload()}
        {currentStep === 'processing' && renderProcessing()}
        {currentStep === 'results' && renderResults()}
        {currentStep === 'settings' && renderSettings()}
      </div>
    </div>
  )
} 