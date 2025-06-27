'use client'

import { useState } from 'react'
import { UploadZone } from './UploadZone'
import { ProcessingStatus } from './ProcessingStatus'
import { VideoPreview } from './VideoPreview'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { useSettingsStore } from '../stores/settingsStore'
import { Bot, Wrench } from 'lucide-react'

interface LegacyVideoProcessorProps {
  className?: string;
}

export function LegacyVideoProcessor({ className = '' }: LegacyVideoProcessorProps) {
  const { preferences } = useSettingsStore()
  const [videos, setVideos] = useState<File[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [processId, setProcessId] = useState<string | null>(null)
  const [generatedScript, setGeneratedScript] = useState<string | null>(null)
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
      formData.append('use_advanced_assembly', String(useAdvancedAssembly))
      formData.append('skip_character_extraction', String(preferences.skipCharacterExtraction))

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
        videoUrl,
        assemblyType: data.assembly_type,
        metadata: data.metadata,
        stats: data.stats
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
    <div className={`${className}`}>
      {!result ? (
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Panel - Script Generation */}
            <Card className="border border-border shadow-lg bg-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl font-semibold text-card-foreground">Script Generation</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-8 text-muted-foreground">
                  <p>Legacy script generation panel removed.</p>
                  <p className="text-sm mt-2">Please use the new Interactive Script Builder mode.</p>
                </div>
              </CardContent>
            </Card>

            {/* Right Panel - Video Input */}
            <Card className="border border-border shadow-lg bg-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl font-semibold text-card-foreground">Video Input</CardTitle>
              </CardHeader>
              <CardContent>
                {!generatedScript ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <p>Generate a script first to unlock video upload</p>
                  </div>
                ) : (
                  <>
                    <UploadZone onFilesAccepted={handleFilesAccepted} />
                    
                    {videos.length > 0 && (
                      <div className="mt-6 space-y-3">
                        <h3 className="font-semibold text-foreground">Uploaded Videos</h3>
                        {videos.map((video, index) => (
                          <div
                            key={index}
                            className="flex items-center justify-between p-4 rounded-xl bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200"
                          >
                            <div>
                              <p className="font-semibold text-foreground">{video.name}</p>
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
                          <div className="mt-6 space-y-4">
                            <div className="flex items-center justify-between p-4 rounded-xl bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
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
                                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                                <span className="ml-3 text-sm font-medium text-foreground">
                                  {useAdvancedAssembly ? 'Advanced' : 'Simple'}
                                </span>
                              </label>
                            </div>
                            <Button
                              className="w-full h-12 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
                              disabled={isProcessing}
                              onClick={() => handleGenerate(generatedScript || '')}
                            >
                              Generate Edited Video
                            </Button>
                          </div>
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
                <div className="space-y-4">
                  <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 max-h-64 overflow-y-auto overflow-x-hidden">
                    <pre className="text-sm whitespace-pre-wrap text-foreground break-words word-wrap overflow-wrap-anywhere w-full">
                      {result.script}
                    </pre>
                  </div>
                  <Button onClick={handleDownloadScript} className="w-full">
                    Download Script
                  </Button>
                </div>
              </CardContent>
            </Card>
            <Card className="border border-border shadow-lg bg-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl font-semibold text-card-foreground">Video Preview</CardTitle>
                {result.assemblyType && (
                  <div className="flex items-center gap-2 mt-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${
                      result.assemblyType === 'advanced' 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                        : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                    }`}>
                      {result.assemblyType === 'advanced' ? (
                        <>
                          <Bot className="h-3 w-3" />
                          Advanced Assembly
                        </>
                      ) : (
                        <>
                          <Wrench className="h-3 w-3" />
                          Simple Assembly
                        </>
                      )}
                    </span>
                  </div>
                )}
              </CardHeader>
              <CardContent>
                <VideoPreview
                  videoUrl={result.videoUrl}
                  onDownload={handleDownloadVideo}
                />
                {result.stats && result.assemblyType === 'advanced' && (
                  <div className="mt-4 p-4 rounded-xl bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
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
        </div>
      )}
    </div>
  )
} 