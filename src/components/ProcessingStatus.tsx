import { useEffect, useState } from 'react'
import { ProgressIndicator } from './ProgressIndicator'

interface ProcessingStatusProps {
  processId: string
  onComplete?: (success: boolean, data?: any) => void
}

interface Status {
  step: string
  progress: number
  message: string
  success?: boolean
  result?: any
}

export function ProcessingStatus({ processId, onComplete }: ProcessingStatusProps) {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('connecting')

  useEffect(() => {
    console.log(`[WS-DEBUG] Starting WebSocket connection for process ${processId}`)
    console.log(`[WS-DEBUG] WebSocket URL: ws://127.0.0.1:8000/ws/${processId}`)
    
    setConnectionStatus('connecting')
    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/${processId}`)

    ws.onopen = () => {
      console.log(`[WS-DEBUG] ✅ WebSocket connected successfully for process ${processId}`)
      setConnectionStatus('connected')
    }

    ws.onmessage = (event) => {
      console.log(`[WS-DEBUG] 📨 Raw message received:`, event.data)
      try {
        const data = JSON.parse(event.data)
        console.log(`[WS-DEBUG] 📋 Parsed progress data:`, {
          step: data.step,
          progress: data.progress,
          message: data.message,
          success: data.success,
          timestamp: data.timestamp
        })
        
        setStatus(data)
        
        if (data.step === 'error') {
          console.log(`[WS-DEBUG] ❌ Error received:`, data.message)
          setError(data.message)
          setConnectionStatus('error')
          if (onComplete) {
            onComplete(false, { error: data.message })
          }
        } else if (data.step === 'completed' && data.success) {
          console.log(`[WS-DEBUG] ✅ Processing completed successfully`)
          setConnectionStatus('connected')
          if (onComplete) {
            onComplete(true, data.result)
          }
        } else {
          console.log(`[WS-DEBUG] 🔄 Progress update: ${data.progress}% - ${data.step} - ${data.message}`)
        }
      } catch (err) {
        console.error(`[WS-DEBUG] ❌ Failed to parse WebSocket message:`, err)
        console.error(`[WS-DEBUG] Raw message was:`, event.data)
      }
    }

    ws.onerror = (error) => {
      console.error(`[WS-DEBUG] ❌ WebSocket error:`, error)
      setConnectionStatus('error')
      setError('Connection error. Please try again.')
      if (onComplete) {
        onComplete(false, { error: 'Connection error. Please try again.' })
      }
    }

    ws.onclose = (event) => {
      console.log(`[WS-DEBUG] 🔌 WebSocket connection closed:`, {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean
      })
      setConnectionStatus('disconnected')
    }

    return () => {
      console.log(`[WS-DEBUG] 🧹 Cleaning up WebSocket connection for process ${processId}`)
      ws.close()
    }
  }, [processId, onComplete])

  if (error) {
    return (
      <div className="w-full max-w-md mx-auto">
        <div className="rounded-xl bg-red-50 dark:bg-red-900/20 p-6 border border-red-200 dark:border-red-800">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/40 flex items-center justify-center">
                <svg className="w-5 h-5 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
            <div className="ml-4">
              <h3 className="text-sm font-medium text-red-800 dark:text-red-200">Processing Error</h3>
              <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
              <p className="text-xs text-red-600 dark:text-red-400 mt-1">Connection Status: {connectionStatus}</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!status && connectionStatus === 'connecting') {
    return (
      <div className="w-full max-w-md mx-auto">
        <div className="text-center py-8 space-y-4">
          <div className="relative">
            <div className="w-16 h-16 mx-auto">
              <div className="absolute inset-0 rounded-full border-4 border-gray-200 dark:border-gray-700"></div>
              <div className="absolute inset-0 rounded-full border-4 border-t-primary animate-spin"></div>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">Connecting to processing server...</p>
            <p className="text-xs text-muted-foreground">Establishing real-time progress updates</p>
            <p className="text-xs text-gray-500">Connection: {connectionStatus} | Process ID: {processId}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!status && connectionStatus === 'connected') {
    return (
      <div className="w-full max-w-md mx-auto">
        <div className="text-center py-8 space-y-4">
          <div className="relative">
            <div className="w-16 h-16 mx-auto">
              <div className="absolute inset-0 rounded-full border-4 border-gray-200 dark:border-gray-700"></div>
              <div className="absolute inset-0 rounded-full border-4 border-t-green-500 animate-spin"></div>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">Connected! Waiting for processing to start...</p>
            <p className="text-xs text-muted-foreground">Server is initializing your video processing</p>
            <p className="text-xs text-green-600">✅ Connected | Process ID: {processId}</p>
          </div>
        </div>
      </div>
    )
  }

  const getStepNumber = (step: string): number => {
    const stepMap: { [key: string]: number } = {
      'upload': 1,
      'transcribe': 2,
      'detect_scenes': 3,
      'generate_script': 4,
      'process_video': 5,
      'completed': 5
    }
    return stepMap[step] || 1
  }

  const getStepName = (step: string): string => {
    const stepNames: { [key: string]: string } = {
      'upload': 'Uploading Videos',
      'transcribe': 'Transcribing Audio',
      'detect_scenes': 'Analyzing Scenes',
      'generate_script': 'Generating Script',
      'process_video': 'Processing Video',
      'completed': 'Complete'
    }
    return stepNames[step] || step
  }

  // Show completion message if processing is done
  if (status?.step === 'completed' && status?.success) {
    return (
      <div className="w-full max-w-md mx-auto">
        <div className="rounded-xl bg-green-50 dark:bg-green-900/20 p-6 border border-green-200 dark:border-green-800">
          <div className="text-center space-y-4">
            <div className="w-16 h-16 mx-auto rounded-full bg-green-100 dark:bg-green-900/40 flex items-center justify-center">
              <svg className="w-8 h-8 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-green-800 dark:text-green-200">Processing Complete!</h3>
              <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                Your video has been generated successfully.
              </p>
              {status.result?.filename && (
                <p className="text-xs text-green-600 dark:text-green-400 mt-2">
                  <strong>Saved:</strong> {status.result.filename}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      <ProgressIndicator
        currentStep={getStepNumber(status?.step || '')}
        totalSteps={5}
        status={status?.message || getStepName(status?.step || '')}
        progress={status?.progress || 0}
      />
    </div>
  )
} 