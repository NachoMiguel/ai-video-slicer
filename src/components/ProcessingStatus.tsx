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

  useEffect(() => {
    console.log(`[WS] Connecting to WebSocket for process ${processId}`)
    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/${processId}`)

    ws.onopen = () => {
      console.log(`[WS] Connected successfully for process ${processId}`)
    }

    ws.onmessage = (event) => {
      console.log(`[WS] Received message:`, event.data)
      try {
        const data = JSON.parse(event.data)
        console.log(`[WS] Parsed data:`, data)
        setStatus(data)
        
        if (data.step === 'error') {
          setError(data.message)
          if (onComplete) {
            onComplete(false, { error: data.message })
          }
        } else if (data.step === 'completed' && data.success) {
          if (onComplete) {
            onComplete(true, data.result)
          }
        }
      } catch (err) {
        console.error(`[WS] Failed to parse message:`, err)
      }
    }

    ws.onerror = (error) => {
      console.error(`[WS] WebSocket error:`, error)
      setError('Connection error. Please try again.')
      if (onComplete) {
        onComplete(false, { error: 'Connection error. Please try again.' })
      }
    }

    ws.onclose = (event) => {
      console.log(`[WS] Connection closed:`, event.code, event.reason)
    }

    return () => {
      console.log(`[WS] Cleaning up WebSocket connection`)
      ws.close()
    }
  }, [processId, onComplete])

  if (error) {
    return (
      <div className="panel">
        <div className="rounded-lg bg-red-50 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm font-medium text-red-800">{error}</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!status) {
    return (
      <div className="panel">
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Connecting to progress updates...</p>
        </div>
      </div>
    )
  }

  const getStepNumber = (step: string): number => {
    const steps = ['upload', 'transcribe', 'detect_scenes', 'generate_script', 'process_video']
    return steps.indexOf(step) + 1
  }

  // Show completion message if processing is done
  if (status?.step === 'completed' && status?.success) {
    return (
      <div className="panel">
        <div className="rounded-lg bg-green-50 p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-green-800">Processing Complete!</h3>
              <div className="mt-2 text-sm text-green-700">
                <p>Your video has been generated successfully.</p>
                {status.result?.filename && (
                  <p className="mt-1">
                    <strong>Saved to Downloads:</strong> {status.result.filename}
                  </p>
                )}
                {status.result?.downloads_path && (
                  <p className="mt-1 text-xs text-green-600 break-all">
                    {status.result.downloads_path}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="panel">
      <ProgressIndicator
        currentStep={getStepNumber(status.step)}
        totalSteps={5}
        status={status.message}
      />
    </div>
  )
} 