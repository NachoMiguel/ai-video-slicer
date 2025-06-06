import { useEffect, useState } from 'react'
import { ProgressIndicator } from './ProgressIndicator'

interface ProcessingStatusProps {
  processId: string
}

interface Status {
  step: string
  progress: number
  message: string
}

export function ProcessingStatus({ processId }: ProcessingStatusProps) {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${processId}`)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setStatus(data)
      
      if (data.step === 'error') {
        setError(data.message)
      }
    }

    ws.onerror = (error) => {
      setError('Connection error. Please try again.')
    }

    return () => {
      ws.close()
    }
  }, [processId])

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
    return null
  }

  const getStepNumber = (step: string): number => {
    const steps = ['upload', 'transcribe', 'detect_scenes', 'generate_script', 'process_video']
    return steps.indexOf(step) + 1
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