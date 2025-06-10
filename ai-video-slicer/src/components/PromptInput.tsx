import { useState } from 'react'

interface PromptInputProps {
  onSubmit: (prompt: string) => void
  isProcessing: boolean
}

export function PromptInput({ onSubmit, isProcessing }: PromptInputProps) {
  const [prompt, setPrompt] = useState('')
  const [isCustom, setIsCustom] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (prompt.trim()) {
      onSubmit(prompt.trim())
    }
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium text-gray-900">Choose or Enter a Prompt</h3>
      
      {!isCustom ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={() => {
              setPrompt("Use base prompt")
              onSubmit("Use base prompt")
            }}
            disabled={isProcessing}
            className="btn-secondary p-4 text-left"
          >
            Use Base Prompt
          </button>
          <button
            onClick={() => setIsCustom(true)}
            disabled={isProcessing}
            className="btn-secondary p-4 text-left"
          >
            + Custom Prompt
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your custom prompt..."
            disabled={isProcessing}
            className="w-full p-4 rounded-lg border border-gray-200 resize-none
                     focus:outline-none focus:ring-2 focus:ring-accent-blue
                     min-h-[100px] disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <div className="flex gap-4">
            <button
              type="submit"
              disabled={!prompt.trim() || isProcessing}
              className="btn-primary"
            >
              {isProcessing ? "Processing..." : "Generate"}
            </button>
            <button
              type="button"
              onClick={() => setIsCustom(false)}
              disabled={isProcessing}
              className="btn-secondary"
            >
              Back to Options
            </button>
          </div>
        </form>
      )}
    </div>
  )
} 