interface ProgressIndicatorProps {
  currentStep: number;
  totalSteps: number;
  status: string;
}

export function ProgressIndicator({ currentStep, totalSteps, status }: ProgressIndicatorProps) {
  const percentage = (currentStep / totalSteps) * 100

  return (
    <div className="w-full max-w-2xl mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-700">{status}</span>
        <span className="text-sm text-gray-500">{Math.round(percentage)}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div 
          className="h-full bg-accent-blue transition-all duration-300" 
          style={{ width: `${percentage}%` }} 
        />
      </div>
    </div>
  )
} 