interface ProgressIndicatorProps {
  currentStep: number;
  totalSteps: number;
  status: string;
  progress?: number; // Optional direct progress value (0-100)
}

export function ProgressIndicator({ 
  currentStep, 
  totalSteps, 
  status,
  progress 
}: ProgressIndicatorProps) {
  // Use direct progress value if provided, otherwise calculate from steps
  const percentage = progress !== undefined ? progress : (currentStep / totalSteps) * 100
  const clampedPercentage = Math.min(Math.max(percentage, 0), 100)
  
  // Calculate circle properties
  const radius = 60
  const strokeWidth = 8
  const normalizedRadius = radius - strokeWidth * 2
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (clampedPercentage / 100) * circumference

  // Get color based on progress
  const getProgressColor = (percent: number) => {
    if (percent < 25) return '#3b82f6' // blue-500
    if (percent < 50) return '#8b5cf6' // violet-500
    if (percent < 75) return '#06b6d4' // cyan-500
    return '#10b981' // emerald-500
  }

  const progressColor = getProgressColor(clampedPercentage)

  return (
    <div className="w-full max-w-md mx-auto space-y-6">
      {/* Circular Progress */}
      <div className="flex flex-col items-center space-y-4">
        <div className="relative">
          {/* Background circle */}
          <svg
            className="transform -rotate-90 w-32 h-32"
            width="120"
            height="120"
          >
            <circle
              stroke="currentColor"
              fill="transparent"
              strokeWidth={strokeWidth}
              r={normalizedRadius}
              cx="60"
              cy="60"
              className="text-gray-200 dark:text-gray-700"
            />
            {/* Progress circle */}
            <circle
              stroke={progressColor}
              fill="transparent"
              strokeWidth={strokeWidth}
              strokeDasharray={circumference + ' ' + circumference}
              style={{ strokeDashoffset }}
              strokeLinecap="round"
              r={normalizedRadius}
              cx="60"
              cy="60"
              className="transition-all duration-500 ease-in-out drop-shadow-sm"
            />
          </svg>
          
          {/* Center percentage */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div 
                className="text-2xl font-bold transition-colors duration-300"
                style={{ color: progressColor }}
              >
                {Math.round(clampedPercentage)}%
              </div>
              <div className="text-xs text-muted-foreground">
                {currentStep}/{totalSteps}
              </div>
            </div>
          </div>
        </div>

        {/* Status Message */}
        <div className="text-center space-y-2">
          <p className="text-sm font-medium text-foreground leading-relaxed">
            {status}
          </p>
          
          {/* Progress dots animation */}
          <div className="flex justify-center space-x-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-current opacity-30 animate-pulse"
                style={{
                  animationDelay: `${i * 0.2}s`,
                  animationDuration: '1.5s',
                  color: progressColor
                }}
              />
            ))}
          </div>
        </div>

        {/* Step Indicators */}
        <div className="w-full max-w-xs">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span className={currentStep >= 1 ? 'text-foreground font-medium' : ''}>
              Start
            </span>
            <span className={currentStep >= Math.ceil(totalSteps / 2) ? 'text-foreground font-medium' : ''}>
              Processing
            </span>
            <span className={currentStep >= totalSteps ? 'text-foreground font-medium' : ''}>
              Complete
            </span>
          </div>
          
          {/* Step progress bar */}
          <div className="mt-2 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full transition-all duration-500 ease-out rounded-full"
              style={{ 
                width: `${clampedPercentage}%`,
                backgroundColor: progressColor
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
} 