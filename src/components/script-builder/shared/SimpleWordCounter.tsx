'use client';

import { Card, CardContent } from '../../ui/card';
import { Target, CheckCircle } from 'lucide-react';

interface SimpleWordCounterProps {
  currentWords: number;
  targetWords: number;
  className?: string;
}

export function SimpleWordCounter({
  currentWords,
  targetWords,
  className = ''
}: SimpleWordCounterProps) {
  const progress = Math.min((currentWords / targetWords) * 100, 100);
  const isComplete = currentWords >= targetWords;
  const isNearTarget = progress >= 80;

  const getProgressColor = () => {
    if (isComplete) return 'bg-green-500';
    if (isNearTarget) return 'bg-yellow-500';
    return 'bg-blue-500';
  };

  const getProgressTextColor = () => {
    if (isComplete) return 'text-green-600';
    if (isNearTarget) return 'text-yellow-600';
    return 'text-blue-600';
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  return (
    <Card className={`${className}`}>
      <CardContent className="p-4">
        <div className="space-y-3">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-foreground">Progress Tracker</span>
            </div>
            {isComplete && (
              <div className="flex items-center gap-1 text-green-600">
                <CheckCircle className="h-4 w-4" />
                <span className="text-xs font-medium">Target Reached!</span>
              </div>
            )}
          </div>

          {/* Word Count Progress */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className={`text-2xl font-bold ${getProgressTextColor()}`}>
                {formatNumber(currentWords)}
              </span>
              <span className="text-sm text-muted-foreground">
                / {formatNumber(targetWords)} words
              </span>
            </div>
            
            {/* Progress Bar */}
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ease-out ${getProgressColor()}`}
                style={{ width: `${progress}%` }}
              />
            </div>
            
            {/* Progress Details */}
            <div className="flex items-center justify-between text-xs">
              <span className={`font-medium ${getProgressTextColor()}`}>
                {progress.toFixed(1)}% complete
              </span>
              <span className="text-muted-foreground">
                {targetWords - currentWords > 0 
                  ? `${formatNumber(targetWords - currentWords)} words remaining`
                  : `${formatNumber(currentWords - targetWords)} words over target`
                }
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 