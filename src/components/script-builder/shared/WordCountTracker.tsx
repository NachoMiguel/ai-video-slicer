'use client';

import { Card, CardContent } from '../../ui/card';
import { Target, TrendingUp, Clock, CheckCircle, AlertTriangle } from 'lucide-react';

interface WordCountTrackerProps {
  currentWords: number;
  targetWords: number;
  estimatedMinutes?: number;
  sectionsCompleted?: number;
  totalSections?: number;
  className?: string;
}

export function WordCountTracker({
  currentWords,
  targetWords,
  estimatedMinutes,
  sectionsCompleted = 0,
  totalSections = 1,
  className = ''
}: WordCountTrackerProps) {
  const progress = Math.min((currentWords / targetWords) * 100, 100);
  const isComplete = currentWords >= targetWords;
  const isNearTarget = progress >= 80;
  const sectionProgress = (sectionsCompleted / totalSections) * 100;

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

  const getEstimatedReadingTime = () => {
    const wordsPerMinute = 180;
    const minutes = Math.ceil(currentWords / wordsPerMinute);
    return minutes;
  };

  return (
    <Card className={`${className}`}>
      <CardContent className="p-4">
        <div className="space-y-4">
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

          {/* Main Progress */}
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
            
            {/* Progress Percentage */}
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

          {/* Section Progress */}
          {totalSections > 1 && (
            <div className="space-y-2 pt-2 border-t border-border">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Sections</span>
                <span className="text-sm font-medium text-foreground">
                  {sectionsCompleted} / {totalSections}
                </span>
              </div>
              
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-purple-500 transition-all duration-300"
                  style={{ width: `${sectionProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Additional Metrics */}
          <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-sm font-medium text-foreground">
                  {estimatedMinutes || getEstimatedReadingTime()} min
                </div>
                <div className="text-xs text-muted-foreground">Reading time</div>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-sm font-medium text-foreground">
                  {currentWords > 0 ? Math.round(currentWords / Math.max(sectionsCompleted, 1)) : 0}
                </div>
                <div className="text-xs text-muted-foreground">Avg per section</div>
              </div>
            </div>
          </div>

          {/* Status Messages */}
          {progress > 110 && (
            <div className="flex items-center gap-2 p-2 bg-amber-50 dark:bg-amber-950 rounded-lg border border-amber-200 dark:border-amber-800">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <span className="text-xs text-amber-700 dark:text-amber-300">
                Script is significantly over target length. Consider condensing content.
              </span>
            </div>
          )}

          {progress < 50 && currentWords > 0 && (
            <div className="flex items-center gap-2 p-2 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
              <Target className="h-4 w-4 text-blue-600" />
              <span className="text-xs text-blue-700 dark:text-blue-300">
                Keep writing! You're making good progress toward your target.
              </span>
            </div>
          )}

          {isComplete && !isNearTarget && (
            <div className="flex items-center gap-2 p-2 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-800">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-xs text-green-700 dark:text-green-300">
                Excellent! You've reached your target word count. Ready to refine and finalize.
              </span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 