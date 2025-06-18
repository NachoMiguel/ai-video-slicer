'use client';

import { CheckCircle, Circle, FileText, Video, Sparkles, Download } from 'lucide-react';

interface WorkflowStep {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface WorkflowProgressProps {
  currentStep: 'welcome' | 'script-building' | 'video-upload' | 'processing' | 'results' | 'legacy' | 'settings';
  className?: string;
}

const steps: WorkflowStep[] = [
  {
    id: 'script-building',
    title: 'Build Script',
    description: 'Create and refine your script with AI assistance',
    icon: FileText
  },
  {
    id: 'video-upload',
    title: 'Upload Videos',
    description: 'Upload your source videos for editing',
    icon: Video
  },
  {
    id: 'processing',
    title: 'AI Processing',
    description: 'AI analyzes and assembles your content',
    icon: Sparkles
  },
  {
    id: 'results',
    title: 'Download',
    description: 'Review and download your finished video',
    icon: Download
  }
];

export function WorkflowProgress({ currentStep, className = '' }: WorkflowProgressProps) {
  if (currentStep === 'welcome' || currentStep === 'legacy' || currentStep === 'settings') {
    return null; // Don't show progress on welcome screen, legacy mode, or settings
  }

  const getStepStatus = (stepId: string) => {
    const stepIndex = steps.findIndex(step => step.id === stepId);
    const currentIndex = steps.findIndex(step => step.id === currentStep);
    
    if (stepIndex < currentIndex) return 'completed';
    if (stepIndex === currentIndex) return 'current';
    return 'upcoming';
  };

  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between max-w-4xl mx-auto">
        {steps.map((step, index) => {
          const status = getStepStatus(step.id);
          const IconComponent = step.icon;
          const isLast = index === steps.length - 1;

          return (
            <div key={step.id} className="flex items-center flex-1">
              {/* Step Circle */}
              <div className="flex flex-col items-center">
                <div className={`relative flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300 ${
                  status === 'completed'
                    ? 'bg-green-500 border-green-500 text-white'
                    : status === 'current'
                    ? 'bg-primary border-primary text-primary-foreground'
                    : 'bg-background border-gray-300 text-gray-400'
                }`}>
                  {status === 'completed' ? (
                    <CheckCircle className="h-6 w-6" />
                  ) : (
                    <IconComponent className="h-6 w-6" />
                  )}
                </div>
                
                {/* Step Info */}
                <div className="mt-2 text-center">
                  <div className={`text-sm font-medium ${
                    status === 'current' 
                      ? 'text-foreground' 
                      : status === 'completed'
                      ? 'text-green-600'
                      : 'text-muted-foreground'
                  }`}>
                    {step.title}
                  </div>
                  <div className="text-xs text-muted-foreground max-w-[120px] mt-1">
                    {step.description}
                  </div>
                </div>
              </div>

              {/* Connector Line */}
              {!isLast && (
                <div className="flex-1 mx-4 mb-8">
                  <div className={`h-0.5 transition-all duration-300 ${
                    status === 'completed' 
                      ? 'bg-green-500' 
                      : 'bg-gray-300'
                  }`} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
} 