'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { Youtube, Upload, Clock, Star, Zap } from 'lucide-react';

interface EntryMethod {
  id: 'youtube' | 'upload';
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  features: string[];
  estimatedTime: string;
  difficulty: 'Easy' | 'Medium';
  badge?: string;
}

interface EntryMethodSelectorProps {
  onMethodSelected: (method: 'youtube' | 'upload') => void;
  isLoading?: boolean;
}

const entryMethods: EntryMethod[] = [
  {
    id: 'youtube',
    title: 'YouTube Video',
    description: 'Extract and rewrite script from a YouTube video using AI',
    icon: Youtube,
    features: [
      'Automatic transcript extraction',
      'AI-powered script generation', 
      'Optimized for video content',
      'Built-in engagement optimization'
    ],
    estimatedTime: '5-10 minutes',
    difficulty: 'Easy',
    badge: 'Most Popular'
  },
  {
    id: 'upload',
    title: 'Upload Script',
    description: 'Upload your existing script for AI-powered analysis and refinement',
    icon: Upload,
    features: [
      'Comprehensive script analysis',
      'AI-powered improvement suggestions',
      'Structure optimization',
      'Readability enhancement'
    ],
    estimatedTime: '2-5 minutes',
    difficulty: 'Easy'
  }
];

export function EntryMethodSelector({ onMethodSelected, isLoading = false }: EntryMethodSelectorProps) {
  const [selectedMethod, setSelectedMethod] = useState<'youtube' | 'upload' | null>(null);
  
  const handleMethodClick = (methodId: 'youtube' | 'upload') => {
    setSelectedMethod(methodId);
  };
  
  const handleContinue = () => {
    if (selectedMethod) {
      onMethodSelected(selectedMethod);
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold text-foreground">
          Choose Your Starting Point
        </h2>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Select how you'd like to begin building your script. Both methods lead to the same powerful interactive editor.
        </p>
      </div>

      {/* Method Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {entryMethods.map((method) => {
          const IconComponent = method.icon;
          const isSelected = selectedMethod === method.id;
          
          return (
            <Card 
              key={method.id}
              className={`relative cursor-pointer transition-all duration-300 hover:shadow-lg hover:scale-[1.02] ${
                isSelected 
                  ? 'ring-2 ring-primary shadow-lg bg-primary/5' 
                  : 'hover:bg-accent/50'
              }`}
              onClick={() => handleMethodClick(method.id)}
            >
              {/* Badge */}
              {method.badge && (
                <div className="absolute -top-3 left-6 z-10">
                  <div className="bg-primary text-primary-foreground px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1">
                    <Star className="h-3 w-3" />
                    {method.badge}
                  </div>
                </div>
              )}
              
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-3 rounded-lg ${
                      isSelected ? 'bg-primary text-primary-foreground' : 'bg-accent text-accent-foreground'
                    }`}>
                      <IconComponent className="h-6 w-6" />
                    </div>
                    <div>
                      <CardTitle className="text-xl">{method.title}</CardTitle>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          method.difficulty === 'Easy' 
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        }`}>
                          {method.difficulty}
                        </span>
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <Clock className="h-3 w-3" />
                          <span className="text-xs">{method.estimatedTime}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Selection indicator */}
                  <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all ${
                    isSelected 
                      ? 'border-primary bg-primary' 
                      : 'border-muted-foreground/30'
                  }`}>
                    {isSelected && (
                      <div className="w-2 h-2 bg-primary-foreground rounded-full" />
                    )}
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <p className="text-muted-foreground">{method.description}</p>
                
                {/* Features List */}
                <div className="space-y-2">
                  <h4 className="font-medium text-sm text-foreground">Key Features:</h4>
                  <ul className="space-y-1">
                    {method.features.map((feature, index) => (
                      <li key={index} className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Zap className="h-3 w-3 text-primary flex-shrink-0" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
                
                {/* Preview of next steps */}
                <div className="pt-2 border-t border-border">
                  <p className="text-xs text-muted-foreground">
                    Next: {method.id === 'youtube' 
                      ? 'Enter YouTube URL → Extract transcript → Interactive building'
                      : 'Upload script → AI analysis → Interactive refinement'
                    }
                  </p>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Continue Button */}
      <div className="flex justify-center pt-4">
        <Button
          onClick={handleContinue}
          disabled={!selectedMethod || isLoading}
          size="lg"
          className="px-8 py-3 text-lg font-medium"
        >
          {isLoading ? (
            <div className="flex items-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Starting...</span>
            </div>
          ) : selectedMethod ? (
            `Continue with ${entryMethods.find(m => m.id === selectedMethod)?.title}`
          ) : (
            'Select a method to continue'
          )}
        </Button>
      </div>

      {/* Info Section */}
      <div className="text-center pt-6 border-t border-border">
        <p className="text-sm text-muted-foreground">
          Both methods lead to the same powerful interactive script editor where you can refine your content section by section.
        </p>
      </div>
    </div>
  );
} 