'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Sparkles, Zap, Settings, Info } from 'lucide-react';

interface AppModeToggleProps {
  mode: 'interactive' | 'legacy';
  onModeChange: (mode: 'interactive' | 'legacy') => void;
  className?: string;
}

export function AppModeToggle({ mode, onModeChange, className = '' }: AppModeToggleProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <Card className={`border border-border ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Settings className="h-5 w-5" />
            App Mode
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDetails(!showDetails)}
            className="text-muted-foreground hover:text-foreground"
          >
            <Info className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${
              mode === 'interactive' 
                ? 'bg-primary text-primary-foreground' 
                : 'bg-gray-100 dark:bg-gray-800 text-muted-foreground'
            }`}>
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <div className="font-medium text-foreground">
                {mode === 'interactive' ? 'Interactive Mode' : 'Legacy Mode'}
              </div>
              <div className="text-sm text-muted-foreground">
                {mode === 'interactive' 
                  ? 'AI-guided script building with human oversight'
                  : 'Traditional workflow with automated processing'
                }
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <span className={`text-sm ${mode === 'legacy' ? 'text-foreground' : 'text-muted-foreground'}`}>
              Legacy
            </span>
            <Switch
              checked={mode === 'interactive'}
              onCheckedChange={(checked: boolean) => onModeChange(checked ? 'interactive' : 'legacy')}
            />
            <span className={`text-sm ${mode === 'interactive' ? 'text-foreground' : 'text-muted-foreground'}`}>
              Interactive
            </span>
          </div>
        </div>

        {showDetails && (
          <div className="space-y-3 pt-3 border-t border-border">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <span className="font-medium text-foreground">Interactive Mode (Recommended)</span>
              </div>
              <ul className="text-sm text-muted-foreground space-y-1 ml-6">
                <li>• Step-by-step guided workflow</li>
                <li>• Interactive script building with AI chat</li>
                <li>• Real-time progress tracking and feedback</li>
                <li>• Human oversight at every step</li>
                <li>• Better for long-form content creation</li>
              </ul>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-yellow-500" />
                <span className="font-medium text-foreground">Legacy Mode</span>
              </div>
              <ul className="text-sm text-muted-foreground space-y-1 ml-6">
                <li>• Original single-step workflow</li>
                <li>• Automated script generation from YouTube</li>
                <li>• Direct video processing</li>
                <li>• Faster for simple use cases</li>
                <li>• Less control over output</li>
              </ul>
            </div>
            
            <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-700 dark:text-blue-300">
                <strong>New to AI Video Slicer?</strong> Try Interactive Mode for the best experience. 
                It provides better control and higher quality results.
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 